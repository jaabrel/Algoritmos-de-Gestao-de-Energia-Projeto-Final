import numpy as np
from numba import jit
import cv2
import time
import h5py
from datetime import datetime
import os
import json
import gymnasium as gym
from gymnasium import spaces
from QLearning import QLearningAgent as RLQLearningAgent


# ======================================================================================
# 1. FUNÇÕES AUXILIARES DE ALTO DESEMPENHO (COMPILAÇÃO JIT COM NUMBA)
# ======================================================================================


@jit(nopython=True)
def update_puffs_numba(
    puffs_array, dt, wind_velocity, meandering_vy, turbulence_intensity, puff_decay_rate, puff_expansion_rate, min_mass
):
    """
    Atualiza o estado físico de todos os pacotes de odor (puffs) para o modelo padrão/OU.
    Esta função é compilada com Numba (nopython) para máxima eficiência temporal.

    Operações por pacote:
    - Advecção pelo vento/corrente média e velocidade transversal de meandro.
    - Perturbação turbulenta estocástica isotrópica (Browniana).
    - Expansão volumétrica do raio proporcional à raiz quadrada da idade do puff.
    - Decaimento exponencial da massa de odor por evaporação/dissipação.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)

    for i in range(num_puffs):
        if puffs_array[i, 3] <= min_mass:
            continue

        # Advecção pelo escoamento médio e meandro transversal
        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 1] += wind_velocity[1] * dt
        puffs_array[i, 1] += meandering_vy * dt

        # Perturbação turbulenta isotrópica para dispersão natural
        turbulence_x = np.random.randn() * turbulence_intensity * sqrt_dt
        turbulence_y = np.random.randn() * turbulence_intensity * sqrt_dt
        puffs_array[i, 0] += turbulence_x
        puffs_array[i, 1] += turbulence_y

        # Expansão radial e decaimento exponencial de massa
        puffs_array[i, 2] = puff_expansion_rate * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= 1.0 - puff_decay_rate * dt
        puffs_array[i, 4] += dt

        # Desativar pacotes que saíram dos limites físicos do domínio para libertar slots
        if puffs_array[i, 0] > 115.0 or puffs_array[i, 0] < -15.0 or puffs_array[i, 1] > 65.0 or puffs_array[i, 1] < -15.0:
            puffs_array[i, 3] = 0.0

    return puffs_array


@jit(nopython=True)
def update_puffs_sinusoid_numba(
    puffs_array, dt, current_time, src_x, src_y, wind_velocity, amplitude, frequency, turbulence_intensity, puff_decay_rate, puff_expansion_rate, min_mass
):
    """
    Atualiza os pacotes de odor para o modelo de pluma sinusoidal meandrada.
    Garante que a pluma emerge de forma contínua e centrada na fonte (src_x, src_y),
    abrindo suavemente a amplitude ondulatória à medida que os pacotes avançam a jusante.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)
    vx = wind_velocity[0] if abs(wind_velocity[0]) > 1e-4 else 0.4
    ramp_L = 5.0
    two_pi_f = 2.0 * np.pi * frequency

    for i in range(num_puffs):
        if puffs_array[i, 3] <= min_mass:
            continue

        # Advecção longitudinal com turbulência
        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 0] += np.random.randn() * turbulence_intensity * sqrt_dt

        # Onda sinusoidal com rampa de transição suave a partir da fonte
        dx = max(0.0, puffs_array[i, 0] - src_x)
        ramp = np.tanh(dx / ramp_L)
        target_center_y = src_y + amplitude * ramp * np.sin(two_pi_f * (current_time - dx / vx))

        # Posição transversal com componente turbulenta
        puffs_array[i, 1] = target_center_y + np.random.randn() * turbulence_intensity * sqrt_dt
        puffs_array[i, 2] = puff_expansion_rate * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= 1.0 - puff_decay_rate * dt
        puffs_array[i, 4] += dt

        # Desativar pacotes que saíram dos limites físicos do domínio para libertar slots
        if puffs_array[i, 0] > 115.0 or puffs_array[i, 0] < -15.0 or puffs_array[i, 1] > 65.0 or puffs_array[i, 1] < -15.0:
            puffs_array[i, 3] = 0.0

    return puffs_array


@jit(nopython=True)
def calculate_concentration_numba(agent_pos, agent_radius, puffs_array):
    """
    Calcula a concentração sensorial no ponto do veículo segundo o modelo Top-Hat
    (concentração uniforme no interior do raio de cada pacote de odor).
    """
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]

    for i in range(puffs_array.shape[0]):
        puff_mass = puffs_array[i, 3]
        if puff_mass < 1e-6:
            continue

        puff_x, puff_y, puff_r = (
            puffs_array[i, 0],
            puffs_array[i, 1],
            puffs_array[i, 2],
        )
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        radii_sum_sq = (agent_radius + puff_r) ** 2
        if dist_sq < radii_sum_sq:
            puff_concentration = (
                puff_mass / (np.pi * puff_r**2) if puff_r > 1e-6 else 0.0
            )
            total_concentration += puff_concentration
    return total_concentration


@jit(nopython=True)
def calculate_concentration_gaussian_numba(agent_pos, puffs_array):
    """
    Calcula a concentração local segundo um perfil Gaussiano contínuo,
    em que cada pacote de odor atua como uma distribuição normal com desvio padrão sigma = r / 2.
    """
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]
    for i in range(puffs_array.shape[0]):
        puff_x, puff_y, puff_r, puff_mass = (
            puffs_array[i, 0],
            puffs_array[i, 1],
            puffs_array[i, 2],
            puffs_array[i, 3],
        )
        if puff_mass < 1e-6:
            continue
        sigma = puff_r / 2.0
        if sigma < 1e-6:
            continue
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        if dist_sq > (puff_r * 3) ** 2:
            continue
        sigma_sq = sigma**2
        concentration_k = (puff_mass / (2 * np.pi * sigma_sq)) * np.exp(
            -dist_sq / (2 * sigma_sq)
        )
        total_concentration += concentration_k
    return total_concentration


def plume_exists_at_x(x, plume, config):
    if plume is None:
        return False
    world_h = config["world_height"]
    agent_config = config.get("agent_config", {})
    conc_thresh = agent_config.get("agent_concentration_threshold", 1.0)
    agent_radius = agent_config.get("agent_radius", 1.0)
    model = config.get("concentration_model", "top-hat")
    
    y_coords = np.linspace(0.0, world_h, 25)
    for y in y_coords:
        pos = np.array([x, y])
        if model == "gaussian":
            conc = calculate_concentration_gaussian_numba(pos, plume.puffs_array)
        else:
            conc = calculate_concentration_numba(pos, agent_radius, plume.puffs_array)
        if conc > conc_thresh:
            return True
    return False


def get_plume_center_y(x, t, plume, config):
    """
    Retorna a linha central transversal (coordenada Y) da pluma na posição longitudinal x
    e instante de simulação t.
    - Para plumas sinusoidais: calcula analiticamente o eixo ondulatório instantâneo.
    - Para plumas lineares ou OU: retorna a coordenada Y da fonte emissora.
    """
    src_x, src_y = config["source_position"]
    is_sinusoid = config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]
    if is_sinusoid:
        sin_cfg = config.get("sinusoid_meander_config", {})
        amplitude = float(sin_cfg.get("amplitude", sin_cfg.get("v_var", 8.0)))
        frequency = float(sin_cfg.get("frequency", 0.018))
        ramp_L = 5.0
        two_pi_f = 2.0 * np.pi * frequency
        vx = config["mean_wind_velocity"][0] if abs(config["mean_wind_velocity"][0]) > 1e-4 else 0.4
        dx = max(0.0, x - src_x)
        ramp = np.tanh(dx / ramp_L)
        return float(src_y + amplitude * ramp * np.sin(two_pi_f * (t - dx / vx)))
    return float(src_y)


# ======================================================================================
# 2. MODELOS MATEMÁTICOS DE MEANDRO DA PLUMA
# ======================================================================================


class MeanderModel:
    """Classe base abstrata para modelos de meandro e flutuação transversal da pluma."""

    def __init__(self, main_config, model_config):
        self.main_config = main_config
        self.model_config = model_config

    def update(self, dt, current_time):
        raise NotImplementedError


class Linear_MeanderModel(MeanderModel):
    """
    Modelo de meandro linear (sem oscilação transversal).
    A velocidade transversal vy é nula e a pluma alinha-se diretamente com o escoamento médio.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)
        self.meandering_vy = 0.0

    def update(self, dt, current_time):
        return 0.0


class OU_MeanderModel(MeanderModel):
    """
    Modelo de meandro baseado no processo estocástico de Ornstein-Uhlenbeck (OU).
    Modela flutuações contínuas com reversão à média e dispersão browniana.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)
        self.meandering_vy = 0.0
        timescale = self.model_config.get("timescale", 1.0)
        intensity = self.model_config.get("intensity", 0.2)
        self.ou_theta = 1.0 / timescale if timescale > 0 else 0
        self.ou_sigma = intensity

    def update(self, dt, current_time):
        """Atualiza a velocidade transversal vy através da equação diferencial estocástica de OU."""
        if self.ou_theta > 0:
            drift = -self.ou_theta * self.meandering_vy * dt
            diffusion = self.ou_sigma * np.sqrt(dt) * np.random.randn()
            self.meandering_vy += drift + diffusion
        return self.meandering_vy


class Sinusoid_MeanderModel(MeanderModel):
    """
    Modelo de meandro sinusoidal harmónico.
    Produz uma oscilação ondulatória transversal periódica que simula esteiras
    e correntes sinuosas típicas de escoamentos marinhos e fluviais.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)

        self.num_harmonics = self.model_config.get("num_harmonics", 10)
        self.v_var = self.model_config.get("v_var", 0.25)
        self.L = self.model_config.get("integral_length_scale_L", 20.0)
        self.meandering_vy = 0.0

        drift_timescale = self.model_config.get("drift_correction_timescale", 45.0)

        if drift_timescale > 0:
            self.drift_theta = 1.0 / drift_timescale
        else:
            self.drift_theta = 0.0

        wind_speed = np.linalg.norm(self.main_config["mean_wind_velocity"])
        if wind_speed < 1e-6:
            self.amplitudes = np.zeros(self.num_harmonics)
            self.freqs = np.ones(self.num_harmonics)
            self.phases = np.zeros(self.num_harmonics)
            return

        low_freq = wind_speed / self.L
        high_freq = 100 * low_freq
        self.freqs = np.logspace(
            np.log10(low_freq), np.log10(high_freq), self.num_harmonics
        )

        wavenumbers = 2 * np.pi * self.freqs / wind_speed
        energy_spectrum = (self.v_var * (2 * self.L / np.pi)) / (
            1 + (wavenumbers * self.L) ** 2
        ) ** (5 / 6)

        freq_spectrum = energy_spectrum * (2 * np.pi / wind_speed)
        df = np.gradient(self.freqs)
        self.amplitudes = np.sqrt(2 * freq_spectrum * df)
        self.phases = np.random.rand(self.num_harmonics) * 2 * np.pi

    def update(self, dt, current_time):
        """
        Calcula a velocidade transversal instantânea física do fluido (dy/dt da esteira sinusoidal).
        A derivada temporal de y(t) = A * sin(2*pi*f*t) é dy/dt = A * 2*pi*f * cos(2*pi*f*t).
        """
        amplitude = self.model_config.get("amplitude", self.model_config.get("v_var", 8.0))
        wind_speed = np.linalg.norm(self.main_config.get("mean_wind_velocity", [0.4, 0.0]))
        L = self.model_config.get("integral_length_scale_L", 22.0)
        default_freq = (wind_speed / L) if L > 0 else 0.018
        frequency = self.model_config.get("frequency", default_freq)
        two_pi_f = 2.0 * np.pi * frequency
        self.meandering_vy = amplitude * two_pi_f * np.cos(two_pi_f * current_time)
        return self.meandering_vy


# ======================================================================================
# 3. CLASSES PRINCIPAIS DO SISTEMA DE SIMULAÇÃO
# ======================================================================================


class DataLogger:
    """Gere o registo e exportação de dados temporais da simulação para ficheiros HDF5 (.h5)."""

    def __init__(self, config, experiment_num):
        self.save_path = config["save_path"]
        self.experiment_name = config["experiment_name"]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            self.save_path, f"{self.experiment_name}_exp{experiment_num}_{timestamp}.h5"
        )
        self.data_buffer = {
            "time": [],
            "agent_position": [],
            "concentration": [],
            "agent_state": [],
            "battery": [],
            "custom_osl_data": [],
        }
        self.config = config

    def log_step(self, sim_time, agent_pos, concentration, agent_state, battery, custom_osl_data=None):
        """Regista um passo temporal na memória interna para posterior gravação."""
        self.data_buffer["time"].append(sim_time)
        self.data_buffer["agent_position"].append(agent_pos)
        self.data_buffer["concentration"].append(concentration)
        self.data_buffer["agent_state"].append(agent_state)
        self.data_buffer["battery"].append(battery)
        if custom_osl_data:
            self.data_buffer["custom_osl_data"].append(json.dumps(custom_osl_data))

    def save(self):
        """Escreve todos os dados acumulados no ficheiro HDF5 estruturado."""
        with h5py.File(self.filename, "w") as f:
            for key, data in self.data_buffer.items():
                if not data:
                    continue
                if key == "custom_osl_data":
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset("osl/custom_data", data=data, dtype=dt)
                elif key == "agent_state":
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset("agent/states", data=data, dtype=dt)
                elif key == "agent_position":
                    f.create_dataset("agent/positions", data=np.array(data))
                elif key == "concentration":
                    f.create_dataset("agent/concentrations", data=np.array(data))
                elif key == "battery":
                    f.create_dataset("agent/battery", data=np.array(data))
                else:
                    f.create_dataset(key, data=np.array(data))

            for key, value in self.config.items():
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = str(value)


class FilamentPlume:
    """
    Modelo lagrangeano de dispersão de filamentos de pluma no meio fluido.
    Gere a emissão periódica, expansão volumétrica, decaimento químico e
    meandro transversal dos pacotes de odor (puffs).
    """

    def __init__(self, config):
        self.config = config
        self.source_pos = np.array(config["source_position"])

        # Inicialização do array estático pré-alocado para máxima performance:
        # Coluna 0: Posição X em metros
        # Coluna 1: Posição Y em metros
        # Coluna 2: Raio do pacote (r) em metros
        # Coluna 3: Massa ativa do pacote (0.0 se inativo)
        # Coluna 4: Idade do pacote em segundos
        self.max_puffs = self.config.get("max_puffs", 4000)
        self.puffs_array = np.zeros((self.max_puffs, 5), dtype=np.float32)
        self.time_since_last_puff = 0.0

        model_name = self.config.get("meander_model", "ou").lower()
        if model_name in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]:
            model_config = self.config.get("sinusoid_meander_config", {})
            self.meander_generator = Sinusoid_MeanderModel(self.config, model_config)
        elif model_name == "linear":
            self.meander_generator = Linear_MeanderModel(self.config, {})
        else:
            model_config = self.config.get("ou_meander_config", {})
            self.meander_generator = OU_MeanderModel(self.config, model_config)

    def update(self, dt, current_time):
        meandering_vy = self.meander_generator.update(dt, current_time)
        self.time_since_last_puff += dt
        release_interval = 1.0 / self.config["emission_rate"]
        puffs_to_release = int(self.time_since_last_puff / release_interval)
        min_mass = self.config["initial_puff_mass"] * 0.01

        if puffs_to_release > 0:
            # Encontrar índices de slots inativos (massa <= min_mass)
            inactive_indices = np.where(self.puffs_array[:, 3] <= min_mass)[0]

            # Fallback de crescimento dinâmico se a capacidade for excedida
            if puffs_to_release > len(inactive_indices):
                print(f"Warning: Plume puffs array capacity exceeded. Resizing dynamically from {self.max_puffs} to {self.max_puffs * 2}.")
                old_puffs = self.puffs_array.copy()
                self.max_puffs *= 2
                self.puffs_array = np.zeros((self.max_puffs, 5), dtype=np.float32)
                self.puffs_array[:old_puffs.shape[0]] = old_puffs
                inactive_indices = np.where(self.puffs_array[:, 3] <= min_mass)[0]

            slots_to_use = inactive_indices[:puffs_to_release]

            if len(slots_to_use) > 0:
                self.puffs_array[slots_to_use, 0] = self.source_pos[0]
                self.puffs_array[slots_to_use, 1] = self.source_pos[1]
                self.puffs_array[slots_to_use, 2] = 0.0
                self.puffs_array[slots_to_use, 3] = self.config["initial_puff_mass"]
                self.puffs_array[slots_to_use, 4] = 0.0

            self.time_since_last_puff = 0.0

        if self.puffs_array.shape[0] > 0:
            is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]
            if is_sinusoid:
                sin_cfg = self.config.get("sinusoid_meander_config", {})
                amplitude = float(sin_cfg.get("amplitude", sin_cfg.get("v_var", 8.0)))
                frequency = float(sin_cfg.get("frequency", 0.018))
                expansion_rate = float(sin_cfg.get("puff_expansion_rate", self.config.get("puff_expansion_rate", 0.29)))
                turbulence = float(sin_cfg.get("turbulence_intensity", self.config.get("turbulence_intensity", 0.18)))

                self.puffs_array = update_puffs_sinusoid_numba(
                    self.puffs_array,
                    dt,
                    current_time,
                    float(self.source_pos[0]),
                    float(self.source_pos[1]),
                    np.array(self.config["mean_wind_velocity"], dtype=np.float32),
                    amplitude,
                    frequency,
                    turbulence,
                    float(self.config["puff_decay_rate"]),
                    expansion_rate,
                    min_mass,
                )
            else:
                self.puffs_array = update_puffs_numba(
                    self.puffs_array,
                    dt,
                    np.array(self.config["mean_wind_velocity"]),
                    meandering_vy,
                    self.config["turbulence_intensity"],
                    self.config["puff_decay_rate"],
                    self.config.get("puff_expansion_rate", 0.25),
                    min_mass,
                )
            # Atualizar inatividade no próprio array (sem realocação de memória)
            dead_mask = (self.puffs_array[:, 3] <= min_mass) | (
                self.puffs_array[:, 4] >= self.config["max_time"]
            )
            self.puffs_array[dead_mask, 3] = 0.0


class Agent:
    """
    Classe base para agentes de navegação autónoma subaquática (AUV).
    Modela as propriedades cinemáticas, consumo de bateria sob esforço de corrente
    e medições sensoriais de fluxo e concentração química.
    """

    def __init__(self, config):
        self.config = config
        self.radius = self.config["agent_radius"]
        self.speed = self.config["agent_speed"]
        self.measured_flow_direction = np.array([1.0, 0.0])
        self.battery_capacity = self.config.get("battery_capacity", 100.0)
        self.battery_base_decay = self.config.get("battery_base_decay", 0.08)
        self.battery_wind_factor = self.config.get("battery_wind_factor", 0.80)
        self.battery_min_decay = self.config.get("battery_min_decay", 0.02)
        self.reset()

    def reset(self):
        """Reinicializa as variáveis de estado do veículo para um novo ensaio."""
        start_y = float(np.random.uniform(0.30 * self.config["world_height"], 0.70 * self.config["world_height"]))
        self.config["agent_start_region_center"] = (90.0, start_y)
        self.pos = np.array([90.0, start_y], dtype=np.float64)
        self.start_pos = self.pos.copy()
        self.direction = np.array([1.0, 0.0])
        self.current_concentration = 0.0
        self.measured_flow_direction = np.array([1.0, 0.0])
        if hasattr(self, "internal_state"):
            self.internal_state = "casting"

        self.state = "waiting"
        self.first_detection_pos = None
        self.mission_complete = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.source_reached = False
        self.prev_dist_to_detection = None
        self.skirting_side_ida = None
        self.battery = self.battery_capacity
        self.return_waypoint_reached = False
        self.low_battery_direct_return = False
        self.low_battery_start_return = False
        self.low_battery_aborted = False
        self.incomplete_mission = False
        self.theta = 0.0
        self.return_contour_completed = False
        self.return_tip_reached = False
        self.distal_tip_entered = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.pos_history = []

    def get_circling_target(self, dt):
        """
        Calcula o ponto-alvo para manobra de contorno circular suave em redor da fonte (360°),
        antes de iniciar o estado de retorno à posição da primeira deteção.
        """
        source_pos = np.array(self.config["source_position"])
        radius = self.config.get("agent_config", {}).get("success_distance", 2.0) * 1.0
        ida_side = self.skirting_side_ida if self.skirting_side_ida is not None else 1.0
        
        if self.circling_step == 0:
            self.circling_angle = np.pi / 2.0 if ida_side == 1.0 else 3.0 * np.pi / 2.0
            
        # Update angle smoothly at physical speed (v/R * dt)
        angular_speed = self.speed / radius
        if ida_side == 1.0:
            self.circling_angle += angular_speed * dt
            if self.circling_angle >= 3.0 * np.pi / 2.0:
                self.state = "returning"
                self.source_reached = False
                self.circling_completed = True
                self.return_contour_completed = False
                self.return_tip_reached = False
                self.return_side_ida_reached = False
                self.return_at_detection_x = False
                self.theta = None
                exit_pos = np.array([source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64)
                self.pos = exit_pos.copy()
                print(f"\n[Agent] Circling complete! State changed to RETURNING on opposite margin ({exit_pos[0]:.1f}, {exit_pos[1]:.1f}). Goal: {self.first_detection_pos}")
                return exit_pos
        else:
            self.circling_angle -= angular_speed * dt
            if self.circling_angle <= np.pi / 2.0:
                self.state = "returning"
                self.source_reached = False
                self.circling_completed = True
                self.return_contour_completed = False
                self.return_tip_reached = False
                self.return_side_ida_reached = False
                self.return_at_detection_x = False
                self.theta = None
                exit_pos = np.array([source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64)
                self.pos = exit_pos.copy()
                print(f"\n[Agent] Circling complete! State changed to RETURNING on opposite margin ({exit_pos[0]:.1f}, {exit_pos[1]:.1f}). Goal: {self.first_detection_pos}")
                return exit_pos
                
        target = source_pos + np.array([radius * np.cos(self.circling_angle), radius * np.sin(self.circling_angle)])
        self.circling_step += 1
        return target

    def measure_flow(self, plume):
        mean_wind = np.array(self.config["mean_wind_velocity"])

        if isinstance(plume.meander_generator, (OU_MeanderModel, Sinusoid_MeanderModel)):
            vy = plume.meander_generator.meandering_vy
            self.measured_flow_direction = mean_wind + np.array([0.0, vy])
        else:
            self.measured_flow_direction = mean_wind.copy()

        return self.measured_flow_direction

    def measure_concentration(self, plume):
        model = self.config.get("concentration_model", "top-hat")
        if model == "gaussian":
            self.current_concentration = calculate_concentration_gaussian_numba(
                self.pos, plume.puffs_array
            )
        else:
            self.current_concentration = calculate_concentration_numba(
                self.pos, self.radius, plume.puffs_array
            )
        return self.current_concentration

    def move(self, target_pos, dt):
        if target_pos is None:
            return
        world_w, world_h = self.config["world_width"], self.config["world_height"]
        valid_target_pos = np.array(
            [np.clip(target_pos[0], 0, world_w), np.clip(target_pos[1], 0, world_h)]
        )
        direction_vec = valid_target_pos - self.pos
        distance = np.linalg.norm(direction_vec)
        if distance > 1e-6:
            self.direction = direction_vec / distance
            displacement = self.direction * self.speed * dt
            self.pos = (
                valid_target_pos
                if np.linalg.norm(displacement) > distance
                else self.pos + displacement
            )

    def check_success(self, source_pos):
        return np.linalg.norm(self.pos - source_pos) < self.config.get(
            "success_distance", 1.0
        )

    def run_search_algorithm(self, dt):
        raise NotImplementedError("This method should be implemented by a subclass.")


class SimpleUpwindAgent(Agent):
    """A simple agent for basic testing."""

    def __init__(self, config):
        self.internal_state, self.cast_direction, self.last_hit_pos = None, None, None
        super().__init__(config)

    def reset(self):
        super().reset()
        self.internal_state, self.cast_direction, self.last_hit_pos = "casting", 1, None

    def run_search_algorithm(self, dt):
        concentration_threshold = self.config.get("agent_concentration_threshold", 1.0)
        surge_len = self.config.get("surge_length", 5.0)
        cast_len = self.config.get("cast_length", 8.0)

        custom_data = {"internal_state": self.internal_state}
        if self.current_concentration > concentration_threshold:
            self.internal_state = "surging"
            self.last_hit_pos = self.pos.copy()
            upwind_vec = -np.array(self.config["mean_wind_velocity"])
            target_pos = self.pos + upwind_vec * surge_len
        else:
            if self.internal_state == "surging":
                self.internal_state, self.cast_direction = "casting", self.cast_direction * -1
            crosswind_vec = np.array(
                [
                    -self.config["mean_wind_velocity"][1],
                    self.config["mean_wind_velocity"][0],
                ]
            )
            target_pos = self.pos + (crosswind_vec * self.cast_direction * cast_len)
        return target_pos, custom_data


class BioInspiredAgent(Agent):
    """
    Agente com estratégia de busca bio-inspirada (Cast-and-Surge com movimento em traça).
    Implementa histerese de decisão sensorial, busca transversal progressiva ao perder o sinal
    e retorno de emergência à posição de partida se a bateria descer abaixo de 15%.
    """

    def __init__(self, config):
        self.main_config = config
        super().__init__(config)

    def reset(self):
        """Reinicializa as variáveis de navegação e controlo comportamental do agente."""
        super().reset()
        self.state, self.behavior, self.internal_state = "SEARCHING", "SEARCHING", "DECIDING"
        self.target_pos, self.cast_direction, self.search_direction = None, 1, 1
        self.consecutive_hits, self.consecutive_misses = 0, 0
        self.current_cast_length = self.config.get("cast_length", 25.0)

    def _compute_zig_zag_target(self, length, angle_deg, direction_sign):
        """Calcula o vetor de deslocamento em ziguezague relativamente à corrente contra o vento."""
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm < 1e-6:
            return self.pos
        upwind_dir = -self.measured_flow_direction / flow_norm
        angle_rad = np.deg2rad(angle_deg * direction_sign)
        rot_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        return self.pos + (rot_matrix @ upwind_dir) * length

    def _set_search_target(self):
        print(f"BEHAVIOR: Searching (direction: {self.search_direction})")
        self.behavior = "SEARCHING"
        potential_target = self._compute_zig_zag_target(
            self.config["search_length"],
            self.config["search_angle"],
            self.search_direction,
        )
        if not (2.0 < potential_target[1] < self.main_config["world_height"] - 2.0):
            print("  > Search target out of bounds. Inverting direction.")
            self.search_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.config["search_length"],
                self.config["search_angle"],
                self.search_direction,
            )
        else:
            self.target_pos = potential_target
        if self.target_pos is not None:
            self.target_pos[0] = np.clip(self.target_pos[0], 2.0, self.main_config["world_width"] - 2.0)
            self.target_pos[1] = np.clip(self.target_pos[1], 2.0, self.main_config["world_height"] - 2.0)
        self.internal_state = "MOVING_TO_TARGET"

    def _set_cast_target(self):
        print(
            f"BEHAVIOR: Casting (direction: {self.cast_direction}, length: {self.current_cast_length:.2f})"
        )
        self.behavior = "CASTING"
        potential_target = self._compute_zig_zag_target(
            self.current_cast_length, self.config["cast_angle"], self.cast_direction
        )
        if not (2.0 < potential_target[1] < self.main_config["world_height"] - 2.0):
            print("  > Cast target out of bounds. Inverting direction.")
            self.cast_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.current_cast_length, self.config["cast_angle"], self.cast_direction
            )
        else:
            self.target_pos = potential_target
        if self.target_pos is not None:
            self.target_pos[0] = np.clip(self.target_pos[0], 2.0, self.main_config["world_width"] - 2.0)
            self.target_pos[1] = np.clip(self.target_pos[1], 2.0, self.main_config["world_height"] - 2.0)
        self.internal_state = "MOVING_TO_TARGET"

    def _set_surge_target(self):
        print("BEHAVIOR: Surging")
        self.behavior = "SURGING"
        self.current_cast_length = self.config.get("cast_length", 25.0)
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm > 1e-6:
            upwind_dir = -self.measured_flow_direction / flow_norm
            self.target_pos = self.pos + upwind_dir * self.config["surge_length"]
        self.internal_state = "MOVING_TO_TARGET"

    def run_search_algorithm(self, dt):
        is_hit = (
            self.current_concentration > self.config["agent_concentration_threshold"]
        )
        if is_hit:
            self.consecutive_hits, self.consecutive_misses = (
                self.consecutive_hits + 1,
                0,
            )
        else:
            self.consecutive_misses, self.consecutive_hits = (
                self.consecutive_misses + 1,
                0,
            )
        confirmation_steps = self.config.get("confirmation_steps", 1)

        if self.state == "SEARCHING" and self.consecutive_hits >= confirmation_steps:
            print(
                f"PLUME FOUND ({self.consecutive_hits} consecutive hits)! Switching to TRACKING state."
            )
            self.state, self.internal_state = "TRACKING", "DECIDING"
        elif (
            self.state == "TRACKING"
            and self.behavior == "SURGING"
            and self.consecutive_misses >= confirmation_steps
        ):
            print(
                f"PLUME LOST ({self.consecutive_misses} consecutive misses)! Switching to CAST."
            )
            self.internal_state = "DECIDING"
        elif (
            self.state == "TRACKING"
            and self.behavior == "CASTING"
            and self.consecutive_hits >= confirmation_steps
        ):
            print(
                f"PLUME RE-ACQUIRED ({self.consecutive_hits} consecutive hits)! Switching to SURGE."
            )
            self.internal_state = "DECIDING"

        if self.internal_state == "DECIDING":
            if self.state == "SEARCHING":
                self._set_search_target()
            elif self.state == "TRACKING":
                if self.consecutive_hits > 0:
                    self._set_surge_target()
                else:
                    if self.behavior == "CASTING":
                        self.current_cast_length *= 1.0 + self.config.get(
                            "cast_length_increase_factor", 0.1
                        )
                        print(
                            f"  > Cast failed. Increasing cast length to {self.current_cast_length:.2f}"
                        )
                        self.cast_direction *= -1
                    self._set_cast_target()
        elif self.internal_state == "MOVING_TO_TARGET":
            hit_wall = (
                self.pos[1] <= 2.2 or self.pos[1] >= self.main_config["world_height"] - 2.2 or
                self.pos[0] <= 2.2 or self.pos[0] >= self.main_config["world_width"] - 2.2
            )
            if (
                self.target_pos is None
                or np.linalg.norm(self.pos - self.target_pos) < 0.8
                or hit_wall
            ):
                self.internal_state, self.target_pos = "DECIDING", None

        custom_data = {
            "state": self.state,
            "behavior": self.behavior,
            "hits": self.consecutive_hits,
            "misses": self.consecutive_misses,
        }
        return self.target_pos, custom_data


class PlumeRLWrapper(gym.Env):
    """
    Ambiente de simulação compatível com a API Gymnasium para treino por Reforço.
    Encapsula a física da dispersão lagrangeana, dinâmica de bateria do AUV e
    calcula as funções de recompensa para orientação e localização da fonte.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config):
        """
        Inicializa o ambiente de treino, define os espaços de ação discreta (4 direções)
        e os espaços de observação contínua normalizados (8 dimensões).
        """
        super().__init__()
        self.config = config

        # Ações discretas: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
        self.action_space = spaces.Discrete(4)

        # Vetor de observação com 8 variáveis normalizadas:
        # [x, y, concentração, fluxo_x, fluxo_y, estado_missão, nível_bateria, sentido_varrimento]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 3, 1, 1], dtype=np.float32),
            shape=(8,),
            dtype=np.float32,
        )
        self.reward_config = {
            "step_penalty": -0.05,
            "concentration_improvement": 1.0,
            "concentration_decrease": -0.05,
            "revisit_threshold": 1.0,
            "revisit_penalty": -0.2,
            "plume_detection_bonus": 0.3,
            "upwind_bonus": 1.0,
            "goal_reward": 500.0,
            "battery_drain_penalty_factor": 0.5,
            "battery_depleted_penalty": -200.0,
        }

        self.conc_max = 10.0
        self.flow_max = 5.0

        self.battery_capacity = self.config.get("battery_capacity", 100.0)
        self.battery = self.battery_capacity
        self.last_drain = 0.0

        # Rastreio de variáveis de estado e máquina de estados
        self.speed = self.config["agent_config"]["agent_speed"]
        self.plume = None
        self.pos = None
        self.time = 0.0
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_pos = None
        self.prev_conc = 0.0
        self.current_conc = 0.0
        self.state = "searching"
        self.cast_dir = 0
        self.first_detection_pos = None
        self.mission_complete = False
        self.source_reached = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.return_tip_reached = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.skirting_side_ida = None
        self.prev_dist_to_detection = None
        self.low_battery_return = False
        self.return_to_start = False
        self.start_pos = None
        self.searching_consecutive_straight_steps = 0
        self.prev_dist_to_source = None
        self.min_dist_to_source = float("inf")
        self.prev_dist_to_sweep = None
        self.recent_positions = []
        self.consecutive_loop_steps = 0

    def _validate_config(self):
        """Valida a consistência dos parâmetros físicos configurados no ambiente."""
        assert self.config["world_width"] > 0, "world_width deve ser positivo"
        assert self.config["world_height"] > 0, "world_height deve ser positivo"
        assert self.config["dt"] > 0 and self.config["dt"] < 1, "dt deve situar-se no intervalo (0, 1)"
        assert self.config["agent_config"]["agent_speed"] > 0, (
            "agent_speed deve ser positivo"
        )
        assert self.config["emission_rate"] > 0, "emission_rate must be positive"
        assert self.config["max_time"] > 0, "max_time must be positive"

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode
        """
        super().reset(seed=seed)

        # Validate config
        self._validate_config()

        start_y = float(np.random.uniform(0.30 * self.config["world_height"], 0.70 * self.config["world_height"]))
        self.config["agent_start_region_center"] = (90.0, start_y)

        self.plume = FilamentPlume(self.config)
        self.time = 0.0

        # Initialize agent position: x = 90, y random between 10 and 40
        self.pos = np.array([90.0, start_y], dtype=np.float64)
        self.start_pos = self.pos.copy()
        self.prev_pos = self.pos.copy()

        # Reset tracking
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_conc = 0.0
        self.current_conc = 0.0
        self.first_detection_pos = None
        self.has_moved_away = False
        self.path_length = 0.0

        # Propriedades da máquina de estados da missão
        self.state = "searching"
        self.first_detection_pos = None
        self.mission_complete = False
        self.source_reached = False
        self.circling_step = 0
        self.circling_angle = 0.0
        self.circling_completed = False
        self.return_tip_reached = False
        self.distal_tip_entered = False
        self.return_side_ida_reached = False
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.low_battery_aborted = False
        self.prev_action = None
        self.skirting_side_ida = None
        self.prev_dist_to_detection = None
        self.prev_dist_to_source = None
        self.min_dist_to_source = float("inf")
        self.prev_dist_to_sweep = None
        self.low_battery_return = False
        self.return_to_start = False
        self.battery = self.battery_capacity
        self.last_drain = 0.0
        self.searching_consecutive_straight_steps = 0
        self.cast_dir = int(np.random.choice([0, 1]))
        self.recent_positions = []
        self.consecutive_loop_steps = 0
        self.max_x_returning = 20.0
        self.min_x_returning_back = 75.0
        self.min_dist_to_det = 70.0

        # Esperar 8 segundos no início de cada episódio antes de iniciar o movimento
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        for _ in range(int(initial_wait_time / self.config["dt"])):
            self.plume.update(self.config["dt"], self.time)
            self.time += self.config["dt"]

        return self._get_obs(), {"at_goal": False}

    def step(self, action):
        """
        Execute one step in environment
        """
        dt = self.config["dt"]
        self.prev_pos = self.pos.copy()

        if self.state == "searching":
            search_angle_deg = float(self.config.get("agent_config", {}).get("search_angle", 75.0))
            crab_upwind = float(np.cos(np.deg2rad(search_angle_deg)))  # ~0.2588 contra o vento (-x)
            crosswind = float(np.sin(np.deg2rad(search_angle_deg)))    # ~0.9659 no eixo vertical y
            dirs = {
                0: np.array([-crab_upwind, -crosswind]),  # Baixo (-y) com avanço contra o vento (-x)
                1: np.array([-crab_upwind, crosswind]),   # Cima (+y) com avanço contra o vento (-x)
                2: np.array([-1.0, 0.0]),                 # Esquerda direto contra o vento (-x)
                3: np.array([1.0, 0.0]),                  # Direita a favor do vento (+x)
            }
        else:
            dirs = {
                0: np.array([0.0, -1.0]),  # Baixo (-y, Sul)
                1: np.array([0.0, 1.0]),   # Cima (+y, Norte)
                2: np.array([-1.0, 0.0]),  # Esquerda (-x, Oeste)
                3: np.array([1.0, 0.0]),   # Direita (+x, Este)
            }

        # --- High-level state transitions in training env ---
        conc_thresh = self.config["agent_config"]["agent_concentration_threshold"]
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.pos - source_pos)
        success_distance = self.config["agent_config"]["success_distance"]

        # Definição do decay-rate da bateria dependendendo do vento
        direction = dirs[action]
        vy = self.plume.meander_generator.meandering_vy
        flow_vec = np.array(self.config["mean_wind_velocity"]) + np.array([0.0, vy])
        opposing_wind = -flow_vec
        wind_resistance = np.dot(direction, opposing_wind)

        base_decay = self.config.get("battery_base_decay", 0.08)
        wind_factor = self.config.get("battery_wind_factor", 0.80)
        min_decay = self.config.get("battery_min_decay", 0.02)

        drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
        self.last_drain = drain_rate * dt
        self.battery = max(0.0, self.battery - self.last_drain)
        self.current_action = action

        if self.battery <= 0:
            terminated = True
            truncated = False
            reward = self.reward_config.get("battery_depleted_penalty", -200.0)
            self.prev_action = action
            return self._get_obs(), reward, terminated, truncated, {"at_goal": False, "battery_depleted": True}

        # Verificação universal de segurança de bateria baixa (<= 15%):
        battery_pct = (self.battery / self.battery_capacity) * 100.0
        if battery_pct <= 15.0:
            self.low_battery_return = True
            self.low_battery_aborted = True
            if self.first_detection_pos is None:
                self.return_to_start = True
                start_target = getattr(self, "start_pos", np.array([90.0, 25.0]))
                self.prev_dist_to_detection = np.linalg.norm(self.pos - start_target)
            else:
                self.prev_dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)

        if self.state == "searching":
            if action == 2:  # Ação 2: Esquerda (avanço direto contra o vento)
                self.searching_consecutive_straight_steps += 1
            else:
                self.searching_consecutive_straight_steps = 0

            # Limites de varrimento em traça (cobre entre 30% e 70% da altura do mundo)
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]

            if self.pos[1] >= y_max_sweep:
                if self.cast_dir == 0:
                    self.cast_dir = 1
                    self.prev_dist_to_sweep = None
            elif self.pos[1] <= y_min_sweep:
                if self.cast_dir == 1:
                    self.cast_dir = 0
                    self.prev_dist_to_sweep = None

            self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)
            if self.current_conc > conc_thresh:
                self.state = "following_boundary"
                self.first_detection_pos = self.pos.copy()
                local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
                self.skirting_side_ida = 1.0 if self.pos[1] >= local_center_y else -1.0
                self.prev_dist_to_source = dist_to_source
                self.min_dist_to_source = dist_to_source
            elif battery_pct <= 15.0:
                self.state = "returning"

        elif self.state == "following_boundary":
            if (dist_to_source < success_distance or (self.pos[0] <= source_pos[0] + 1.8 and dist_to_source < success_distance * 1.8)) and not self.source_reached:
                self.source_reached = True
                self.circling_step = 0
            
            if not self.source_reached and battery_pct <= 15.0:
                self.state = "returning"

        # --- Circling source is handled programmatically as part of following_boundary ---
        if self.state == "following_boundary" and self.source_reached:
            radius = success_distance * 1.0
            ida_side = self.skirting_side_ida if self.skirting_side_ida is not None else 1.0
            
            if self.circling_step == 0:
                self.circling_angle = np.pi / 2.0 if ida_side == 1.0 else 3.0 * np.pi / 2.0
                
            # Update angle smoothly at physical speed (v/R * dt)
            angular_speed = self.speed / radius
            if ida_side == 1.0:
                self.circling_angle += angular_speed * dt
                if self.circling_angle >= 3.0 * np.pi / 2.0:
                    self.state = "returning"
                    self.source_reached = False
                    self.circling_completed = True
                    self.return_tip_reached = False
                    self.return_side_ida_reached = False
                    self.return_at_detection_x = False
                    self.return_contour_completed = False
                    self.max_x_returning = self.pos[0]
                    self.min_x_returning_back = 75.0
                    if self.first_detection_pos is None:
                        self.first_detection_pos = self.pos.copy()
                    self.pos = np.array([source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64)
                    self.prev_dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
                    self.min_dist_to_det = self.prev_dist_to_detection
                    self.recent_positions = []
                    self.current_cell = None
                    self.steps_in_current_cell = 0
            else:
                self.circling_angle -= angular_speed * dt
                if self.circling_angle <= np.pi / 2.0:
                    self.state = "returning"
                    self.source_reached = False
                    self.circling_completed = True
                    self.return_tip_reached = False
                    self.return_side_ida_reached = False
                    self.return_at_detection_x = False
                    self.return_contour_completed = False
                    self.max_x_returning = self.pos[0]
                    self.min_x_returning_back = 75.0
                    if self.first_detection_pos is None:
                        self.first_detection_pos = self.pos.copy()
                    self.pos = np.array([source_pos[0] + 0.8, source_pos[1] - ida_side * 0.8], dtype=np.float64)
                    self.prev_dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
                    self.min_dist_to_det = self.prev_dist_to_detection
                    self.recent_positions = []
                    self.current_cell = None
                    self.steps_in_current_cell = 0
                    
            if not self.circling_completed:
                self.pos = source_pos + np.array([radius * np.cos(self.circling_angle), radius * np.sin(self.circling_angle)])
            self.circling_step += 1
            
            # Move plume and increment time
            self.plume.update(dt, self.time)
            self.time += dt
            self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)
            obs = self._get_obs()
            reward = 2.0  # constant small reward for circling
            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            terminated = False
            truncated = self.time >= self.config["max_time"]
            self.prev_action = action
            return obs, reward, terminated, truncated, {"at_goal": False}

        # --- Normal step movement (for searching, following, returning) ---
        target = self.pos + dirs[action] * self.config["agent_config"]["agent_speed"] * dt

        hit_boundary = False
        if (target[0] < 0.0 or target[0] > self.config["world_width"] or
            target[1] < 0.0 or target[1] > self.config["world_height"]):
            hit_boundary = True

        self.pos[0] = np.clip(target[0], 0, self.config["world_width"])
        self.pos[1] = np.clip(target[1], 0, self.config["world_height"])

        if not hasattr(self, "recent_positions") or self.recent_positions is None:
            self.recent_positions = []
        self.recent_positions.append(self.pos.copy())
        if len(self.recent_positions) > 16:
            self.recent_positions.pop(0)

        # Update plume
        self.plume.update(dt, self.time)
        self.time += dt

        # Track cell visitation
        x_b = int(np.clip(self.pos[0] / self.config["world_width"] * 50, 0, 49))
        y_b = int(np.clip(self.pos[1] / self.config["world_height"] * 25, 0, 24))
        new_cell = (x_b, y_b)

        if new_cell != self.current_cell:
            self.visit_map[new_cell] = self.visit_map.get(new_cell, 0) + 1
            self.current_cell = new_cell
            self.steps_in_current_cell = 0
        else:
            self.steps_in_current_cell += 1

        visitas_nesta_celula = self.visit_map.get(new_cell, 1)

        self.prev_conc = self.current_conc
        self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)

        # Verificação de conclusão de objetivo / término
        if self.state == "returning":
            if getattr(self, "return_to_start", False):
                det_pos = getattr(self, "start_pos", np.array([90.0, 25.0]))
            else:
                det_pos = self.first_detection_pos if self.first_detection_pos is not None else getattr(self, "start_pos", self.pos)
            dist_to_detection = np.linalg.norm(self.pos - det_pos)

            battery_pct = (self.battery / self.battery_capacity) * 100.0
            is_low_battery = getattr(self, "low_battery_return", False) or getattr(self, "low_battery_aborted", False) or (battery_pct <= 15.0)

            # Contorno total e preciso da pluma no regresso:
            if not is_low_battery:
                det_x = det_pos[0]
                side_ida = getattr(self, "skirting_side_ida", 1.0)
                if side_ida is None:
                    side_ida = 1.0
                local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
                is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]

                if is_sinusoid:
                    # Na pluma sinusoidal: o agente contorna a margem oposta até alcançar o mesmo X da deteção.
                    # Ao alcançar o mesmo x (self.pos[0] >= det_x - 0.5), conclui o contorno da pluma e desloca-se diretamente para o ponto de deteção!
                    opposite_side = -side_ida
                    is_on_opposite_side = (self.pos[1] >= local_center_y) if opposite_side == 1.0 else (self.pos[1] <= local_center_y)
                    if not getattr(self, "return_tip_reached", False):
                        if self.pos[0] >= det_x - 0.5 and (is_on_opposite_side or self.current_conc >= 0.15 * conc_thresh):
                            self.return_tip_reached = True
                            self.return_contour_completed = True
                    else:
                        if dist_to_detection < success_distance:
                            self.return_contour_completed = True
                else:
                    # Na pluma linear: contorno completo de toda a pluma até à ponta distal (x_turn ~80m),
                    # virando depois para a margem de ida a montante (-x) até alcançar det_pos:
                    x_turn = min(80.0, max(det_x + 16.0, 78.0))
                    if not getattr(self, "return_tip_reached", False):
                        is_on_ida_side = (self.pos[1] >= local_center_y) if side_ida == 1.0 else (self.pos[1] <= local_center_y)
                        if self.pos[0] >= x_turn:
                            self.distal_tip_entered = True
                        if getattr(self, "distal_tip_entered", False) and is_on_ida_side:
                            self.return_tip_reached = True
                            self.min_x_returning_back = self.pos[0]
                            self.min_dist_to_det = dist_to_detection

                    if getattr(self, "return_tip_reached", False):
                        if dist_to_detection < success_distance:
                            self.return_contour_completed = True

            contour_done = getattr(self, "return_contour_completed", False)

            if is_low_battery:
                reached_detection = dist_to_detection < success_distance
                at_goal = False
                terminated = reached_detection
            else:
                reached_detection = dist_to_detection < success_distance
                at_goal = reached_detection and contour_done and getattr(self, "circling_completed", False) and (battery_pct > 15.0)
                terminated = at_goal

            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(self.current_conc, self.prev_conc, visitas_nesta_celula, at_goal, hit_boundary, action)
            if truncated and not at_goal:
                reward -= 100.0
        else:
            at_goal = False
            terminated = False  # A simulação não termina ao alcançar a fonte, segue para contorno e regresso
            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(self.current_conc, self.prev_conc, visitas_nesta_celula, False, hit_boundary, action)
            if truncated:
                reward -= 100.0

        self.prev_action = action
        return self._get_obs(), reward, terminated, truncated, {"at_goal": at_goal}

    def _calculate_reward(self, conc, prev_conc, visitas_nesta_celula, at_goal, hit_boundary, action=None):
        threshold = self.config["agent_config"]["agent_concentration_threshold"]
        source_pos = np.array(self.config["source_position"])
        step_size = self.speed * self.config["dt"]
        movement = self.pos - self.prev_pos

        # Verificação de inversão imediata de direção
        is_reversal = False
        if action is not None and getattr(self, "prev_action", None) is not None:
            is_reversal = (
                (action == 0 and self.prev_action == 1) or
                (action == 1 and self.prev_action == 0) or
                (action == 2 and self.prev_action == 3) or
                (action == 3 and self.prev_action == 2)
            )

        # Penalização estrita de anti-estagnação / anti-bloqueio (janela móvel de 16 passos)
        anti_stall_penalty = 0.0
        if hasattr(self, "recent_positions") and len(self.recent_positions) >= 16:
            disp_16 = float(np.linalg.norm(self.pos - self.recent_positions[-16]))
            if disp_16 < 0.40:
                anti_stall_penalty = 8.0 * (1.0 - disp_16 / 0.40)

        # =========================================================================
        # ESTADO 3: RETURNING (Contorno do resto da pluma até à primeira deteção)
        # =========================================================================
        if self.state == "returning":
            if at_goal:
                return 500.0  # Recompensa terminal ao regressar à posição de deteção após contorno completo

            battery_pct = (self.battery / self.battery_capacity) * 100.0
            is_low_battery = getattr(self, "low_battery_return", False) or getattr(self, "low_battery_aborted", False) or (battery_pct <= 15.0)

            if getattr(self, "return_to_start", False):
                det_pos = getattr(self, "start_pos", np.array([90.0, 25.0]))
            else:
                det_pos = self.first_detection_pos if self.first_detection_pos is not None else getattr(self, "start_pos", self.pos)

            dist_to_detection = np.linalg.norm(self.pos - det_pos)

            if is_low_battery:
                # Retorno por bateria baixa: NUNCA pontua como sucesso!
                if dist_to_detection < self.config["agent_config"]["success_distance"]:
                    return -30.0  # Penalização terminal por missão abortada devido a esgotamento da bateria
                reward = -0.5
                curr_dist = np.linalg.norm(self.pos - det_pos)
                prev_dist = np.linalg.norm(self.prev_pos - det_pos)
                if curr_dist < prev_dist:
                    reward += 1.0
                return reward

            reward = self.reward_config["step_penalty"]
            dy = movement[1]
            dx = movement[0]
            # No regresso a montante (-x), delta_x_upstream > 0 significa avanço para Oeste em direção à deteção
            delta_x_upstream = self.prev_pos[0] - self.pos[0]
            # No regresso a jusante (+x), delta_x_downstream > 0 significa avanço para Este em direção à ponta
            delta_x_downstream = self.pos[0] - self.prev_pos[0]

            side_ida = getattr(self, "skirting_side_ida", 1.0)
            if side_ida is None:
                side_ida = 1.0
            opposite_side = -side_ida
            local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)

            is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]
            tip_reached = getattr(self, "return_tip_reached", False)
            tip_entered = getattr(self, "distal_tip_entered", False)

            curr_d = np.linalg.norm(self.pos - det_pos)
            prev_d = np.linalg.norm(self.prev_pos - det_pos)

            if is_sinusoid:
                # =====================================================================
                # PLUMA SINUSOIDAL:
                # Contorno ondulante da margem oposta a jusante (+x) até ao mesmo X da deteção.
                # Ao alcançar o mesmo X (self.pos[0] >= det_pos[0] - 0.5), desloca-se diretamente para det_pos!
                # =====================================================================
                is_at_detection_x = tip_reached or (self.pos[0] >= det_pos[0] - 0.5)

                if is_at_detection_x:
                    # Deslocamento direto para o ponto de deteção
                    prog_d = (prev_d - curr_d) / step_size
                    if prog_d > 0.01:
                        reward += 8.0 * prog_d
                    elif prog_d < -0.01:
                        reward -= 6.0 * abs(prog_d)
                    else:
                        reward -= 1.0
                else:
                    # Até alcançar det_pos[0], contorna a margem oposta da pluma sinusoidal a jusante (+x)
                    target_side = opposite_side
                    prog_x = delta_x_downstream / step_size
                    prog_goal = prog_x

                    is_on_correct_side = (self.pos[1] >= local_center_y) if target_side == 1.0 else (self.pos[1] <= local_center_y)
                    if not is_on_correct_side:
                        reward -= 10.0
                        if dy * target_side > 0:
                            reward += 3.0
                    else:
                        dist_prev = abs(self.prev_pos[1] - local_center_y)
                        dist_curr = abs(self.pos[1] - local_center_y)
                        dy_outward = (dist_curr - dist_prev) / step_size
                        dy_inward = (dist_prev - dist_curr) / step_size

                        if conc > 1.8 * threshold:
                            if dy_outward > 0.05:
                                reward += 6.0 * dy_outward
                            else:
                                reward -= 3.0
                            if dy_inward > 0.05:
                                reward -= 4.0 * dy_inward
                            if prog_goal > 0.01:
                                reward += 1.0 * prog_goal
                            elif prog_goal < -0.01:
                                reward -= 3.0 * abs(prog_goal)
                        elif conc < 0.20 * threshold:
                            # Em água limpa fora da pluma: aproximação OBRIGATÓRIA à margem ondulante
                            if dy_inward > 0.05:
                                reward += 8.0 * dy_inward
                            else:
                                reward -= 4.0
                            if dy_outward > 0.05:
                                reward -= 6.0 * dy_outward
                            # Proibir estritamente cruzeiro downstream em água limpa (impedir atalhos!)
                            if prog_goal > 0.01:
                                if self.pos[0] > source_pos[0] + 3.0:
                                    reward -= 10.0 * prog_goal
                                else:
                                    reward += 2.0 * prog_goal
                        else:
                            # Medição matemática da convergência ao contorno do puff (erro de concentração):
                            prev_err = abs(prev_conc - threshold)
                            curr_err = abs(conc - threshold)
                            delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                            log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                            adh = float(np.exp(-2.5 * log_ratio))

                            if prog_goal > 0.01:
                                reward += 4.0 * prog_goal + 6.0 * adh
                                if delta_contour > 0.01:
                                    reward += 2.0 * min(delta_contour, 2.0)
                                elif conc > threshold and dy_outward > 0.05:
                                    reward += 2.0 * dy_outward
                                elif conc < threshold and dy_inward > 0.05:
                                    reward += 2.0 * dy_inward
                            elif prog_goal < -0.01:
                                reward -= 5.0 * abs(prog_goal)
                            else:
                                if delta_contour > 0.01 or (conc > threshold and dy_outward > 0.05) or (conc < threshold and dy_inward > 0.05) or adh > 0.3:
                                    reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                                else:
                                    reward -= 2.0
            else:
                # =====================================================================
                # PLUMA LINEAR:
                # Contorno completo da pluma inteira até à ponta distal (x_turn ~80m),
                # virando depois para a margem de ida até regressar ao ponto de deteção!
                # =====================================================================
                x_turn = min(80.0, max(det_pos[0] + 16.0, 78.0))
                is_at_detection_x = tip_reached and (self.pos[0] <= det_pos[0] + 0.5)
                is_docking = tip_reached and (curr_d <= 5.0 or is_at_detection_x)

                if is_docking:
                    prog_d = (prev_d - curr_d) / step_size
                    if is_at_detection_x:
                        if prog_d > 0.01:
                            reward += 8.0 * prog_d
                        elif prog_d < -0.01:
                            reward -= 6.0 * abs(prog_d)
                        else:
                            reward -= 1.5
                        if self.pos[0] < det_pos[0] - 0.5 and delta_x_upstream > 0.01:
                            reward -= 4.0 * (delta_x_upstream / step_size)
                    else:
                        if prog_d > 0.01:
                            reward += 6.0 * prog_d
                        elif prog_d < -0.01:
                            reward -= 5.0 * abs(prog_d)
                        else:
                            reward -= 1.0
                else:
                    is_rounding_tip = (not tip_reached) and (tip_entered or self.pos[0] >= x_turn)

                    if is_rounding_tip:
                        if dy * side_ida > 0:
                            if conc > 1.5 * threshold:
                                reward -= 4.0
                            else:
                                prev_err = abs(prev_conc - threshold)
                                curr_err = abs(conc - threshold)
                                delta_c = (prev_err - curr_err) / max(threshold, 1e-4)
                                reward += 5.5 + 2.0 * max(0.0, delta_c)
                        elif dy * side_ida < 0:
                            reward -= 4.0
                        else:
                            reward -= 1.0

                        if self.pos[0] > x_turn + 2.0 and movement[0] > 0.01:
                            reward -= 3.5
                    else:
                        if not tip_reached:
                            target_side = opposite_side
                            prog_x = delta_x_downstream / step_size
                            prog_goal = prog_x
                        else:
                            target_side = side_ida
                            prog_x = delta_x_upstream / step_size
                            prog_goal = prog_x

                        is_on_correct_side = (self.pos[1] >= local_center_y) if target_side == 1.0 else (self.pos[1] <= local_center_y)
                        if not is_on_correct_side:
                            reward -= 6.0
                            if dy * target_side > 0:
                                reward += 2.0
                        else:
                            dist_prev = abs(self.prev_pos[1] - local_center_y)
                            dist_curr = abs(self.pos[1] - local_center_y)
                            dy_outward = (dist_curr - dist_prev) / step_size
                            dy_inward = (dist_prev - dist_curr) / step_size

                            if conc > 1.8 * threshold:
                                if dy_outward > 0.05:
                                    reward += 5.0 * dy_outward
                                else:
                                    reward -= 2.0
                                if dy_inward > 0.05:
                                    reward -= 4.0 * dy_inward
                                if prog_goal > 0.01:
                                    reward += 1.0 * prog_goal
                                elif prog_goal < -0.01:
                                    reward -= 3.0 * abs(prog_goal)
                            elif conc < 0.20 * threshold:
                                if dy_inward > 0.05:
                                    reward += 6.0 * dy_inward
                                else:
                                    reward -= 3.0
                                if dy_outward > 0.05:
                                    reward -= 6.0 * dy_outward
                                if abs(prog_goal) > 0.01:
                                    reward -= 2.5
                            else:
                                prev_err = abs(prev_conc - threshold)
                                curr_err = abs(conc - threshold)
                                delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                                log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                                adh = float(np.exp(-2.5 * log_ratio))

                                if prog_goal > 0.01:
                                    reward += 4.0 * prog_goal + 6.0 * adh
                                    if delta_contour > 0.01:
                                        reward += 2.0 * min(delta_contour, 2.0)
                                    elif conc > threshold and dy_outward > 0.05:
                                        reward += 2.0 * dy_outward
                                    elif conc < threshold and dy_inward > 0.05:
                                        reward += 2.0 * dy_inward
                                elif prog_goal < -0.01:
                                    reward -= 5.0 * abs(prog_goal)
                                else:
                                    if delta_contour > 0.01 or (conc > threshold and dy_outward > 0.05) or (conc < threshold and dy_inward > 0.05) or adh > 0.3:
                                        reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                                    else:
                                        reward -= 2.0

            if hit_boundary or np.linalg.norm(movement) < 1e-6:
                reward -= 4.0

            if is_reversal:
                reward -= 8.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

        # =========================================================================
        # ESTADO 2: FOLLOWING_BOUNDARY (Contorno ondulante da margem até à fonte)
        # =========================================================================
        elif self.state == "following_boundary":
            source_pos = np.array(self.config["source_position"])
            dist_to_source = np.linalg.norm(self.pos - source_pos)
            if dist_to_source < self.config["agent_config"]["success_distance"]:
                return 500.0  # Chegou à fonte com sucesso e inicia circling

            reward = self.reward_config["step_penalty"]
            step_size = self.speed * self.config["dt"]
            movement = self.pos - self.prev_pos
            dy = movement[1]
            delta_x = self.prev_pos[0] - self.pos[0]  # delta_x > 0 é Oeste / Esquerda (-x, em direção à fonte)

            side = getattr(self, "skirting_side_ida", 1.0)
            if side is None:
                side = 1.0

            local_center_y = get_plume_center_y(self.pos[0], self.time, self.plume, self.config)
            is_on_correct_side = (self.pos[1] >= local_center_y) if side == 1.0 else (self.pos[1] <= local_center_y)

            # Guia próximo à fonte: incentiva aproximação direta quando a distância é curta
            near_source_radius = self.config["agent_config"]["success_distance"] * 4.0
            if dist_to_source < near_source_radius:
                prev_dist_src = np.linalg.norm(self.prev_pos - source_pos)
                approach_delta = (prev_dist_src - dist_to_source) / step_size
                # Bónus crescente proporcional à proximidade da fonte e à taxa de aproximação
                proximity_factor = 1.0 - (dist_to_source / near_source_radius)
                if approach_delta > 0.01:
                    reward += 6.0 * proximity_factor * approach_delta
                elif approach_delta < -0.01:
                    reward -= 4.0 * proximity_factor * abs(approach_delta)

            if not is_on_correct_side:
                reward -= 10.0
                if dy * side > 0:
                    reward += 3.0
            else:
                prog_x = delta_x / step_size
                dist_prev = abs(self.prev_pos[1] - local_center_y)
                dist_curr = abs(self.pos[1] - local_center_y)
                dy_outward = (dist_curr - dist_prev) / step_size
                dy_inward = (dist_prev - dist_curr) / step_size

                if conc > 1.8 * threshold:
                    # Demasiado no interior/núcleo: incentivar saída para a margem
                    if dy_outward > 0.05:
                        reward += 6.0 * dy_outward
                    else:
                        reward -= 3.0
                    if dy_inward > 0.05:
                        reward -= 4.0 * dy_inward
                    if prog_x > 0.01:
                        reward += 1.0 * prog_x
                    elif prog_x < -0.01:
                        reward -= 3.0 * abs(prog_x)
                elif conc < 0.20 * threshold:
                    # Em água limpa fora da pluma: aproximação OBRIGATÓRIA à margem ondulante
                    if dy_inward > 0.05:
                        reward += 8.0 * dy_inward
                    else:
                        reward -= 4.0
                    if dy_outward > 0.05:
                        reward -= 6.0 * dy_outward
                    # Proibir estritamente avanço em água limpa sem contacto com o contorno da pluma
                    if prog_x > 0.01:
                        reward -= 10.0 * prog_x
                else:
                    # Medição matemática da convergência ao contorno do puff (erro de concentração):
                    prev_err = abs(prev_conc - threshold)
                    curr_err = abs(conc - threshold)
                    delta_contour = (prev_err - curr_err) / max(threshold, 1e-4)

                    log_ratio = abs(np.log10(max(conc, 1e-4) / threshold))
                    adh = float(np.exp(-2.5 * log_ratio))

                    if prog_x > 0.01:
                        reward += 4.0 * prog_x + 6.0 * adh
                        if delta_contour > 0.01:
                            reward += 2.0 * min(delta_contour, 2.0)
                        elif conc > threshold and dy_outward > 0.05:
                            reward += 2.0 * dy_outward
                        elif conc < threshold and dy_inward > 0.05:
                            reward += 2.0 * dy_inward
                    elif prog_x < -0.01:
                        reward -= 5.0 * abs(prog_x)
                    else:
                        # Passo lateral (ação 0 ou 1) para acompanhamento da ondulação sinusoidal
                        if delta_contour > 0.01 or (conc > threshold and dy_outward > 0.05) or (conc < threshold and dy_inward > 0.05) or adh > 0.3:
                            reward += 6.0 * adh + 3.0 * max(0.0, delta_contour)
                        else:
                            reward -= 2.0

            if hit_boundary or np.linalg.norm(movement) < 1e-6:
                reward -= 4.0

            if is_reversal:
                reward -= 8.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

        # =========================================================================
        # ESTADO 1: SEARCHING (Movimentação pura em traça / Cast-and-Surge)
        # =========================================================================
        else:
            reward = self.reward_config["step_penalty"]
            step_size = self.speed * self.config["dt"]
            movement = self.pos - self.prev_pos
            movement_norm = np.linalg.norm(movement)

            # 1. Bónus de Primeira Deteção da Pluma (Interceção direta da pluma)
            if conc > threshold:
                return 50.0  # Forte recompensa imediata por intercetar a pluma através do varrimento

            # 2. Recompensa estrita para Movimento em Traça (Cast):
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]
            target_y = y_max_sweep if getattr(self, "cast_dir", 0) == 0 else y_min_sweep
            dist_before = abs(self.prev_pos[1] - target_y)
            dist_after = abs(self.pos[1] - target_y)
            delta_sweep = dist_before - dist_after
            dx = self.pos[0] - self.prev_pos[0]

            # A. Varrimento transversal (Cast): recompensa contínua ao aproximar-se da margem oposta
            if delta_sweep > 0.01:
                reward += 6.0 * (delta_sweep / step_size)
                # Bónus adicional de avanço contra o vento APENAS quando em varrimento transversal efetivo
                if dx < -0.01:
                    reward += 3.0 * (-dx / step_size)
            elif delta_sweep < -0.01:
                # Penalização por afastar-se da margem-alvo do varrimento (inversão vertical errada)
                reward -= 10.0 * (abs(delta_sweep) / step_size)
            else:
                reward -= 4.0

            # B. Penalização de avanço a favor do vento (recuo longitudinal no searching)
            if dx > 0.01:
                reward -= 8.0 * (dx / step_size)

            # C. Penalização estrita por movimentos puramente lineares sem varrimento (Ação 2 e 3)
            # No estado searching, o agente deve estritamente varrer transversalmente (traça)
            if action == 2:
                reward -= 15.0
            elif action == 3:
                reward -= 18.0

            if hit_boundary or movement_norm < 1e-6:
                reward -= 4.0

            if is_reversal and (0.32 * self.config["world_height"] < self.pos[1] < 0.68 * self.config["world_height"]):
                reward -= 10.0

            if anti_stall_penalty > 0.0:
                reward -= anti_stall_penalty

            if visitas_nesta_celula > 8:
                reward -= 0.6 * min(visitas_nesta_celula - 8, 10)

            if self.steps_in_current_cell > 15:
                reward -= 0.5 * min(self.steps_in_current_cell - 15, 4)

            reward -= self.reward_config.get("battery_drain_penalty_factor", 0.5) * self.last_drain
            return reward

    def get_current_plume_center_y(self, x=None):
        if x is None:
            x = self.pos[0]
        return get_plume_center_y(x, self.time, self.plume, self.config)

    def _get_obs(self):
        conc_normalized = np.clip(self.current_conc / self.conc_max, 0, 1)

        vy = self.plume.meander_generator.meandering_vy
        flow = np.array(self.config["mean_wind_velocity"]) + np.array([0.0, vy])

        flow_x_normalized = np.clip(flow[0] / self.flow_max, -1, 1)
        flow_y_normalized = np.clip(flow[1] / self.flow_max, -1, 1)

        obs_x = self.pos[0] / self.config["world_width"]
        obs_y = self.pos[1] / self.config["world_height"]
        conc_val = conc_normalized

        if self.state == "searching":
            conc_val = 0.0
            state_val = 0.0
            border_side_or_cast = float(getattr(self, "cast_dir", 0))
        elif self.state == "following_boundary":
            state_val = 1.0
            border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
        else:  # returning
            is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]
            det_pos = self.first_detection_pos if self.first_detection_pos is not None else getattr(self, "start_pos", np.array([90.0, 25.0]))

            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X da deteção, transita diretamente para docking
                if getattr(self, "return_tip_reached", False) or self.pos[0] >= det_pos[0] - 0.5:
                    state_val = 3.0
                    border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                else:
                    state_val = 2.0
                    ret_side = -getattr(self, "skirting_side_ida", 1.0) if getattr(self, "skirting_side_ida", 1.0) is not None else -1.0
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0
            else:
                # Pluma Linear: contorno completo até à ponta distal e regresso pela margem de ida
                if getattr(self, "return_tip_reached", False):
                    state_val = 3.0
                    if self.pos[0] <= det_pos[0] + 0.5:
                        border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                    else:
                        border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
                else:
                    state_val = 2.0
                    ret_side = -getattr(self, "skirting_side_ida", 1.0) if getattr(self, "skirting_side_ida", 1.0) is not None else -1.0
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0

        battery_level = np.clip(self.battery / self.battery_capacity, 0.0, 1.0)

        return np.array(
            [
                obs_x,
                obs_y,
                conc_val,
                flow_x_normalized,
                flow_y_normalized,
                state_val,
                battery_level,
                border_side_or_cast
            ],
            dtype=np.float32,
        )

    def print_maze_info(self):
        """Print environment information"""
        print(
            f"Continuous Plume Environment: {self.config['world_width']}x{self.config['world_height']}"
        )
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")


class QLearningSimAgent(Agent):
    """O Agente que utiliza o modelo Q-Learning para tomar decisões no Simulador principal."""

    def __init__(self, config):
        if "agent_radius" not in config and "agent_config" in config:
            config = {**config, **config.get("agent_config", {})}
        super().__init__(config)
        dummy_env = PlumeRLWrapper(config)

        self.q_agent = RLQLearningAgent(dummy_env)
        self.model_name = "Desconhecido"

        # 1. Se foi passado o agente já treinado diretamente em memória:
        if config.get("trained_q_agent") is not None:
            self.q_agent = config["trained_q_agent"]
            self.model_name = getattr(self.q_agent, "loaded_model_name", "Treino Atual (Em Memória)")
            print(f"\n[+] Simulador: Q-Table em memória conectada ({len(self.q_agent.q_table)} estados aprendidos).")
        else:
            # 2. Carregar ficheiro .pkl especificado com resolução robusta de caminho
            chosen_model = self.config.get("model_to_load", "qlearning_plume.pkl")
            caminhos_tentar = [
                chosen_model,
                os.path.join("modelos", chosen_model),
                f"{chosen_model}.pkl" if not chosen_model.endswith(".pkl") else chosen_model,
                os.path.join("modelos", f"{chosen_model}.pkl") if not chosen_model.endswith(".pkl") else os.path.join("modelos", chosen_model),
            ]
            carregou = False
            for p in caminhos_tentar:
                if p and os.path.exists(p):
                    if self.q_agent.load(p):
                        self.model_name = os.path.basename(p)
                        self.q_agent.loaded_model_name = self.model_name
                        carregou = True
                        saved_meander = getattr(self.q_agent, "meander_model", None)
                        if saved_meander:
                            self.config["meander_model"] = saved_meander
                        elif "sinusoid" in self.model_name.lower():
                            self.config["meander_model"] = "sum_of_sinusoids"
                        else:
                            self.config["meander_model"] = "linear"
                        print(f"\n[+] Simulador: Modelo '{p}' carregado com sucesso ({len(self.q_agent.q_table)} estados na Q-Table). Tipo de pluma: {self.config.get('meander_model')}.")
                        break
            if not carregou:
                self.model_name = os.path.basename(chosen_model) if chosen_model else "Vazio"
                print(f"\n[-] AVISO: Não foi possível encontrar o ficheiro do modelo '{chosen_model}'. Q-Table iniciada a zeros ({len(self.q_agent.q_table)} estados)!")

    def reset(self):
        super().reset()
        self.current_action = None
        self.cast_dir = int(np.random.choice([0, 1]))
        self.return_at_detection_x = False
        self.return_contour_completed = False
        self.distal_tip_entered = False
        self.theta = None
        self.circling_step = 0
        self.circling_angle = 0.0
        self.recent_sim_positions = []
        self.recent_sim_actions = []

    def run_search_algorithm(self, dt):
        """
        Executa as decisões do agente no simulador COM BASE NA Q-TABLE TREINADA,
        com proteção de momento e anti-chattering para evitar aprisionamento determinístico
        em ciclos de 2 estados / reversões de 180° entre células vizinhas.
        """
        conc_thresh = float(self.config.get("agent_config", {}).get("agent_concentration_threshold", 1.0))
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.pos - source_pos)
        success_distance = float(self.config.get("agent_config", {}).get("success_distance", 2.0))

        # --- Máquina de Estados da Missão ---
        if self.state == "searching":
            y_min_sweep = 0.30 * self.config["world_height"]
            y_max_sweep = 0.70 * self.config["world_height"]
            if self.pos[1] >= y_max_sweep:
                self.cast_dir = 1
            elif self.pos[1] <= y_min_sweep:
                self.cast_dir = 0

            if self.current_concentration > conc_thresh:
                self.state = "following_boundary"
                self.first_detection_pos = self.pos.copy()
                local_center_y = get_plume_center_y(self.pos[0], self.simulation_time, self.plume, self.config)
                self.skirting_side_ida = 1.0 if self.pos[1] >= local_center_y else -1.0

        elif self.state == "following_boundary":
            if (dist_to_source < success_distance or (self.pos[0] <= source_pos[0] + 1.8 and dist_to_source < success_distance * 1.8)) and not self.source_reached:
                self.source_reached = True
                self.circling_step = 0

        # Contorno circular da fonte
        if self.state == "following_boundary" and self.source_reached:
            target_pos = self.get_circling_target(dt)
            return target_pos, {"action": "circling", "state": "circling_source"}


        # Estado Returning: verificação do alcance da ponta e conclusão de contorno
        if self.state == "returning":
            det_pos = self.first_detection_pos if self.first_detection_pos is not None else np.array([90.0, 25.0])
            side_ida = getattr(self, "skirting_side_ida", 1.0)
            if side_ida is None:
                side_ida = 1.0
            local_center_y = get_plume_center_y(self.pos[0], self.simulation_time, self.plume, self.config)
            is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]

            dist_to_det = np.linalg.norm(self.pos - det_pos)

            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X do ponto de deteção, desloca-se diretamente para det_pos!
                opposite_side = -side_ida
                is_on_opposite_side = (self.pos[1] >= local_center_y) if opposite_side == 1.0 else (self.pos[1] <= local_center_y)
                if self.pos[0] >= det_pos[0] - 0.5 and (is_on_opposite_side or self.current_concentration >= 0.15 * conc_thresh):
                    self.return_tip_reached = True
                    self.return_contour_completed = True

                if dist_to_det < success_distance:
                    self.mission_complete = True
                    self.return_contour_completed = True

                if getattr(self, "return_tip_reached", False):
                    goal_vec = det_pos - self.pos
                    goal_norm = np.linalg.norm(goal_vec)
                    move_vec = goal_vec / goal_norm if goal_norm > 1e-6 else np.array([0.0, 0.0])

                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    action = 1 if det_pos[1] > self.pos[1] else 0
                    return target_pos, {"action": action, "theta": self.theta}
            else:
                # Pluma Linear: contorno completo de toda a pluma até à ponta distal (x_turn ~80m)
                x_turn = min(80.0, max(det_pos[0] + 16.0, 78.0))

                if not getattr(self, "return_tip_reached", False):
                    is_on_ida_side = (self.pos[1] >= local_center_y) if side_ida == 1.0 else (self.pos[1] <= local_center_y)
                    if self.pos[0] >= x_turn:
                        self.distal_tip_entered = True
                    if getattr(self, "distal_tip_entered", False) and is_on_ida_side:
                        self.return_tip_reached = True

                if getattr(self, "return_tip_reached", False) and dist_to_det < success_distance:
                    self.mission_complete = True
                    self.return_contour_completed = True

                # Quando estiver no mesmo X que o ponto de deteção e após o contorno total da pluma linear:
                if getattr(self, "return_tip_reached", False) and self.pos[0] <= det_pos[0] + 0.5:
                    goal_vec = det_pos - self.pos
                    goal_norm = np.linalg.norm(goal_vec)
                    move_vec = goal_vec / goal_norm if goal_norm > 1e-6 else np.array([0.0, 0.0])

                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    action = 1 if det_pos[1] > self.pos[1] else 0
                    return target_pos, {"action": action, "theta": self.theta}

                # Contorno distal fluido da margem direita: viragem suave em redor dos puffs para mudar de lado:
                if getattr(self, "distal_tip_entered", False) and not getattr(self, "return_tip_reached", False):
                    turn_act = 1 if side_ida == 1.0 else 0
                    if self.current_concentration > 1.5 * conc_thresh:
                        action = 2  # passo em -x para contornar o bordo do puff pelo exterior
                    else:
                        action = turn_act

                    dirs = {0: np.array([0.0, -1.0]), 1: np.array([0.0, 1.0]), 2: np.array([-1.0, 0.0]), 3: np.array([1.0, 0.0])}
                    move_vec = dirs[action]
                    target_pos = self.pos + move_vec * self.speed * dt
                    target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
                    target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

                    desired_theta = np.arctan2(move_vec[1], move_vec[0])
                    if self.theta is None:
                        self.theta = desired_theta
                    else:
                        diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
                        self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

                    return target_pos, {"action": action, "theta": self.theta}

        # --- Vetor de Observação para a Q-Table ---
        obs_x = self.pos[0] / self.config["world_width"]
        obs_y = self.pos[1] / self.config["world_height"]
        conc_val = np.clip(self.current_concentration / 10.0, 0, 1)

        vy = self.measured_flow_direction[1]
        flow_x_normalized = np.clip(self.measured_flow_direction[0] / 5.0, -1, 1)
        flow_y_normalized = np.clip(vy / 5.0, -1, 1)

        if self.state == "searching":
            conc_val = 0.0
            state_val = 0.0
            border_side_or_cast = float(getattr(self, "cast_dir", 0))
        elif self.state == "following_boundary":
            state_val = 1.0
            border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
        else:
            is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]
            det_pos = self.first_detection_pos if self.first_detection_pos is not None else np.array([90.0, 25.0])
            if is_sinusoid:
                # Pluma Sinusoidal: ao alcançar o mesmo X de det_pos, transita para docking
                if getattr(self, "return_tip_reached", False) or self.pos[0] >= det_pos[0] - 0.5:
                    state_val = 3.0
                    border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                else:
                    state_val = 2.0
                    ret_side = -getattr(self, "skirting_side_ida", 1.0) if getattr(self, "skirting_side_ida", 1.0) is not None else -1.0
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0
            else:
                # Pluma Linear: contorno completo
                if getattr(self, "return_tip_reached", False):
                    state_val = 3.0
                    if self.pos[0] <= det_pos[0] + 0.5:
                        border_side_or_cast = 1.0 if det_pos[1] > self.pos[1] else 0.0
                    else:
                        border_side_or_cast = 1.0 if getattr(self, "skirting_side_ida", 1.0) == 1.0 else 0.0
                else:
                    state_val = 2.0
                    ret_side = -getattr(self, "skirting_side_ida", 1.0) if getattr(self, "skirting_side_ida", 1.0) is not None else -1.0
                    border_side_or_cast = 1.0 if ret_side == 1.0 else 0.0

        battery_level = np.clip(self.battery / self.battery_capacity, 0.0, 1.0)

        obs = np.array(
            [
                obs_x,
                obs_y,
                conc_val,
                flow_x_normalized,
                flow_y_normalized,
                state_val,
                battery_level,
                border_side_or_cast
            ],
            dtype=np.float32,
        )

        # Escolha da ação com base na Q-Table aprendida, com filtro de anti-bloqueio não-reversivo
        key = self.q_agent._state_key(obs)
        q_vals = self.q_agent.q_table.get(key, None)
        opposite_dirs = {0: 1, 1: 0, 2: 3, 3: 2}

        if q_vals is None or np.all(q_vals == 0):
            action = int(np.random.choice(4))
        else:
            sorted_acts = [int(a) for a in np.argsort(q_vals)[::-1]]
            best = sorted_acts[0]

            if self.state == "searching":
                action = best
            else:
                is_ping_pong = False
                if hasattr(self, "recent_sim_actions") and len(self.recent_sim_actions) >= 4:
                    acts = self.recent_sim_actions[-4:]
                    if acts[0] == acts[2] and acts[1] == acts[3] and acts[0] == opposite_dirs.get(acts[1]):
                        if hasattr(self, "recent_sim_positions") and len(self.recent_sim_positions) >= 4:
                            disp = np.linalg.norm(self.pos - self.recent_sim_positions[-4])
                            if disp < 0.15:
                                is_ping_pong = True

                is_stalled = False
                if hasattr(self, "recent_sim_positions") and len(self.recent_sim_positions) >= 20:
                    disp = np.linalg.norm(self.pos - self.recent_sim_positions[-20])
                    if disp < 0.20:
                        is_stalled = True

                if is_ping_pong or is_stalled:
                    reversing_act = opposite_dirs.get(self.recent_sim_actions[-1]) if hasattr(self, "recent_sim_actions") and self.recent_sim_actions else None
                    action = best
                    for alt in sorted_acts:
                        if alt != reversing_act:
                            action = alt
                            break
                else:
                    action = best

                # Quando próximo da fonte em contorno de ida, proibir recuo a jusante (ação 3)
                if self.state == "following_boundary" and dist_to_source < success_distance * 3.0:
                    if action == 3:
                        for alt in sorted_acts:
                            if alt != 3:
                                action = alt
                                break

                # No estado returning, proibir recuo a montante em direção à fonte (ação 2)
                if self.state == "returning" and action == 2:
                    for alt in sorted_acts:
                        if alt != 2:
                            action = alt
                            break

        self.current_action = action
        if not hasattr(self, "recent_sim_actions"):
            self.recent_sim_actions = []
        self.recent_sim_actions.append(action)
        if len(self.recent_sim_actions) > 25:
            self.recent_sim_actions.pop(0)

        if not hasattr(self, "recent_sim_positions"):
            self.recent_sim_positions = []
        self.recent_sim_positions.append(self.pos.copy())
        if len(self.recent_sim_positions) > 25:
            self.recent_sim_positions.pop(0)

        if self.state == "searching":
            search_angle_deg = float(self.config.get("agent_config", {}).get("search_angle", 75.0))
            crab_upwind = float(np.cos(np.deg2rad(search_angle_deg)))
            crosswind = float(np.sin(np.deg2rad(search_angle_deg)))
            dirs = {
                0: np.array([-crab_upwind, -crosswind]),
                1: np.array([-crab_upwind, crosswind]),
                2: np.array([-1.0, 0.0]),
                3: np.array([1.0, 0.0]),
            }
        else:
            dirs = {
                0: np.array([0.0, -1.0]),
                1: np.array([0.0, 1.0]),
                2: np.array([-1.0, 0.0]),
                3: np.array([1.0, 0.0]),
            }

        move_vec = dirs[action]
        target_pos = self.pos + move_vec * self.speed * dt
        target_pos[0] = np.clip(target_pos[0], 0.0, self.config["world_width"])
        target_pos[1] = np.clip(target_pos[1], 0.0, self.config["world_height"])

        # Orientação visual contínua para renderização realista do sprite
        desired_theta = np.arctan2(move_vec[1], move_vec[0])
        if self.theta is None:
            self.theta = desired_theta
        else:
            diff = (desired_theta - self.theta + np.pi) % (2 * np.pi) - np.pi
            self.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

        return target_pos, {"action": action, "theta": self.theta}



AGENT_REGISTRY = {
    "QLearningAgent": QLearningSimAgent,
}
# ======================================================================================
# 4. SIMULADOR GRÁFICO 2D E ORQUESTRADOR DE ENSAIOS
# ======================================================================================


class Simulator:
    """
    Orquestrador principal da simulação física e visualização gráfica em tempo real.
    Gere o ciclo de vida dos ensaios, a integração temporal do fluido, os sensores
    do veículo autónomo e a renderização do painel HUD marinho em Pygame / OpenCV.
    """

    def __init__(self, config):
        """Inicializa o simulador, instancia o agente configurado e prepara a janela visual."""
        self.config = config
        agent_class_name = config["agent_class"]
        if agent_class_name not in AGENT_REGISTRY:
            raise ValueError(f"Classe de agente '{agent_class_name}' não registada.")
        agent_full_config = {**config, **config.get("agent_config", {})}
        self.agent = AGENT_REGISTRY[agent_class_name](agent_full_config)
        if hasattr(self.agent, "config") and "meander_model" in self.agent.config:
            self.config["meander_model"] = self.agent.config["meander_model"]
        self.plume, self.logger = None, None
        self.simulation_time, self.current_experiment = 0.0, 0
        self.loop_time_ema, self.ema_alpha = 0.0, 0.1
        self.is_paused = False
        self.user_quit = False
        self.skip_episode = False

        # Configurar episódios a apresentar (últimos 5 episódios gravados do modelo selecionado)
        q_agent_ref = getattr(self.agent, "q_agent", None)
        self.saved_paths = getattr(q_agent_ref, "saved_paths", {})
        self.saved_detection_points = getattr(q_agent_ref, "saved_detection_points", {})

        all_eps = sorted(self.saved_paths.keys()) if self.saved_paths else []
        if len(all_eps) >= 5:
            self.episodes_to_show = all_eps[-5:]
        else:
            self.episodes_to_show = all_eps

        self.current_ep_num = None
        self.current_ep_path = None
        self.current_path_step = 0

        if config["visualize"]:
            self._init_visualization()

    def reset_experiment(self):
        """Reinicializa as variáveis de ambiente, posicionamento e registo para um novo ensaio."""
        if hasattr(self.agent, "config") and "meander_model" in self.agent.config:
            self.config["meander_model"] = self.agent.config["meander_model"]
        self.logger = DataLogger(self.config, self.current_experiment)
        self.simulation_time = 0.0
        self.is_paused = False
        self.skip_episode = False

        if hasattr(self, "episodes_to_show") and self.episodes_to_show and self.current_experiment < len(self.episodes_to_show):
            self.current_ep_num = self.episodes_to_show[self.current_experiment]
            self.current_ep_path = self.saved_paths.get(self.current_ep_num, None)
            self.current_path_step = 0

            det_pt = self.saved_detection_points.get(self.current_ep_num, None)
            self.saved_ep_detection_pt = np.array(det_pt, dtype=np.float64) if det_pt is not None else None
            self.agent.first_detection_pos = None

            total_show = len(self.episodes_to_show)
            ep_idx_str = f"{self.current_experiment + 1}/{total_show}"
            model_disp = getattr(self.agent, 'model_name', 'Modelo')
            print(f"\n[Simulator] A apresentar o Episódio #{self.current_ep_num} ({ep_idx_str}) de '{model_disp}' ({len(self.current_ep_path)} passos)...")
        else:
            self.current_ep_num = self.current_experiment + 1
            self.current_ep_path = None
            self.current_path_step = 0
            self.saved_ep_detection_pt = None
            self.agent.first_detection_pos = None

        # Sincronização e reprodutibilidade estocástica: seed determinístico por episódio
        seed_val = int(self.current_ep_num) * 1009 + 42
        np.random.seed(seed_val)

        self.plume = FilamentPlume(self.config)
        self.agent.plume = self.plume
        self.agent.reset()
        self.config["agent_start_region_center"] = self.agent.config["agent_start_region_center"]

        # Aquecimento inicial idêntico ao ambiente de treino (8.0 segundos)
        # para sincronização temporal e de fase absoluta da pluma com a trajetória gravada
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        self.plume_time = 0.0
        if getattr(self, "current_ep_path", None) is not None:
            for _ in range(int(initial_wait_time / self.config["dt"])):
                self.plume.update(self.config["dt"], self.plume_time)
                self.plume_time += self.config["dt"]

        if self.current_ep_path and len(self.current_ep_path) > 0:
            p0 = self.current_ep_path[0]
            self.agent.pos = np.array([p0[0] * self.config["world_width"], p0[1] * self.config["world_height"]], dtype=np.float64)
            self.agent.start_pos = self.agent.pos.copy()

    def step(self):
        dt = self.config["dt"]

        if getattr(self, "skip_episode", False):
            print(f"\n[Simulator] Episódio #{self.current_ep_num} avançado para o próximo.")
            return True

        # Caso 1: Apresentação dos episódios gravados do modelo selecionado
        if getattr(self, "current_ep_path", None) is not None:
            step_idx = self.current_path_step
            if step_idx >= len(self.current_ep_path):
                print(f"\n[Simulator] Episódio #{self.current_ep_num} concluído com sucesso!")
                return True

            norm_pos = self.current_ep_path[step_idx]
            world_w = self.config["world_width"]
            world_h = self.config["world_height"]
            new_pos = np.array([norm_pos[0] * world_w, norm_pos[1] * world_h], dtype=np.float64)
            old_pos = self.agent.pos.copy()

            if step_idx > 0:
                prev_norm = self.current_ep_path[step_idx - 1]
                delta_p = np.array([norm_pos[0] - prev_norm[0], norm_pos[1] - prev_norm[1]])
                norm_d = np.linalg.norm(delta_p)
                if norm_d > 1e-5:
                    self.agent.direction = delta_p / norm_d
                    desired_theta = np.arctan2(delta_p[1], delta_p[0])
                    if self.agent.theta is None:
                        self.agent.theta = desired_theta
                    else:
                        diff = (desired_theta - self.agent.theta + np.pi) % (2 * np.pi) - np.pi
                        self.agent.theta += np.clip(diff, -6.0 * dt, 6.0 * dt)

            self.agent.pos = new_pos
            self.current_path_step += 1

            self.plume.update(dt, self.plume_time)
            self.plume_time += dt
            self.agent.simulation_time = self.plume_time
            self.agent.plume = self.plume
            self.agent.measure_concentration(self.plume)
            self.agent.measure_flow(self.plume)

            # Consumo real de bateria baseado no deslocamento e vento
            displacement = new_pos - old_pos
            dist_moved = np.linalg.norm(displacement)
            direction = displacement / dist_moved if dist_moved > 1e-6 else np.array([0.0, 0.0])
            flow_vec = self.agent.measured_flow_direction
            opposing_wind = -flow_vec
            wind_resistance = np.dot(direction, opposing_wind)
            base_decay = self.agent.battery_base_decay
            wind_factor = self.agent.battery_wind_factor
            min_decay = self.agent.battery_min_decay
            drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
            self.agent.battery = max(0.0, self.agent.battery - drain_rate * dt)

            # Máquina de estados visual do ensaio
            source_pos = np.array(self.config["source_position"])
            dist_to_source = np.linalg.norm(self.agent.pos - source_pos)
            det_pos = getattr(self.agent, "first_detection_pos", None)
            total_steps = max(1, len(self.current_ep_path))

            if det_pos is None:
                conc_thresh = self.config.get("agent_config", {}).get("agent_concentration_threshold", 1.0)
                det_reached = False
                if getattr(self, "saved_ep_detection_pt", None) is not None:
                    if np.linalg.norm(self.agent.pos - self.saved_ep_detection_pt) <= 1.8:
                        det_reached = True

                if self.agent.current_concentration > conc_thresh or det_reached:
                    self.agent.first_detection_pos = self.saved_ep_detection_pt.copy() if getattr(self, "saved_ep_detection_pt", None) is not None else self.agent.pos.copy()
                    self.agent.state = "following_boundary"
                    print(f"\n[Simulator] Pluma DETETADA em ({self.agent.first_detection_pos[0]:.1f}m, {self.agent.first_detection_pos[1]:.1f}m)! A iniciar contorno da pluma até à fonte.")
                else:
                    self.agent.state = "searching"
            else:
                dist_to_det = np.linalg.norm(self.agent.pos - det_pos)
                if not getattr(self.agent, "source_reached", False):
                    if dist_to_source <= 2.2:
                        self.agent.source_reached = True
                        self.agent.state = "following_boundary"
                    else:
                        self.agent.state = "following_boundary"
                else:
                    if dist_to_det <= 2.0 and step_idx > int(0.65 * total_steps):
                        self.agent.mission_complete = True
                        self.agent.state = "returning"
                    else:
                        self.agent.state = "returning"

            custom_data = {"state": self.agent.state, "ep_num": self.current_ep_num}
            self.logger.log_step(
                self.simulation_time,
                self.agent.pos.copy(),
                self.agent.current_concentration,
                self.agent.state,
                self.agent.battery,
                custom_data,
            )
            self.simulation_time += dt

            if self.agent.battery <= 0:
                print(f"\n[Simulator] Bateria ESGOTADA no Episódio #{self.current_ep_num}!")
                return True
            if self.agent.mission_complete and step_idx >= len(self.current_ep_path) - 3:
                print(f"\n[Simulator] SUCESSO! Episódio #{self.current_ep_num} concluído com êxito!")
                return True
            return False

        # Caso 2: Simulação dinâmica em tempo real (fallback se o modelo não tiver rotas gravadas)
        if self.agent.battery <= 0:
            print(f"\n[Simulator] Agent started step with depleted battery! Mission failed.")
            return True

        self.plume.update(dt, self.simulation_time)
        self.agent.simulation_time = self.simulation_time
        self.agent.plume = self.plume
        self.agent.measure_concentration(self.plume)
        self.agent.measure_flow(self.plume)

        # Esperar 8 segundos no início de cada episódio antes de iniciar o movimento
        initial_wait_time = self.config.get("agent_initial_wait_time", 8.0)
        if self.simulation_time < initial_wait_time:
            self.agent.state = "waiting"
            custom_data = {"state": "waiting"}
            self.logger.log_step(
                self.simulation_time,
                self.agent.pos.copy(),
                self.agent.current_concentration,
                self.agent.state,
                self.agent.battery,
                custom_data,
            )
            self.simulation_time += dt
            return False
        elif self.agent.state == "waiting":
            self.agent.state = "searching"

        conc_thresh = self.config["agent_config"].get("agent_concentration_threshold", 1.0)
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.agent.pos - source_pos)
        success_distance = self.config["agent_config"].get("success_distance", 2.0)

        # --- LOW BATTERY SAFETY CHECK (<= 15%) ---
        battery_pct = (self.agent.battery / self.agent.battery_capacity) * 100.0
        is_low_battery = (battery_pct <= 15.0) or getattr(self.agent, "low_battery_direct_return", False) or getattr(self.agent, "low_battery_start_return", False) or getattr(self.agent, "low_battery_aborted", False)

        if is_low_battery:
            self.agent.mission_complete = False
            self.agent.low_battery_aborted = True

        # 1. State transition from searching -> following_boundary
        if self.agent.state == "searching":
            if self.agent.current_concentration > conc_thresh:
                self.agent.state = "following_boundary"
                self.agent.first_detection_pos = self.agent.pos.copy()
                self.agent.theta = None  # Reset heading for following_boundary
                local_center_y = get_plume_center_y(self.agent.pos[0], self.simulation_time, self.plume, self.config)
                self.agent.skirting_side_ida = 1.0 if self.agent.pos[1] >= local_center_y else -1.0
                side_str = "TOP" if self.agent.skirting_side_ida == 1.0 else "BOTTOM"
                print(f"\n[Simulator] Plume DETECTED at {self.agent.first_detection_pos}! Side: {side_str}. Following boundary to source.")

        # 2. Check source arrival when following boundary
        elif self.agent.state == "following_boundary":
            if (dist_to_source < success_distance or (self.agent.pos[0] <= source_pos[0] + 1.8 and dist_to_source < success_distance * 1.8)) and not self.agent.source_reached:
                self.agent.source_reached = True
                self.agent.circling_step = 0
                print(f"\n[Simulator] Source REACHED! Starting circling/contouring maneuver.")

        # 3. Check mission completion when returning normally
        elif self.agent.state == "returning":
            if self.agent.first_detection_pos is not None:
                dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
                det_x = self.agent.first_detection_pos[0]
                side_ida = getattr(self.agent, "skirting_side_ida", 1.0)
                if side_ida is None:
                    side_ida = 1.0
                local_center_y = get_plume_center_y(self.agent.pos[0], self.simulation_time, self.plume, self.config)
                is_sinusoid = self.config.get("meander_model", "ou").lower() in ["sum_of_sinusoids", "sinusoid", "sinusoidal"]

                if not is_low_battery:
                    if is_sinusoid:
                        # Pluma Sinusoidal: ao alcançar o mesmo X do ponto de deteção, conclui o contorno da pluma:
                        opposite_side = -side_ida
                        is_on_opposite_side = (self.agent.pos[1] >= local_center_y) if opposite_side == 1.0 else (self.agent.pos[1] <= local_center_y)
                        if self.agent.pos[0] >= det_x - 0.5 and (is_on_opposite_side or self.agent.current_concentration >= 0.15 * conc_thresh):
                            self.agent.return_tip_reached = True
                            self.agent.return_contour_completed = True
                        if getattr(self.agent, "return_tip_reached", False) and dist_to_detection < success_distance:
                            self.agent.return_contour_completed = True
                    else:
                        # Pluma Linear: contorno completo até à ponta distal (x_turn ~80m) e regresso pela margem de ida:
                        x_turn = max(det_x + 8.0, 75.0)
                        x_turn = min(self.config["world_width"] - 6.0, x_turn)

                        # 1. Contorno a jusante da margem oposta (+x) até virar em redor da ponta para a margem de ida:
                        if not getattr(self.agent, "return_tip_reached", False):
                            is_on_ida_side = (self.agent.pos[1] >= local_center_y) if side_ida == 1.0 else (self.agent.pos[1] <= local_center_y)
                            if self.agent.pos[0] >= x_turn and is_on_ida_side:
                                self.agent.return_tip_reached = True

                        # 2. Contorno a montante da margem de ida (-x) até alcançar det_pos:
                        if getattr(self.agent, "return_tip_reached", False):
                            if dist_to_detection < success_distance:
                                self.agent.return_contour_completed = True

                contour_done = getattr(self.agent, "return_contour_completed", False)

                if dist_to_detection < success_distance and contour_done:
                    # O episódio SÓ é um sucesso se realizou todos os estados da forma correta (contornando toda a pluma e sem aborto por bateria)
                    if getattr(self.agent, "circling_completed", False) and not is_low_battery and (battery_pct > 15.0):
                        self.agent.mission_complete = True
                        print(f"\n[Simulator] SUCESSO! Missão concluída com todos os estados realizados com sucesso em {self.simulation_time:.2f}s (Bateria: {battery_pct:.1f}%)!")
                    else:
                        self.agent.mission_complete = False
                        self.agent.incomplete_mission = True
                        print(f"\n[Simulator] Chegou ao ponto de deteção, mas o episódio NÃO teve sucesso (missão incompleta ou retorno por bateria baixa: {battery_pct:.1f}%).")

        old_pos = self.agent.pos.copy()

        # CASO 1: No estado searching (ainda não detetou a pluma de todo), se bateria <= 15% -> Retornar ao ponto de início!
        if (self.agent.state in ["searching", "returning_to_start"] or getattr(self.agent, "low_battery_start_return", False)) and self.agent.first_detection_pos is None and is_low_battery:
            start_target = getattr(self.agent, "start_pos", np.array([90.0, 25.0]))
            if not getattr(self.agent, "low_battery_start_return", False):
                self.agent.low_battery_start_return = True
                self.agent.state = "returning_to_start"
                print(f"\n[Simulator] Bateria baixa no estado searching ({battery_pct:.1f}% <= 15%) sem detetar pluma! Retornando ao ponto de início {start_target}!")

            dist_to_start = np.linalg.norm(self.agent.pos - start_target)
            if dist_to_start < success_distance:
                # Regresso ao ponto de início por bateria baixa sem detetar a pluma: SEM SUCESSO!
                self.agent.mission_complete = False
                self.agent.low_battery_aborted = True
                print(f"\n[Simulator] Agente regressou ao ponto de início {self.agent.pos} com bateria baixa ({battery_pct:.1f}%). Episódio SEM SUCESSO (pluma não detetada).")
                custom_data = {"state": "returning_start_low_battery_failed", "battery_pct": battery_pct}
            else:
                goal_vec = start_target - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                if goal_norm > 1e-6:
                    self.agent.direction = goal_vec / goal_norm
                custom_data = {"state": "returning_start_low_battery", "battery_pct": battery_pct}
                self.agent.move(target_pos, dt)

        # CASO 2: Se já tinha detetado a pluma (em contorno ou regresso) e bateria <= 15% -> Retornar ao ponto de deteção!
        elif self.agent.state in ["following_boundary", "returning"] and self.agent.first_detection_pos is not None and is_low_battery:
            if not getattr(self.agent, "low_battery_direct_return", False):
                self.agent.low_battery_direct_return = True
                self.agent.state = "returning"
                print(f"\n[Simulator] Low battery safety trigger! Bateria <= 15% ({battery_pct:.1f}%). Retornando DIRETAMENTE ao ponto de deteção {self.agent.first_detection_pos}!")

            dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
            if dist_to_detection < success_distance:
                # Regresso por bateria baixa: NUNCA tem sucesso!
                self.agent.mission_complete = False
                self.agent.low_battery_aborted = True
                print(f"\n[Simulator] Agente regressou ao ponto de deteção {self.agent.pos} com bateria baixa ({battery_pct:.1f}%). Episódio SEM SUCESSO (não completou todos os estados).")
                custom_data = {"state": "returning_direct_low_battery_failed", "battery_pct": battery_pct}
            else:
                goal_vec = self.agent.first_detection_pos - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                if goal_norm > 1e-6:
                    self.agent.direction = goal_vec / goal_norm
                custom_data = {"state": "returning_direct_low_battery", "battery_pct": battery_pct}
                self.agent.move(target_pos, dt)

        elif self.agent.state == "following_boundary" and self.agent.source_reached:
            target_pos = self.agent.get_circling_target(dt)
            custom_data = {"state": "circling_source", "step": self.agent.circling_step}
            self.agent.move(target_pos, dt)

        elif isinstance(self.agent, QLearningSimAgent):
            # O agente Q-Learning navega ativamente em conformidade com as decisões aprendidas na sua Q-Table
            target_pos, custom_data = self.agent.run_search_algorithm(dt)
            self.agent.move(target_pos, dt)

        elif self.agent.state == "following_boundary":
            local_center_y = get_plume_center_y(self.agent.pos[0], self.simulation_time, self.plume, self.config)
            if self.agent.skirting_side_ida is None:
                self.agent.skirting_side_ida = 1.0 if self.agent.pos[1] >= local_center_y else -1.0
            side = self.agent.skirting_side_ida
            # Initialize theta if not present or None
            if not hasattr(self.agent, 'theta') or self.agent.theta is None or self.agent.theta == 0.0:
                self.agent.theta = np.pi - (np.pi / 4.0 * side)

            # Dynamic steering logic based on concentration (mesma forma do estado returning)
            omega = 3.2  # turning rate in rad/s, responsive to sinusoidal boundary changes
            if self.agent.current_concentration > conc_thresh:
                # Inside plume: turn away from plume interior
                if side == 1.0:
                    self.agent.theta -= omega * dt
                else:
                    self.agent.theta += omega * dt
            else:
                # Outside plume: turn towards plume interior
                if side == 1.0:
                    self.agent.theta += omega * dt
                else:
                    self.agent.theta -= omega * dt

            # Symmetric clip around upwind direction (pi)
            # Allows steering up to 75 deg (5*pi/12) up or down, hugging sinusoidal wave peaks and troughs
            theta_max = 5.0 * np.pi / 12.0
            self.agent.theta = np.clip(self.agent.theta, np.pi - theta_max, np.pi + theta_max)

            target_pos = self.agent.pos + np.array([np.cos(self.agent.theta), np.sin(self.agent.theta)]) * self.agent.speed * dt
            custom_data = {"state": "following_boundary_steering", "theta": self.agent.theta}
            self.agent.move(target_pos, dt)

        elif self.agent.state == "returning":
            opposite_side = -self.agent.skirting_side_ida if self.agent.skirting_side_ida is not None else -1.0
            local_center_y = get_plume_center_y(self.agent.pos[0], self.simulation_time, self.plume, self.config)
            on_returning_side = (self.agent.pos[1] < local_center_y) if opposite_side == -1.0 else (self.agent.pos[1] >= local_center_y)

            # Check if we reached the X coordinate of detection point after the contour has been completed
            reached_detection_x = False
            if self.agent.return_waypoint_reached and not on_returning_side:
                if self.agent.pos[0] <= self.agent.first_detection_pos[0]:
                    reached_detection_x = True

            if reached_detection_x:
                goal_vec = np.array([0.0, self.agent.first_detection_pos[1] - self.agent.pos[1]])
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                custom_data = {"state": "returning_vertical_final"}
                self.agent.move(target_pos, dt)
            else:
                # Initialize agent.theta if not present or 0.0
                if not hasattr(self.agent, 'theta') or self.agent.theta is None or self.agent.theta == 0.0:
                    self.agent.theta = -np.pi / 4.0 if opposite_side == -1.0 else np.pi / 4.0

                # Dynamic steering logic based on concentration
                omega = 2.4  # turning rate in rad/s
                if self.agent.current_concentration > conc_thresh:
                    # Inside: turn away from center
                    if opposite_side == -1.0:
                        self.agent.theta -= omega * dt
                    else:
                        self.agent.theta += omega * dt
                else:
                    # Outside: turn towards center
                    if opposite_side == -1.0:
                        self.agent.theta += omega * dt
                    else:
                        self.agent.theta -= omega * dt

                # Update return state
                if not self.agent.return_waypoint_reached:
                    if self.agent.pos[0] >= self.agent.first_detection_pos[0]:
                        if not plume_exists_at_x(self.agent.pos[0], self.plume, self.config):
                            self.agent.return_waypoint_reached = True
                    
                    if self.agent.pos[0] >= 98.0:
                        self.agent.return_waypoint_reached = True

                # Clip theta depending on returning phase to follow silhouette dynamically
                if not self.agent.return_waypoint_reached:
                    # Phase 1: downwind boundary following (limit heading to downwind directions)
                    theta_max_ret = 5.0 * np.pi / 12.0
                    self.agent.theta = np.clip(self.agent.theta, -theta_max_ret, theta_max_ret)
                else:
                    # Phase 2: crossing the end or returning upwind
                    if on_returning_side:
                        # Crossing the end of the plume: allow turning around the end (up to 180 degrees)
                        if opposite_side == -1.0:
                            self.agent.theta = np.clip(self.agent.theta, -np.pi/3, np.pi)
                        else:
                            self.agent.theta = np.clip(self.agent.theta, -np.pi, np.pi/3)
                    else:
                        # Returning on the original side (limit heading to upwind directions)
                        if opposite_side == -1.0:
                            self.agent.theta = np.clip(self.agent.theta, np.pi/2, 1.5 * np.pi)
                        else:
                            self.agent.theta = np.clip(self.agent.theta, -1.5 * np.pi, -np.pi/2)

                target_pos = self.agent.pos + np.array([np.cos(self.agent.theta), np.sin(self.agent.theta)]) * self.agent.speed * dt
                custom_data = {"state": "returning_steering", "theta": self.agent.theta}
                self.agent.move(target_pos, dt)
        else:
            target_pos, custom_data = self.agent.run_search_algorithm(dt)
            self.agent.move(target_pos, dt)

        # Calculate battery drain based on actual displacement
        displacement = self.agent.pos - old_pos
        dist_moved = np.linalg.norm(displacement)
        direction = displacement / dist_moved if dist_moved > 1e-6 else np.array([0.0, 0.0])

        flow_vec = self.agent.measured_flow_direction
        opposing_wind = -flow_vec
        wind_resistance = np.dot(direction, opposing_wind)

        base_decay = self.agent.battery_base_decay
        wind_factor = self.agent.battery_wind_factor
        min_decay = self.agent.battery_min_decay

        drain_rate = max(min_decay, base_decay + wind_factor * wind_resistance)
        drain = drain_rate * dt
        self.agent.battery = max(0.0, self.agent.battery - drain)

        self.logger.log_step(
            self.simulation_time,
            self.agent.pos.copy(),
            self.agent.current_concentration,
            self.agent.state,
            self.agent.battery,
            custom_data,
        )

        self.simulation_time += dt
        if self.agent.battery <= 0:
            print(f"\n[Simulator] Bateria ESGOTADA em {self.agent.pos}! Missão falhada em t={self.simulation_time:.2f}s.")
            return True
        if self.agent.mission_complete:
            print(f"\nSimulation complete! SUCESSO: completou todos os estados e contornou a pluma em {self.simulation_time:.2f}s!")
            return True
        if getattr(self.agent, "low_battery_aborted", False) or getattr(self.agent, "incomplete_mission", False):
            print(f"\nSimulation complete! Episódio terminado SEM SUCESSO (retorno antecipado por bateria baixa ou contorno incompleto).")
            return True
        return False

    def run(self):
        if hasattr(self, "episodes_to_show") and self.episodes_to_show:
            num_experiments = len(self.episodes_to_show)
        else:
            num_experiments = self.config.get("num_experiments", 5)

        stabilization_time = self.config.get("plume_stabilization_time", 0.0)
        for i in range(num_experiments):
            if getattr(self, "user_quit", False):
                break
            self.current_experiment = i
            self.reset_experiment()
            print(f"Stabilizing plume for {stabilization_time} seconds...")
            self.simulation_time = 0.0
            while self.simulation_time < stabilization_time:
                if getattr(self, "user_quit", False):
                    break
                self.plume.update(self.config["dt"], self.simulation_time)
                self.simulation_time += self.config["dt"]
                if self.config["visualize"]:
                    self._draw()

            if getattr(self, "user_quit", False):
                break

            print("Stabilization complete. Starting agent search...")
            self.simulation_time = 0.0
            step_counter = 0
            while self.simulation_time < self.config["max_time"]:
                if getattr(self, "user_quit", False):
                    break
                loop_start_time = time.time()
                if self.is_paused:
                    if self.config["visualize"]:
                        self._draw()
                    time.sleep(0.01)
                    continue

                if self.step():
                    break
                if self.config["visualize"]:
                    self._draw()
                elapsed_time = time.time() - loop_start_time
                if self.loop_time_ema == 0.0:
                    self.loop_time_ema = elapsed_time
                else:
                    self.loop_time_ema = (self.ema_alpha * elapsed_time) + (
                        (1 - self.ema_alpha) * self.loop_time_ema
                    )
                step_counter += 1
                if step_counter % 20 == 0:
                    hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
                    print(
                        f"  > Sim Time: {self.simulation_time:.1f}s, Running Freq: {hz:.1f} Hz",
                        end="\r",
                    )
                if self.config["visualize"]:
                    if self.config["real_time_pacing"]:
                        wait_time = self.config["dt"] - elapsed_time
                        if wait_time > 0:
                            time.sleep(wait_time)
                    else:
                        time.sleep(0.003)  # Atrasa ligeiramente a simulação (~300 FPS max)
            print()
            if self.logger:
                self.logger.save()
        print("-" * 40 + "\nAll experiments finished.")
        if self.config["visualize"]:
            cv2.destroyAllWindows()

    def _init_visualization(self):
        self.viz_w, self.viz_h = self.config["viz_width"], self.config["viz_height"]
        self.world_w, self.world_h = (
            self.config["world_width"],
            self.config["world_height"],
        )
        cv2.namedWindow("Odour Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Odour Simulation", self.viz_w, self.viz_h)

        self.agent_sprite = None
        if os.path.exists("boat_sprite.png"):
            img = cv2.imread("boat_sprite.png", cv2.IMREAD_UNCHANGED)
            if img is not None:
                agent_radius = self.config.get("agent_config", {}).get("agent_radius", 0.5)
                sprite_size_px = int((agent_radius * 8) / self.world_w * self.viz_w)
                h, w = img.shape[:2]
                if w > h:
                    new_w = sprite_size_px
                    new_h = int(h * (new_w / w))
                else:
                    new_h = sprite_size_px
                    new_w = int(w * (new_h / h))
                self.agent_sprite = cv2.resize(img, (new_w, new_h))

    def _world_to_pixel(self, pos):
        px = int(pos[0] / self.world_w * self.viz_w)
        py = int(self.viz_h - (pos[1] / self.world_h * self.viz_h))
        return px, py

    def _draw_axes(self, canvas):
        axis_color, font, font_scale, font_thickness = (
            (200, 200, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )
        cv2.line(
            canvas,
            self._world_to_pixel((0, 0)),
            self._world_to_pixel((self.world_w, 0)),
            axis_color,
            2,
        )
        cv2.line(
            canvas,
            self._world_to_pixel((0, 0)),
            self._world_to_pixel((0, self.world_h)),
            axis_color,
            2,
        )
        for i in range(6):
            x_w = (i / 5) * self.world_w
            x_p, y_p = self._world_to_pixel((x_w, 0))
            cv2.line(canvas, (x_p, y_p - 5), (x_p, y_p + 5), axis_color, 1)
            t_s = cv2.getTextSize(str(int(x_w)), font, font_scale, font_thickness)[0]
            cv2.putText(
                canvas,
                str(int(x_w)),
                (x_p - t_s[0] // 2, y_p + 25),
                font,
                font_scale,
                axis_color,
                font_thickness,
            )
        for i in range(6):
            y_w = (i / 5) * self.world_h
            x_p, y_p = self._world_to_pixel((0, y_w))
            cv2.line(canvas, (x_p - 5, y_p), (x_p + 5, y_p), axis_color, 1)
            t_s = cv2.getTextSize(str(int(y_w)), font, font_scale, font_thickness)[0]
            cv2.putText(
                canvas,
                str(int(y_w)),
                (x_p - t_s[0] - 15, y_p + t_s[1] // 2),
                font,
                font_scale,
                axis_color,
                font_thickness,
            )
        cv2.putText(
            canvas,
            "(0,0)",
            (
                self._world_to_pixel((0, 0))[0] + 10,
                self._world_to_pixel((0, 0))[1] - 10,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        max_x_t = f"({int(self.world_w)}, 0)"
        t_s = cv2.getTextSize(max_x_t, font, font_scale, font_thickness)[0]
        cv2.putText(
            canvas,
            max_x_t,
            (
                self._world_to_pixel((self.world_w, 0))[0] - t_s[0] - 10,
                self._world_to_pixel((self.world_w, 0))[1] - 10,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        max_y_t = f"(0, {int(self.world_h)})"
        t_s = cv2.getTextSize(max_y_t, font, font_scale, font_thickness)[0]
        cv2.putText(
            canvas,
            max_y_t,
            (
                self._world_to_pixel((0, self.world_h))[0] + 10,
                self._world_to_pixel((0, self.world_h))[1] + 50,
            ),
            font,
            font_scale,
            axis_color,
            font_thickness,
        )
        start_c = self._world_to_pixel(self.config["agent_start_region_center"])
        start_r = int(
            self.config["agent_start_region_radius"] / self.world_w * self.viz_w
        )
        cv2.circle(canvas, start_c, start_r, (0, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _draw(self):
        # 1. Create a canvas with marine blue background
        marine_bg_color = (110, 55, 20)  # BGR: tom de azul marinho
        canvas = np.full((self.viz_h, self.viz_w, 3), marine_bg_color, dtype=np.uint8)

        # Draw axes onto the canvas first
        canvas = self._draw_axes(canvas)

        # 2. Draw puffs
        if self.plume.puffs_array.shape[0] > 0:
            min_mass = self.config["initial_puff_mass"] * 0.01
            for puff in self.plume.puffs_array:
                if puff[3] <= min_mass:
                    continue
                pos_px, radius_px = (
                    self._world_to_pixel(puff[0:2]),
                    max(1, int(puff[2] / self.world_w * self.viz_w)),
                )
                color_val = min(
                    255, int(puff[3] / self.config["initial_puff_mass"] * 255)
                )
                cv2.circle(
                    canvas,
                    pos_px,
                    radius_px,
                    (color_val, color_val, color_val),
                    -1,
                    cv2.LINE_AA,
                )

        source_px = self._world_to_pixel(self.config["source_position"])
        cv2.drawMarker(canvas, source_px, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

        if self.agent.first_detection_pos is not None and self.agent.state not in ["searching", "waiting"]:
            detection_px = self._world_to_pixel(self.agent.first_detection_pos)
            cv2.circle(canvas, detection_px, 6, (0, 255, 255), -1, cv2.LINE_AA)  # Yellow solid dot
            cv2.putText(canvas, 'detection', (detection_px[0] + 10, detection_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        elif getattr(self.agent, "low_battery_start_return", False) and hasattr(self.agent, "start_pos"):
            start_px = self._world_to_pixel(self.agent.start_pos)
            cv2.circle(canvas, start_px, 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, 'start point', (start_px[0] + 10, start_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # 4. Draw agent
        conc_thresh = self.config.get("agent_config", {}).get(
            "agent_concentration_threshold", 1.0
        )
        agent_color = (
            (0, 255, 0)
            if self.agent.current_concentration > conc_thresh
            else (255, 0, 0)
        )
        agent_px = self._world_to_pixel(self.agent.pos)
        agent_radius_px = max(2, int(self.agent.radius / self.world_w * self.viz_w))

        if hasattr(self, 'agent_sprite') and self.agent_sprite is not None:
            sprite = self.agent_sprite

            # Não rodar a imagem, manter a original
            rotated_sprite = sprite
            bound_w = sprite.shape[1]
            bound_h = sprite.shape[0]

            if rotated_sprite.shape[2] == 4:
                bgr = rotated_sprite[:, :, :3]
                alpha = rotated_sprite[:, :, 3] / 255.0
            else:
                bgr = rotated_sprite
                alpha = np.ones(bgr.shape[:2])

            top_left_x = agent_px[0] - bound_w // 2
            top_left_y = agent_px[1] - bound_h // 2

            y1, y2 = max(0, top_left_y), min(canvas.shape[0], top_left_y + bound_h)
            x1, x2 = max(0, top_left_x), min(canvas.shape[1], top_left_x + bound_w)

            y1o, y2o = max(0, -top_left_y), min(bound_h, canvas.shape[0] - top_left_y)
            x1o, x2o = max(0, -top_left_x), min(bound_w, canvas.shape[1] - top_left_x)

            if y1 < y2 and x1 < x2:
                alpha_crop = alpha[y1o:y2o, x1o:x2o, np.newaxis]
                canvas[y1:y2, x1:x2] = (alpha_crop * bgr[y1o:y2o, x1o:x2o] + (1 - alpha_crop) * canvas[y1:y2, x1:x2]).astype(np.uint8)
        else:
            cv2.circle(canvas, agent_px, agent_radius_px, agent_color, 2, cv2.LINE_AA)
            arrow_end_px = self._world_to_pixel(
                self.agent.pos + self.agent.direction * self.agent.radius * 2.5
            )
            cv2.arrowedLine(canvas, agent_px, arrow_end_px, (255, 255, 0), 2, cv2.LINE_AA)

        # 5. Draw updated text info
        font_scale = 0.55
        hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
        flow_vec = self.agent.measured_flow_direction
        state_str = self.agent.state.upper()
        if getattr(self.agent, "low_battery_direct_return", False):
            state_str = "RETURNING (LOW BAT <15%)"
        elif getattr(self.agent, "low_battery_start_return", False):
            state_str = "RETURNING TO START (LOW BAT <=15%)"

        # Barra de fundo escura para o HUD no topo
        cv2.rectangle(canvas, (0, 0), (self.viz_w, 36), (15, 15, 20), -1)

        wind_arrow = "->" if flow_vec[0] >= 0 else "<-"
        total_eps = len(getattr(self, "episodes_to_show", [])) or 5
        curr_ep_idx = self.current_experiment + 1
        ep_prefix = f"Ep #{self.current_ep_num} ({curr_ep_idx}/{total_eps}) | " if getattr(self, "current_ep_num", None) is not None else f"Episódio {curr_ep_idx}/{total_eps} | "
        info_left = f"{ep_prefix}Time: {self.simulation_time:.1f}s | State: {state_str} | Conc: {self.agent.current_concentration:.2f} | Bat: {self.agent.battery:.1f}% | Vento: {wind_arrow} ({flow_vec[0]:.2f}, {flow_vec[1]:.2f}) m/s"
        info_right = f"Agent Pos: ({self.agent.pos[0]:.1f}, {self.agent.pos[1]:.1f})"
        hud_color = (0, 255, 255)  # Amarelo de alto contraste sobre o fundo marinho

        cv2.putText(
            canvas,
            info_left,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        t_s_r = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        cv2.putText(
            canvas,
            info_right,
            (self.viz_w - t_s_r[0] - 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            hud_color,
            1,
            cv2.LINE_AA,
        )

        # 6. Display
        cv2.imshow("Odour Simulation", canvas)
        wait_ms = 30 if self.is_paused else 1
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 32:  # Barra de espaço: comuta entre pausa e retomada
            self.is_paused = not self.is_paused
        elif key == ord("n") or key == ord("N"):  # Tecla 'n': avançar imediatamente para o próximo episódio
            self.skip_episode = True
            self.is_paused = False
        elif key == 27 or key == ord("q"):  # ESC (27) ou 'q': terminar e fechar a simulação
            self.user_quit = True
            self.is_paused = False
            self.simulation_time = self.config["max_time"]
            cv2.destroyAllWindows()

        try:
            if cv2.getWindowProperty("Odour Simulation", cv2.WND_PROP_VISIBLE) < 1:
                self.user_quit = True
                self.is_paused = False
                self.simulation_time = self.config["max_time"]
        except Exception:
            pass


# ======================================================================================
# 5. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    import glob
    import json
    import os
    import tkinter as tk
    from tkinter import ttk, messagebox

    DEFAULT_CONFIG = {
        "dt": 0.1,
        "max_time": 300.0,
        "save_path": "experiments",
        "experiment_name": "bio_inspired_test",
        "visualize": True,
        "real_time_pacing": False,
        "viz_width": 1200,
        "viz_height": 600,
        "world_width": 100.0,
        "world_height": 50.0,
        "source_position": (20.0, 25.0),
        "emission_rate": 5,
        "initial_puff_mass": 450.0,
        "puff_decay_rate": 0.02,
        "puff_expansion_rate": 0.20,
        "battery_capacity": 100.0,
        "battery_base_decay": 0.08,
        "battery_wind_factor": 0.80,
        "battery_min_decay": 0.02,
        "concentration_model": "gaussian",
        "num_experiments": 5,
        "plume_stabilization_time": 0.0,
        "agent_initial_wait_time": 8.0,
        "agent_start_region_center": (90.0, 25.0),
        "agent_start_region_radius": 2.0,
        # --- MEANDER MODEL CONFIGURATION ---
        "meander_model": "OU",
        "ou_meander_config": {
            "intensity": 0.4,
            "timescale": 0.15,
        },
        "sinusoid_meander_config": {
            "num_harmonics": 10,
            "v_var": 0.8,
            "integral_length_scale_L": 20.0,
            "drift_correction_timescale": 10,
            "amplitude": 1.5,  # Elegant, stable spatial wave sweep
            "frequency": 0.08,
        },
        # 'sinusoid_meander_config': {
        # 'num_harmonics': 10,
        # 'v_var': 0.4,  # <-- Strong, but not excessive
        # 'integral_length_scale_L': 25.0,
        # 'drift_correction_timescale': 45.0
        # },
        # --- TURBULENCE INTENSITY ---
        "turbulence_intensity": 0.15,
        # 'turbulence_intensity': 0.7,
        # --- MEAN WIND VELOCITY ---
        "mean_wind_velocity": (1.5, 0),
        "agent_class": "QLearningAgent",
        "agent_config": {
            "agent_radius": 0.5,
            "agent_speed": 2.4,
            "success_distance": 2.0,
            "agent_concentration_threshold": 1.0,
            "surge_length": 4.0,
            "cast_length": 25.0,
            "cast_angle": 75.0,
            "search_length": 25.0,
            "search_angle": 75.0,
            "cast_length_increase_factor": 0.1,
            "confirmation_steps": 3,
        },
    }

    import subprocess
    import sys


    config_file_path = "config.json"

    if not os.path.exists(config_file_path):
        with open(config_file_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)



    with open(config_file_path, "r") as f:
        CONFIG = json.load(f)

    pasta_modelos = "modelos"
    os.makedirs(pasta_modelos, exist_ok=True)
    modelos = glob.glob(os.path.join(pasta_modelos, "*.pkl"))
    modelos.sort(key=os.path.getmtime, reverse=True)
    nomes_modelos = [os.path.basename(m) for m in modelos]

    escolhas = {
            "acao": None,
            "tipo_treino": "novo",
            "modelo_alvo": None,
            "nome_guardar": None,
            "episodios": 3000,
    }
        # --- INÍCIO DA INTERFACE GRÁFICA (TEMA CLARO E PROFISSIONAL) ---
    root = tk.Tk()
    root.title("Lançador - Simulador de Pluma & RL")
    root.geometry("520x560")
    root.minsize(480, 520)

    # Paleta de Cores Clara
    BG_MAIN = "#f1f5f9"
    BG_CARD = "#ffffff"
    BORDER_COLOR = "#cbd5e1"
    TEXT_DARK = "#0f172a"
    TEXT_MUTED = "#64748b"
    PRIMARY_COLOR = "#0284c7"

    root.configure(bg=BG_MAIN)
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", "#0f172a")
    root.option_add("*TCombobox*Listbox.selectBackground", PRIMARY_COLOR)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    style = ttk.Style(root)
    style.theme_use("clam")

    # Configurações do TTK para Tema Claro
    style.configure(".", background=BG_MAIN, foreground=TEXT_DARK)
    style.configure("TFrame", background=BG_MAIN)
    style.configure("Card.TLabelframe", background=BG_CARD, bordercolor=BORDER_COLOR)
    style.configure(
        "Card.TLabelframe.Label",
        background=BG_CARD,
        foreground=TEXT_DARK,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure("TLabel", background=BG_CARD, foreground=TEXT_DARK, font=("Segoe UI", 9))
    style.configure("Muted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 8))
    style.configure("TRadiobutton", background=BG_CARD, foreground=TEXT_DARK, font=("Segoe UI", 9))
    style.configure("TEntry", fieldbackground="#ffffff", foreground=TEXT_DARK, insertcolor=TEXT_DARK)
    style.map("TEntry", fieldbackground=[("!disabled", "#ffffff"), ("disabled", "#f1f5f9")], foreground=[("!disabled", TEXT_DARK), ("disabled", TEXT_MUTED)])
    style.configure("TCombobox", fieldbackground="#ffffff", foreground=TEXT_DARK, background="#ffffff")
    style.map("TCombobox", fieldbackground=[("readonly", "#ffffff"), ("!disabled", "#ffffff")], foreground=[("readonly", TEXT_DARK), ("!disabled", TEXT_DARK)])

    style.configure(
        "Action.TButton",
        background=PRIMARY_COLOR,
        foreground="#ffffff",
        font=("Segoe UI", 10, "bold"),
        padding=8,
        borderwidth=0,
    )
    style.map("Action.TButton", background=[("active", "#0369a1")])

    # Variáveis do Tkinter
    var_acao = tk.StringVar(value="treinar")
    var_tipo_treino = tk.StringVar(value="novo")
    var_episodios = tk.StringVar(value="3000")
    var_nome_novo = tk.StringVar(value="")
    var_modelo = tk.StringVar()
    var_modelo_simular = tk.StringVar()

    if nomes_modelos:
        var_modelo.set(nomes_modelos[0])
        var_modelo_simular.set(nomes_modelos[0])

    def atualizar_interface(*args):
        """Ativa/Desativa/Mostra os frames dependendo da opção principal escolhida"""
        frame_treino.pack_forget()
        frame_simular.pack_forget()
        frame_dashboard.pack_forget()

        if var_acao.get() == "treinar":
            frame_treino.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            if var_tipo_treino.get() == "continuar":
                combo_modelo_treino.config(state="readonly")
                entry_nome_novo.config(state="disabled")
            else:
                combo_modelo_treino.config(state="disabled")
                entry_nome_novo.config(state="normal")
            btn_iniciar.config(text="Iniciar Treino")
        elif var_acao.get() == "simular":
            frame_simular.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            btn_iniciar.config(text="Iniciar Simulação")
        else:  # dashboard
            frame_dashboard.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
            btn_iniciar.config(text="Abrir Dashboard")

    def iniciar():
        """Guarda as opções e fecha a janela para o script continuar"""
        escolhas["acao"] = var_acao.get()

        if escolhas["acao"] == "dashboard":
            import matplotlib.pyplot as plt
            from dashboard import DashboardApp

            janela_dashboard = tk.Toplevel(root)
            app_dash = DashboardApp(janela_dashboard)

            def fechar_dashboard():
                plt.close("all")
                janela_dashboard.destroy()

            janela_dashboard.protocol("WM_DELETE_WINDOW", fechar_dashboard)
            return

        if escolhas["acao"] == "treinar":
            escolhas["episodios"] = int(var_episodios.get() or 3000)
            escolhas["tipo_treino"] = var_tipo_treino.get()
            if var_tipo_treino.get() == "continuar":
                escolhas["modelo_alvo"] = var_modelo.get()
                escolhas["nome_guardar"] = var_modelo.get()
            else:
                escolhas["modelo_alvo"] = None
                nome_digitado = var_nome_novo.get().strip()
                if nome_digitado:
                    import re
                    nome_limpo = re.sub(r'[\\/*?:"<>|]', "", nome_digitado).strip()
                    if not nome_limpo.endswith(".pkl"):
                        nome_limpo += ".pkl"
                    escolhas["nome_guardar"] = nome_limpo
                else:
                    escolhas["nome_guardar"] = None
        elif escolhas["acao"] == "simular":
            escolhas["modelo_alvo"] = var_modelo_simular.get()
            if not escolhas["modelo_alvo"]:
                messagebox.showerror("Erro", "Nenhum modelo selecionado para simular!")
                return

        root.destroy()

    # --- CABEÇALHO ---
    header_frame = tk.Frame(root, bg=BG_CARD, bd=1, relief="solid", padx=16, pady=10)
    header_frame.pack(fill=tk.X, padx=14, pady=(12, 6))

    lbl_logo = tk.Label(
        header_frame,
        text="Simulador de Pluma & Reinforcement Learning",
        font=("Segoe UI", 12, "bold"),
        bg=BG_CARD,
        fg=TEXT_DARK,
    )
    lbl_logo.pack(anchor="w")

    lbl_sublogo = tk.Label(
        header_frame,
        text="Navegação Robótica Subaquática",
        font=("Segoe UI", 9),
        bg=BG_CARD,
        fg=TEXT_MUTED,
    )
    lbl_sublogo.pack(anchor="w", pady=(1, 0))

    # --- MODO DE OPERAÇÃO ---
    frame_modo = ttk.LabelFrame(root, text="  Modo de Operação  ", style="Card.TLabelframe", padding=10)
    frame_modo.pack(fill=tk.X, padx=14, pady=6)

    ttk.Radiobutton(
        frame_modo,
        text="Treinar Modelo Q-Learning",
        variable=var_acao,
        value="treinar",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    ttk.Radiobutton(
        frame_modo,
        text="Executar Simulação Visual 2D",
        variable=var_acao,
        value="simular",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    ttk.Radiobutton(
        frame_modo,
        text="Abrir Dashboard",
        variable=var_acao,
        value="dashboard",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=3)

    # --- OPÇÕES DE TREINO (Dinâmico) ---
    frame_treino = ttk.LabelFrame(root, text="  Opções de Treino  ", style="Card.TLabelframe", padding=10)

    ttk.Radiobutton(
        frame_treino,
        text="Iniciar Novo Treino",
        variable=var_tipo_treino,
        value="novo",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=(2, 2))

    frame_novo_nome = tk.Frame(frame_treino, bg=BG_CARD)
    frame_novo_nome.pack(fill=tk.X, padx=18, pady=(1, 6))
    ttk.Label(frame_novo_nome, text="Nome do modelo:").pack(anchor=tk.W)
    entry_nome_novo = ttk.Entry(frame_novo_nome, textvariable=var_nome_novo, width=42)
    entry_nome_novo.pack(anchor=tk.W, pady=(2, 0))

    ttk.Radiobutton(
        frame_treino,
        text="Continuar Treino de um Modelo",
        variable=var_tipo_treino,
        value="continuar",
        command=atualizar_interface,
    ).pack(anchor=tk.W, pady=(4, 2))

    frame_continuar = tk.Frame(frame_treino, bg=BG_CARD)
    frame_continuar.pack(fill=tk.X, padx=18, pady=(1, 6))
    combo_modelo_treino = ttk.Combobox(
        frame_continuar,
        textvariable=var_modelo,
        values=nomes_modelos,
        state="disabled",
        width=39,
    )
    combo_modelo_treino.pack(anchor=tk.W)

    frame_ep = tk.Frame(frame_treino, bg=BG_CARD)
    frame_ep.pack(fill=tk.X, padx=4, pady=(6, 2))
    ttk.Label(frame_ep, text="Número de episódios:").pack(anchor=tk.W)
    entry_ep = ttk.Entry(frame_ep, textvariable=var_episodios, width=16)
    entry_ep.pack(anchor=tk.W, pady=(2, 0))

    # --- OPÇÕES DE SIMULAÇÃO (Dinâmico) ---
    frame_simular = ttk.LabelFrame(root, text="  Opções de Simulação  ", style="Card.TLabelframe", padding=10)

    ttk.Label(frame_simular, text="Modelo a carregar:").pack(anchor=tk.W, pady=(2, 4))
    combo_modelo_sim = ttk.Combobox(
        frame_simular,
        textvariable=var_modelo_simular,
        values=nomes_modelos,
        state="readonly",
        width=44,
    )
    combo_modelo_sim.pack(anchor=tk.W, pady=(0, 4))

    ttk.Label(
        frame_simular,
        text="A simulação apresentará os últimos 5 episódios do modelo selecionado.",
        font=("Segoe UI", 8),
        foreground="#475569",
    ).pack(anchor=tk.W, pady=(0, 6))

    if not nomes_modelos:
        tk.Label(
            frame_simular,
            text="Nenhum modelo treinado encontrado na pasta 'modelos'.",
            font=("Segoe UI", 8),
            bg=BG_CARD,
            fg="#dc2626",
        ).pack(anchor=tk.W, pady=(0, 6))

    # --- OPÇÕES DE DASHBOARD (Dinâmico, sem lista de recursos) ---
    frame_dashboard = ttk.LabelFrame(root, text="  Dashboard  ", style="Card.TLabelframe", padding=14)

    ttk.Label(
        frame_dashboard,
        text="Abre a janela do Dashboard para analisar o desempenho,\nrotas de cada episódio e configurações do modelo.",
        font=("Segoe UI", 9),
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=8)

    # --- BOTÃO DE AÇÃO ---
    btn_iniciar = ttk.Button(root, text="Iniciar", command=iniciar, style="Action.TButton")
    btn_iniciar.pack(side=tk.BOTTOM, pady=16, fill=tk.X, padx=14)

    # Inicializa a interface corretamente
    atualizar_interface()

    root.mainloop()



        # --- LÓGICA DE EXECUÇÃO APÓS A INTERFACE FECHAR ---



        # Se a janela for fechada sem clicar em Iniciar, o script termina

    if escolhas["acao"] is None:

        print("Execução cancelada.")

        sys.exit()

    if escolhas["acao"] == "treinar":
        nome_treino = (escolhas.get("nome_guardar") or escolhas.get("modelo_alvo") or "").lower()
        if "linear" in nome_treino:
            CONFIG["meander_model"] = "linear"
            print(f"\n[+] Treino Linear detetado. Ambiente configurado com meander_model = 'linear'.")
        elif "sinusoid" in nome_treino:
            CONFIG["meander_model"] = "sum_of_sinusoids"
            print(f"\n[+] Treino Sinusoidal detetado. Ambiente configurado com meander_model = 'sum_of_sinusoids'.")

        env = PlumeRLWrapper(CONFIG)
        agent = RLQLearningAgent(env)



        if escolhas["tipo_treino"] == "continuar" and escolhas["modelo_alvo"]:

            caminho_modelo = os.path.join(pasta_modelos, escolhas["modelo_alvo"])

            if agent.load(caminho_modelo):

                print(f"\n[!] A continuar o treino a partir do episódio {len(agent.episode_rewards)}")

                print(f"[!] Epsilon atual (taxa de exploração): {agent.epsilon:.4f}")

            else:

                print("[-] Erro ao carregar. A iniciar um novo treino do zero...")



        episodios = escolhas["episodios"]

        agent.train(env, num_episodes=episodios, print_every=50)

        caminho_salvo = agent.save(filename=escolhas.get("nome_guardar"))

        if CONFIG.get("visualize", True):
            nome_ficheiro = os.path.basename(caminho_salvo)

            CONFIG["agent_class"] = "QLearningAgent"
            CONFIG["model_to_load"] = caminho_salvo
            CONFIG["trained_q_agent"] = agent
            CONFIG["num_experiments"] = 5

            print(f"\n[!] Treino concluído com sucesso!")
            print(f"\n[!] A iniciar o simulador para mostrar os últimos 5 episódios do treino atual ({nome_ficheiro})...\n")

            sim = Simulator(CONFIG)
            sim.run()



    elif escolhas["acao"] == "simular":
        ficheiro_escolhido = os.path.join(pasta_modelos, escolhas["modelo_alvo"])

        CONFIG["agent_class"] = "QLearningAgent"
        CONFIG["model_to_load"] = ficheiro_escolhido
        CONFIG["trained_q_agent"] = None
        CONFIG["num_experiments"] = 5

        nome_alvo_low = escolhas["modelo_alvo"].lower()
        if "sinusoid" in nome_alvo_low:
            CONFIG["meander_model"] = "sum_of_sinusoids"
            print(f"\n[+] Modelo Sinusoidal detetado ({escolhas['modelo_alvo']}). Configurando simulador com meander_model = 'sum_of_sinusoids'.")
        else:
            CONFIG["meander_model"] = "linear"
            print(f"\n[+] Modelo Linear detetado ({escolhas['modelo_alvo']}). Configurando simulador com meander_model = 'linear'.")

        print(f"\nIniciando Simulador com: QLearningAgent usando {ficheiro_escolhido}")

        sim = Simulator(CONFIG)
        sim.run()
