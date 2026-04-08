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
from Agents import PPOAgent as RLPPOAgent
from Agents import QLearningAgent as RLQLearningAgent
from Agents import SARSAAgent as RLSARSAAgent


# ======================================================================================
# 1. HIGH-PERFORMANCE HELPER FUNCTIONS (NUMBA)
# ======================================================================================

@jit(nopython=True)
def update_puffs_numba(puffs_array, dt, wind_velocity, meandering_vy, turbulence_intensity, puff_decay_rate):
    """
    Updates the state of all puffs. This function is JIT-compiled by Numba for performance.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)

    for i in range(num_puffs):
        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 1] += wind_velocity[1] * dt
        puffs_array[i, 1] += meandering_vy * dt

        turbulence_x = np.random.randn() * turbulence_intensity * sqrt_dt
        turbulence_y = np.random.randn() * turbulence_intensity * sqrt_dt
        puffs_array[i, 0] += turbulence_x
        puffs_array[i, 1] += turbulence_y

        puffs_array[i, 2] = 0.1 * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= (1.0 - puff_decay_rate * dt)
        puffs_array[i, 4] += dt

    return puffs_array


@jit(nopython=True)
def calculate_concentration_numba(agent_pos, agent_radius, puffs_array):
    """Calculates concentration using a Top-Hat model."""
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]

    for i in range(puffs_array.shape[0]):
        puff_x, puff_y, puff_r, puff_mass = puffs_array[i, 0], puffs_array[i, 1], puffs_array[i, 2], puffs_array[i, 3]
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        radii_sum_sq = (agent_radius + puff_r) ** 2
        if dist_sq < radii_sum_sq:
            puff_concentration = puff_mass / (np.pi * puff_r ** 2) if puff_r > 1e-6 else 0.0
            total_concentration += puff_concentration
    return total_concentration


@jit(nopython=True)
def calculate_concentration_gaussian_numba(agent_pos, puffs_array):
    """Calculates concentration using a Gaussian profile for each puff."""
    total_concentration = 0.0
    agent_x, agent_y = agent_pos[0], agent_pos[1]
    for i in range(puffs_array.shape[0]):
        puff_x, puff_y, puff_r, puff_mass = puffs_array[i, 0], puffs_array[i, 1], puffs_array[i, 2], puffs_array[i, 3]
        sigma = puff_r / 2.0
        if sigma < 1e-6 or puff_mass < 1e-6: continue
        dist_sq = (agent_x - puff_x) ** 2 + (agent_y - puff_y) ** 2
        if dist_sq > (puff_r * 3) ** 2: continue
        sigma_sq = sigma ** 2
        concentration_k = (puff_mass / (2 * np.pi * sigma_sq)) * np.exp(-dist_sq / (2 * sigma_sq))
        total_concentration += concentration_k
    return total_concentration


# ======================================================================================
# MEANDER MODEL DEFINITIONS
# ======================================================================================

class MeanderModel:
    """Base class for meander models."""

    def __init__(self, main_config, model_config):
        self.main_config = main_config
        self.model_config = model_config

    def update(self, dt, current_time):
        raise NotImplementedError


class OU_MeanderModel(MeanderModel):
    """Ornstein-Uhlenbeck process for meandering."""

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)
        self.meandering_vy = 0.0
        timescale = self.model_config.get('timescale', 1.0)
        intensity = self.model_config.get('intensity', 0.2)
        self.ou_theta = 1.0 / timescale if timescale > 0 else 0
        self.ou_sigma = intensity

    def update(self, dt, current_time):
        if self.ou_theta > 0:
            drift = -self.ou_theta * self.meandering_vy * dt
            diffusion = self.ou_sigma * np.sqrt(dt) * np.random.randn()
            self.meandering_vy += drift + diffusion
        return self.meandering_vy


class Sinusoid_MeanderModel(MeanderModel):
    """
    A physically-grounded Sum of Sinusoids model that includes a
    selectable (optional) mean-reverting force to prevent long-term plume drift.
    """

    def __init__(self, main_config, model_config):
        super().__init__(main_config, model_config)

        self.num_harmonics = self.model_config.get('num_harmonics', 10)
        self.v_var = self.model_config.get('v_var', 0.25)
        self.L = self.model_config.get('integral_length_scale_L', 20.0)
        self.meandering_vy = 0.0

        drift_timescale = self.model_config.get('drift_correction_timescale', 45.0)

        if drift_timescale > 0:
            self.drift_theta = 1.0 / drift_timescale
            print(f"  > Drift correction ENABLED (timescale: {drift_timescale}s)")
        else:
            self.drift_theta = 0.0
            print("  > Drift correction DISABLED.")

        wind_speed = np.linalg.norm(self.main_config['mean_wind_velocity'])
        if wind_speed < 1e-6:
            self.amplitudes = np.zeros(self.num_harmonics)
            self.freqs = np.ones(self.num_harmonics)
            self.phases = np.zeros(self.num_harmonics)
            return

        low_freq = wind_speed / self.L
        high_freq = 100 * low_freq
        self.freqs = np.logspace(np.log10(low_freq), np.log10(high_freq), self.num_harmonics)

        wavenumbers = 2 * np.pi * self.freqs / wind_speed
        energy_spectrum = (self.v_var * (2 * self.L / np.pi)) / (1 + (wavenumbers * self.L) ** 2) ** (5 / 6)

        freq_spectrum = energy_spectrum * (2 * np.pi / wind_speed)
        df = np.gradient(self.freqs)
        self.amplitudes = np.sqrt(2 * freq_spectrum * df)
        self.phases = np.random.rand(self.num_harmonics) * 2 * np.pi

    def update(self, dt, current_time):
        # Calculate the instantaneous target velocity from the sum of sines
        target_vy = np.sum(self.amplitudes * np.sin(2 * np.pi * self.freqs * current_time + self.phases))

        # --- CORRECTED LOGIC ---
        # Check if drift correction is enabled
        if self.drift_theta > 0:
            # If ENABLED, use the stateful, integrating model that pulls the velocity
            # back to the mean, preventing long-term drift.
            drift_correction_force = -self.drift_theta * self.meandering_vy
            self.meandering_vy += (drift_correction_force + target_vy) * dt
            return self.meandering_vy
        else:
            # If DISABLED, return the original, stateless calculation.
            # This will behave identically to the version before drift correction was added.
            return target_vy


# ======================================================================================
# 3. CORE CLASS DEFINITIONS
# ======================================================================================

class DataLogger:
    """Handles saving simulation data to an HDF5 file for a single experiment."""

    def __init__(self, config, experiment_num):
        self.save_path = config['save_path']
        self.experiment_name = config['experiment_name']
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.save_path, f"{self.experiment_name}_exp{experiment_num}_{timestamp}.h5")
        self.data_buffer = {'time': [], 'agent_position': [], 'concentration': [], 'custom_osl_data': []}
        self.config = config

    def log_step(self, sim_time, agent_pos, concentration, custom_osl_data=None):
        self.data_buffer['time'].append(sim_time)
        self.data_buffer['agent_position'].append(agent_pos)
        self.data_buffer['concentration'].append(concentration)
        if custom_osl_data: self.data_buffer['custom_osl_data'].append(json.dumps(custom_osl_data))

    def save(self):
        print(f"\nSaving data for experiment to {self.filename}...")
        with h5py.File(self.filename, 'w') as f:
            for key, data in self.data_buffer.items():
                if not data: continue
                if key == 'custom_osl_data':
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset('osl/custom_data', data=data, dtype=dt)
                elif key == 'agent_position':
                    f.create_dataset('agent/positions', data=np.array(data))
                elif key == 'concentration':
                    f.create_dataset('agent/concentrations', data=np.array(data))
                else:
                    f.create_dataset(key, data=np.array(data))

            for key, value in self.config.items():
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = str(value)
        print("Save complete.")


class FilamentPlume:
    """Manages the physics of the odor environment using a selected meander model."""

    def __init__(self, config):
        self.config = config
        self.source_pos = np.array(config['source_position'])
        self.puffs_array = np.empty((0, 5), dtype=np.float32)
        self.time_since_last_puff = 0.0

        model_name = self.config.get('meander_model', 'ou')
        if model_name == 'sum_of_sinusoids':
            print("Using Sum of Sinusoids meander model.")
            model_config = self.config.get('sinusoid_meander_config', {})
            self.meander_generator = Sinusoid_MeanderModel(self.config, model_config)
        else:
            print("Using Ornstein-Uhlenbeck (OU) meander model.")
            model_config = self.config.get('ou_meander_config', {})
            self.meander_generator = OU_MeanderModel(self.config, model_config)

    def update(self, dt, current_time):
        meandering_vy = self.meander_generator.update(dt, current_time)
        self.time_since_last_puff += dt
        release_interval = 1.0 / self.config['emission_rate']
        puffs_to_release = int(self.time_since_last_puff / release_interval)
        if puffs_to_release > 0:
            new_puffs = np.zeros((puffs_to_release, 5), dtype=np.float32)
            new_puffs[:, 0:2] = self.source_pos
            new_puffs[:, 3] = self.config['initial_puff_mass']
            self.puffs_array = np.vstack([self.puffs_array, new_puffs])
            self.time_since_last_puff = 0.0

        if self.puffs_array.shape[0] > 0:
            self.puffs_array = update_puffs_numba(
                self.puffs_array, dt, np.array(self.config['mean_wind_velocity']),
                meandering_vy, self.config['turbulence_intensity'], self.config['puff_decay_rate']
            )
            min_mass = self.config['initial_puff_mass'] * 0.01
            alive_mask = (self.puffs_array[:, 3] > min_mass) & (self.puffs_array[:, 4] < self.config['max_time'])
            self.puffs_array = self.puffs_array[alive_mask]


class Agent:
    """Base class for a searching agent."""

    def __init__(self, config):
        self.config = config
        self.radius = self.config['agent_radius']
        self.speed = self.config['agent_speed']
        self.measured_flow_direction = np.array([1.0, 0.0])
        self.reset()

    def reset(self):
        center = np.array(self.config['agent_start_region_center'])
        radius = self.config['agent_start_region_radius']
        r = radius * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        self.pos = center + np.array([r * np.cos(theta), r * np.sin(theta)])
        self.direction = np.array([1.0, 0.0])
        self.current_concentration = 0.0
        self.measured_flow_direction = np.array([1.0, 0.0])
        if hasattr(self, 'state'): self.state = "casting"

    def measure_flow(self, plume):
        mean_wind = np.array(self.config['mean_wind_velocity'])
        meander_vec = np.array([0.0, plume.meander_generator.meandering_vy if isinstance(plume.meander_generator,
                                                                                         OU_MeanderModel) else plume.meander_generator.update(
            0, 0)])
        self.measured_flow_direction = mean_wind + meander_vec
        return self.measured_flow_direction

    def measure_concentration(self, plume):
        model = self.config.get('concentration_model', 'top-hat')
        if model == 'gaussian':
            self.current_concentration = calculate_concentration_gaussian_numba(self.pos, plume.puffs_array)
        else:
            self.current_concentration = calculate_concentration_numba(self.pos, self.radius, plume.puffs_array)
        return self.current_concentration

    def move(self, target_pos, dt):
        if target_pos is None: return
        world_w, world_h = self.config['world_width'], self.config['world_height']
        valid_target_pos = np.array([np.clip(target_pos[0], 0, world_w), np.clip(target_pos[1], 0, world_h)])
        direction_vec = valid_target_pos - self.pos
        distance = np.linalg.norm(direction_vec)
        if distance > 1e-6:
            self.direction = direction_vec / distance
            displacement = self.direction * self.speed * dt
            self.pos = valid_target_pos if np.linalg.norm(displacement) > distance else self.pos + displacement

    def check_success(self, source_pos):
        return np.linalg.norm(self.pos - source_pos) < self.config.get('success_distance', 1.0)

    def run_search_algorithm(self, dt):
        raise NotImplementedError("This method should be implemented by a subclass.")


class SimpleUpwindAgent(Agent):
    """A simple agent for basic testing."""

    def __init__(self, config):
        self.state, self.cast_direction, self.last_hit_pos = None, None, None
        super().__init__(config)

    def reset(self):
        super().reset()
        self.state, self.cast_direction, self.last_hit_pos = "casting", 1, None

    def run_search_algorithm(self, dt):
        concentration_threshold = self.config.get('agent_concentration_threshold', 1.0)
        surge_len = self.config.get('surge_length', 5.0)
        cast_len = self.config.get('cast_length', 8.0)

        custom_data = {'state': self.state}
        if self.current_concentration > concentration_threshold:
            self.state = "surging"
            self.last_hit_pos = self.pos.copy()
            upwind_vec = -np.array(self.config['mean_wind_velocity'])
            target_pos = self.pos + upwind_vec * surge_len
        else:
            if self.state == "surging":
                self.state, self.cast_direction = "casting", self.cast_direction * -1
            crosswind_vec = np.array([-self.config['mean_wind_velocity'][1], self.config['mean_wind_velocity'][0]])
            target_pos = self.pos + (crosswind_vec * self.cast_direction * cast_len)
        return target_pos, custom_data


class BioInspiredAgent(Agent):
    """A more robust agent using decision hysteresis and an expanding cast search pattern."""

    def __init__(self, config):
        self.main_config = config
        super().__init__(config)

    def reset(self):
        super().reset()
        self.phase, self.behavior, self.state = 'SEARCHING', 'SEARCHING', 'DECIDING'
        self.target_pos, self.cast_direction, self.search_direction = None, 1, 1
        self.consecutive_hits, self.consecutive_misses = 0, 0
        self.current_cast_length = self.config.get('cast_length', 7.0)

    def _compute_zig_zag_target(self, length, angle_deg, direction_sign):
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm < 1e-6: return self.pos
        upwind_dir = -self.measured_flow_direction / flow_norm
        angle_rad = np.deg2rad(angle_deg * direction_sign)
        rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
        return self.pos + (rot_matrix @ upwind_dir) * length

    def _set_search_target(self):
        print(f"BEHAVIOR: Searching (direction: {self.search_direction})")
        self.behavior = 'SEARCHING'
        potential_target = self._compute_zig_zag_target(self.config['search_length'], self.config['search_angle'],
                                                        self.search_direction)
        if not (0 < potential_target[1] < self.main_config['world_height']):
            print("  > Search target out of bounds. Inverting direction.")
            self.search_direction *= -1
            self.target_pos = self._compute_zig_zag_target(self.config['search_length'], self.config['search_angle'],
                                                           self.search_direction)
        else:
            self.target_pos = potential_target
        self.state = 'MOVING_TO_TARGET'

    def _set_cast_target(self):
        print(f"BEHAVIOR: Casting (direction: {self.cast_direction}, length: {self.current_cast_length:.2f})")
        self.behavior = 'CASTING'
        potential_target = self._compute_zig_zag_target(self.current_cast_length, self.config['cast_angle'],
                                                        self.cast_direction)
        if not (0 < potential_target[1] < self.main_config['world_height']):
            print("  > Cast target out of bounds. Inverting direction.")
            self.cast_direction *= -1
            self.target_pos = self._compute_zig_zag_target(self.current_cast_length, self.config['cast_angle'],
                                                           self.cast_direction)
        else:
            self.target_pos = potential_target
        self.state = 'MOVING_TO_TARGET'

    def _set_surge_target(self):
        print("BEHAVIOR: Surging")
        self.behavior = 'SURGING'
        self.current_cast_length = self.config.get('cast_length', 7.0)
        flow_norm = np.linalg.norm(self.measured_flow_direction)
        if flow_norm > 1e-6:
            upwind_dir = -self.measured_flow_direction / flow_norm
            self.target_pos = self.pos + upwind_dir * self.config['surge_length']
        self.state = 'MOVING_TO_TARGET'

    def run_search_algorithm(self, dt):
        is_hit = self.current_concentration > self.config['agent_concentration_threshold']
        if is_hit:
            self.consecutive_hits, self.consecutive_misses = self.consecutive_hits + 1, 0
        else:
            self.consecutive_misses, self.consecutive_hits = self.consecutive_misses + 1, 0
        confirmation_steps = self.config.get('confirmation_steps', 1)

        if self.phase == 'SEARCHING' and self.consecutive_hits >= confirmation_steps:
            print(f"PLUME FOUND ({self.consecutive_hits} consecutive hits)! Switching to TRACKING phase.")
            self.phase, self.state = 'TRACKING', 'DECIDING'
        elif self.phase == 'TRACKING' and self.behavior == 'SURGING' and self.consecutive_misses >= confirmation_steps:
            print(f"PLUME LOST ({self.consecutive_misses} consecutive misses)! Switching to CAST.")
            self.state = 'DECIDING'
        elif self.phase == 'TRACKING' and self.behavior == 'CASTING' and self.consecutive_hits >= confirmation_steps:
            print(f"PLUME RE-ACQUIRED ({self.consecutive_hits} consecutive hits)! Switching to SURGE.")
            self.state = 'DECIDING'

        if self.state == 'DECIDING':
            if self.phase == 'SEARCHING':
                self._set_search_target()
            elif self.phase == 'TRACKING':
                if self.consecutive_hits > 0:
                    self._set_surge_target()
                else:
                    if self.behavior == 'CASTING':
                        self.current_cast_length *= (1.0 + self.config.get('cast_length_increase_factor', 0.1))
                        print(f"  > Cast failed. Increasing cast length to {self.current_cast_length:.2f}")
                        self.cast_direction *= -1
                    self._set_cast_target()
        elif self.state == 'MOVING_TO_TARGET':
            if self.target_pos is None or np.linalg.norm(self.pos - self.target_pos) < 0.5:
                self.state, self.target_pos = 'DECIDING', None

        custom_data = {'phase': self.phase, 'behavior': self.behavior, 'hits': self.consecutive_hits,
                       'misses': self.consecutive_misses}
        return self.target_pos, custom_data





class PlumeRLWrapper(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.maze_grid = np.zeros((1, 1))
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.global_visit_map = np.zeros((1, 1))
        self.path_history = []
        self.agent_pos = (0, 0)

    def print_maze_info(self):
        print(f"Ambiente Contínuo PPO: {self.config['world_width']}x{self.config['world_height']}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plume = FilamentPlume(self.config)
        self.time = 0.0

        # Spawn do agente
        c = np.array(self.config['agent_start_region_center'])
        r = self.config['agent_start_region_radius'] * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        self.pos = c + np.array([r * np.cos(theta), r * np.sin(theta)])
        self.agent_pos = self.pos.copy()  # Compatibilidade
        self.path_history = [self.pos.copy()]

        # Estabilizar a pluma
        for _ in range(int(self.config['plume_stabilization_time'] / self.config['dt'])):
            self.plume.update(self.config['dt'], self.time)
            self.time += self.config['dt']

        return self._get_obs(), {"at_goal": False}

    def step(self, action):
        dt = self.config['dt']

        prev_dist = np.linalg.norm(self.pos - np.array(self.config['source_position']))

        # Ação -> Vetor de Direção
        dirs = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
        target = self.pos + dirs[action] * self.config['agent_config']['agent_speed'] * dt

        # Aplicar limites do mundo
        self.pos[0] = np.clip(target[0], 0, self.config['world_width'])
        self.pos[1] = np.clip(target[1], 0, self.config['world_height'])
        self.agent_pos = self.pos.copy()
        self.path_history.append(self.pos.copy())

        # Atualizar Física
        self.plume.update(dt, self.time)
        self.time += dt

        conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)
        dist_to_source = np.linalg.norm(self.pos - np.array(self.config['source_position']))
        at_goal = dist_to_source < self.config['agent_config']['success_distance']

        terminated = at_goal
        truncated = self.time >= self.config['max_time']

        # Recompensa
        if at_goal:
            reward = 100.0
        else:
            reward = -0.05  # Penalidade de tempo
            if conc > self.config['agent_config']['agent_concentration_threshold']:
                reward += 1.0  # Bónus de tracking da pluma

            world_diag = np.sqrt(self.config['world_width']**2 + self.config['world_height']**2)

            reward += 0.5 * (prev_dist - dist_to_source) / world_diag * 100
        return self._get_obs(), reward, terminated, truncated, {"at_goal": at_goal}

    def _get_obs(self):
        conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)

        vy = self.plume.meander_generator.meandering_vy
        flow = np.array(self.config['mean_wind_velocity']) + np.array([0.0, vy])

        return np.array([
            self.pos[0] / self.config['world_width'],
            self.pos[1] / self.config['world_height'],
            conc, flow[0], flow[1]
        ], dtype=np.float32)

class QLearningSimAgent(Agent):
    """O Agente que utiliza o modelo Q-Learning para tomar decisões no Simulador principal."""

    def __init__(self, config):
        super().__init__(config)
        dummy_env = PlumeRLWrapper(config)
        # Atenção ao typo no teu import original: RLQLearingAgent
        self.q_agent = RLQLearningAgent(dummy_env)

        try:
            self.q_agent.load("qlearning_plume.pkl") # Usando .pkl ou a extensão que usas nos teus agentes
            print("\n[+] Modelo 'qlearning_plume.pkl' carregado com sucesso!")
        except:
            print("\n[-] AVISO: 'qlearning_plume.pkl' não encontrado. O agente vai mover-se aleatoriamente. Treina-o primeiro!")

    def run_search_algorithm(self, dt):
        obs = np.array([
            self.pos[0] / self.config['world_width'],
            self.pos[1] / self.config['world_height'],
            self.current_concentration,
            self.measured_flow_direction[0],
            self.measured_flow_direction[1]
        ], dtype=np.float32)

        action = self.q_agent.choose_action(obs, greedy=True)

        dirs = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
        target_pos = self.pos + dirs[action] * self.speed * dt

        return target_pos, {"action": action}


class SARSASimAgent(Agent):
    """O Agente que utiliza o modelo SARSA para tomar decisões no Simulador principal."""

    def __init__(self, config):
        super().__init__(config)
        dummy_env = PlumeRLWrapper(config)
        self.sarsa_agent = RLSARSAAgent(dummy_env)

        try:
            self.sarsa_agent.load("sarsa_plume.pkl")
            print("\n[+] Modelo 'sarsa_plume.pkl' carregado com sucesso!")
        except:
            print("\n[-] AVISO: 'sarsa_plume.pkl' não encontrado. O agente vai mover-se aleatoriamente. Treina-o primeiro!")

    def run_search_algorithm(self, dt):
        obs = np.array([
            self.pos[0] / self.config['world_width'],
            self.pos[1] / self.config['world_height'],
            self.current_concentration,
            self.measured_flow_direction[0],
            self.measured_flow_direction[1]
        ], dtype=np.float32)

        action = self.sarsa_agent.choose_action(obs, greedy=True)

        dirs = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
        target_pos = self.pos + dirs[action] * self.speed * dt

        return target_pos, {"action": action}

class PPOSimAgent(Agent):
    """O Agente que utiliza o modelo PPO para tomar decisões no Simulador principal."""

    def __init__(self, config):
        super().__init__(config)
        dummy_env = PlumeRLWrapper(config)
        self.ppo = RLPPOAgent(dummy_env)

        try:
            self.ppo.load("ppo_plume.pth")
            print("\n[+] Modelo 'ppo_plume.pth' carregado com sucesso!")
        except:
            print(
                "\n[-] AVISO: 'ppo_plume.pth' não encontrado. O agente vai mover-se aleatoriamente. Treina-o primeiro!")

    def run_search_algorithm(self, dt):
        obs = np.array([
            self.pos[0] / self.config['world_width'],
            self.pos[1] / self.config['world_height'],
            self.current_concentration,
            self.measured_flow_direction[0],
            self.measured_flow_direction[1]
        ], dtype=np.float32)

        # Escolher a ação gananciosa (sem exploração aleatória porque estamos a testar)
        action = self.ppo.choose_action(obs, greedy=True)

        dirs = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
        target_pos = self.pos + dirs[action] * self.speed * dt

        return target_pos, {"action": action}

AGENT_REGISTRY = {
    'SimpleUpwindAgent': SimpleUpwindAgent,
    'BioInspiredAgent': BioInspiredAgent,
    'PPOAgent': PPOSimAgent,
    'QLearningAgent': QLearningSimAgent,
    'SARSAAgent': SARSASimAgent,
}
# ======================================================================================
# 4. SIMULATOR (ORCHESTRATOR)
# ======================================================================================

class Simulator:
    """Main orchestrator for running experiments."""

    def __init__(self, config):
        self.config = config
        agent_class_name = config['agent_class']
        if agent_class_name not in AGENT_REGISTRY: raise ValueError(f"Agent class '{agent_class_name}' not found.")
        agent_full_config = {**config, **config.get('agent_config', {})}
        self.agent = AGENT_REGISTRY[agent_class_name](agent_full_config)
        self.plume, self.logger = None, None
        self.simulation_time, self.current_experiment = 0.0, 0
        self.loop_time_ema, self.ema_alpha = 0.0, 0.1
        if config['visualize']: self._init_visualization()

    def reset_experiment(self):
        print("-" * 40 + f"\nStarting Experiment {self.current_experiment + 1}/{self.config['num_experiments']}")
        self.logger = DataLogger(self.config, self.current_experiment)
        self.simulation_time = 0.0
        self.plume = FilamentPlume(self.config)
        self.agent.reset()

    def step(self):
        dt = self.config['dt']
        self.plume.update(dt, self.simulation_time)
        self.agent.measure_concentration(self.plume)
        self.agent.measure_flow(self.plume)

        target_pos, custom_data = self.agent.run_search_algorithm(dt)
        self.agent.move(target_pos, dt)
        self.logger.log_step(self.simulation_time, self.agent.pos.copy(), self.agent.current_concentration, custom_data)

        self.simulation_time += dt
        if self.agent.check_success(self.plume.source_pos):
            print(f"\nSource found at time {self.simulation_time:.2f}s!")
            return True
        return False

    def run(self):
        num_experiments = self.config.get('num_experiments', 1)
        stabilization_time = self.config.get('plume_stabilization_time', 0.0)
        for i in range(num_experiments):
            self.current_experiment = i
            self.reset_experiment()
            print(f"Stabilizing plume for {stabilization_time} seconds...")
            self.simulation_time = 0.0
            while self.simulation_time < stabilization_time:
                self.plume.update(self.config['dt'], self.simulation_time)
                self.simulation_time += self.config['dt']
                if self.config['visualize']: self._draw()

            print("Stabilization complete. Starting agent search...")
            self.simulation_time = 0.0
            step_counter = 0
            while self.simulation_time < self.config['max_time']:
                loop_start_time = time.time()
                if self.step(): break
                if self.config['visualize']: self._draw()
                elapsed_time = time.time() - loop_start_time
                if self.loop_time_ema == 0.0:
                    self.loop_time_ema = elapsed_time
                else:
                    self.loop_time_ema = (self.ema_alpha * elapsed_time) + ((1 - self.ema_alpha) * self.loop_time_ema)
                step_counter += 1
                if step_counter % 20 == 0:
                    hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
                    print(f"  > Sim Time: {self.simulation_time:.1f}s, Running Freq: {hz:.1f} Hz", end='\r')
                if self.config['visualize'] and self.config['real_time_pacing']:
                    wait_time = self.config['dt'] - elapsed_time
                    if wait_time > 0: time.sleep(wait_time)
            print()
            if self.logger: self.logger.save()
        print("-" * 40 + "\nAll experiments finished.")
        if self.config['visualize']: cv2.destroyAllWindows()

    def _init_visualization(self):
        self.viz_w, self.viz_h = self.config['viz_width'], self.config['viz_height']
        self.world_w, self.world_h = self.config['world_width'], self.config['world_height']
        cv2.namedWindow('Odour Simulation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Odour Simulation', self.viz_w, self.viz_h)

    def _world_to_pixel(self, pos):
        px = int(pos[0] / self.world_w * self.viz_w)
        py = int(self.viz_h - (pos[1] / self.world_h * self.viz_h))
        return px, py

    def _draw_axes(self, canvas):
        axis_color, font, font_scale, font_thickness = (200, 200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        cv2.line(canvas, self._world_to_pixel((0, 0)), self._world_to_pixel((self.world_w, 0)), axis_color, 2)
        cv2.line(canvas, self._world_to_pixel((0, 0)), self._world_to_pixel((0, self.world_h)), axis_color, 2)
        for i in range(6):
            x_w = (i / 5) * self.world_w
            x_p, y_p = self._world_to_pixel((x_w, 0))
            cv2.line(canvas, (x_p, y_p - 5), (x_p, y_p + 5), axis_color, 1)
            t_s = cv2.getTextSize(str(int(x_w)), font, font_scale, font_thickness)[0]
            cv2.putText(canvas, str(int(x_w)), (x_p - t_s[0] // 2, y_p + 25), font, font_scale, axis_color,
                        font_thickness)
        for i in range(6):
            y_w = (i / 5) * self.world_h
            x_p, y_p = self._world_to_pixel((0, y_w))
            cv2.line(canvas, (x_p - 5, y_p), (x_p + 5, y_p), axis_color, 1)
            t_s = cv2.getTextSize(str(int(y_w)), font, font_scale, font_thickness)[0]
            cv2.putText(canvas, str(int(y_w)), (x_p - t_s[0] - 15, y_p + t_s[1] // 2), font, font_scale, axis_color,
                        font_thickness)
        cv2.putText(canvas, "(0,0)", (self._world_to_pixel((0, 0))[0] + 10, self._world_to_pixel((0, 0))[1] - 10), font,
                    font_scale, axis_color, font_thickness)
        max_x_t = f"({int(self.world_w)}, 0)"
        t_s = cv2.getTextSize(max_x_t, font, font_scale, font_thickness)[0]
        cv2.putText(canvas, max_x_t, (self._world_to_pixel((self.world_w, 0))[0] - t_s[0] - 10,
                                      self._world_to_pixel((self.world_w, 0))[1] - 10), font, font_scale, axis_color,
                    font_thickness)
        max_y_t = f"(0, {int(self.world_h)})"
        t_s = cv2.getTextSize(max_y_t, font, font_scale, font_thickness)[0]
        cv2.putText(canvas, max_y_t,
                    (self._world_to_pixel((0, self.world_h))[0] + 10, self._world_to_pixel((0, self.world_h))[1] + 20),
                    font, font_scale, axis_color, font_thickness)
        start_c = self._world_to_pixel(self.config['agent_start_region_center'])
        start_r = int(self.config['agent_start_region_radius'] / self.world_w * self.viz_w)
        cv2.circle(canvas, start_c, start_r, (0, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _draw(self):
        # 1. Create a blank canvas
        canvas = np.zeros((self.viz_h, self.viz_w, 3), dtype=np.uint8)

        # Draw axes onto the canvas first
        canvas = self._draw_axes(canvas)

        # 2. Draw puffs
        if self.plume.puffs_array.shape[0] > 0:
            for puff in self.plume.puffs_array:
                pos_px, radius_px = self._world_to_pixel(puff[0:2]), max(1, int(puff[2] / self.world_w * self.viz_w))
                color_val = min(255, int(puff[3] / self.config['initial_puff_mass'] * 255))
                cv2.circle(canvas, pos_px, radius_px, (color_val, color_val, color_val), -1, cv2.LINE_AA)

        # 3. Draw source
        source_px = self._world_to_pixel(self.config['source_position'])
        cv2.drawMarker(canvas, source_px, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

        # 4. Draw agent
        conc_thresh = self.config.get('agent_config', {}).get('agent_concentration_threshold', 5.0)
        agent_color = (0, 255, 0) if self.agent.current_concentration > conc_thresh else (255, 0, 0)
        agent_px = self._world_to_pixel(self.agent.pos)
        agent_radius_px = max(2, int(self.agent.radius / self.world_w * self.viz_w))
        cv2.circle(canvas, agent_px, agent_radius_px, agent_color, 2, cv2.LINE_AA)
        arrow_end_px = self._world_to_pixel(self.agent.pos + self.agent.direction * self.agent.radius * 2.5)
        cv2.arrowedLine(canvas, agent_px, arrow_end_px, (255, 255, 0), 2, cv2.LINE_AA)

        # 5. Draw updated text info
        font_scale = 0.6
        hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
        flow_vec = self.agent.measured_flow_direction
        info_left = f"Time: {self.simulation_time:.1f}s | Freq: {hz:.1f}Hz | Conc: {self.agent.current_concentration:.2f} | Flow: ({flow_vec[0]:.1f}, {flow_vec[1]:.1f})"
        info_right = f"Agent Pos: ({self.agent.pos[0]:.1f}, {self.agent.pos[1]:.1f})"

        # This line is correct
        cv2.putText(canvas, info_left, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        t_s_r = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        # --- CORRECTED LINE ---
        # The color tuple (255, 255, 255) and thickness integer (1) are now in the correct order.
        cv2.putText(canvas, info_right, (self.viz_w - t_s_r[0] - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1)

        # 6. Display
        cv2.imshow('Odour Simulation', canvas)
        if cv2.waitKey(1) == ord('q'): self.simulation_time = self.config['max_time']


# ======================================================================================
# 5. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == '__main__':
    CONFIG = {
        'dt': 0.1, 'max_time': 500.0,
        'save_path': 'experiments', 'experiment_name': 'bio_inspired_test',
        'visualize': True, 'real_time_pacing': False,
        'viz_width': 1200, 'viz_height': 600,
        'world_width': 100.0, 'world_height': 50.0,
        'source_position': (10.0, 25.0),

        'emission_rate': 10,  # <-- THIS LINE WAS MISSING
        'initial_puff_mass': 100.0,
        'puff_decay_rate': 0.05,

        'concentration_model': 'gaussian',

        'num_experiments': 10,
        'plume_stabilization_time': 1.0,

        'agent_start_region_center': (90.0, 10.0),
        'agent_start_region_radius': 2.0,

        # --- MEANDER MODEL CONFIGURATION ---
        'meander_model': 'sum_of_sinusoids',
        'ou_meander_config': {
            'intensity': 0.2,
            'timescale': 2.0,
        },
        'sinusoid_meander_config': {
            'num_harmonics': 10,
            'v_var': 0.8,  #
            'integral_length_scale_L': 20.0,
            'drift_correction_timescale': 10
        },
        # 'sinusoid_meander_config': {
        # 'num_harmonics': 10,
        # 'v_var': 0.4,  # <-- Strong, but not excessive
        # 'integral_length_scale_L': 25.0,
        # 'drift_correction_timescale': 45.0
        # },

        # --- TURBULENCE INTENSITY ---
        'turbulence_intensity': 0.1,
        # 'turbulence_intensity': 0.7,

        # --- MEAN WIND VELOCITY ---
        'mean_wind_velocity': (1.0, 0.0),

        'agent_class': 'BioInspiredAgent',
        'agent_config': {
            'agent_radius': 0.5, 'agent_speed': 3.0,
            'success_distance': 2.0,
            'agent_concentration_threshold': 1.0,
            'surge_length': 4.0, 'cast_length': 6.0,
            'cast_angle': 60.0,
            'search_length': 8.0, 'search_angle': 75.0,
            'cast_length_increase_factor': 0.1,
            'confirmation_steps': 3,
        }
    }

    print("=" * 40)
    print(" OPÇÕES DE TREINO:")
    print(" 1. Treinar Novo Modelo PPO")
    print(" 2. Treinar Novo Modelo Q-Learning")
    print(" 3. Treinar Novo Modelo SARSA")
    print("-" * 40)
    print(" OPÇÕES DE SIMULAÇÃO (TESTE):")
    print(" 4. Executar Simulador (com Agente PPO)")
    print(" 5. Executar Simulador (com Agente Q-Learning)")
    print(" 6. Executar Simulador (com Agente SARSA)")
    print(" 7. Executar Simulador (com Agente BioInspired)")
    print(" 8. Executar Simulador (com Agente SimpleUpwind)")
    print("=" * 40)

    escolha = input("Escolha (1-8): ").strip()

    if escolha in ["1", "2", "3"]:
        # TREINO
        env = PlumeRLWrapper(CONFIG)
        episodios = int(input("Número de episódios para treinar (ex: 500): ") or "500")

        if escolha == "1":
            agent = RLPPOAgent(env, learning_rate=3e-4)
            agent.train(num_episodes=episodios, print_every=10)
            agent.save("ppo_plume.pth")
        elif escolha == "2":
            agent = RLQLearningAgent(env)
            agent.train(num_episodes=episodios, print_every=10)
            agent.save("qlearning_plume.pkl")
        elif escolha == "3":
            agent = RLSARSAAgent(env)
            agent.train(num_episodes=episodios, print_every=10)
            agent.save("sarsa_plume.pkl")

    else:
        # TESTE
        mapa_escolhas = {
            "4": "PPOAgent",
            "5": "QLearningAgent",
            "6": "SARSAAgent",
            "7": "BioInspiredAgent",
            "8": "SimpleUpwindAgent"
        }

        agente_escolhido = mapa_escolhas.get(escolha, "BioInspiredAgent")  # Default fallback
        print(f"\nIniciando Simulador com: {agente_escolhido}")
        CONFIG['agent_class'] = agente_escolhido
        sim = Simulator(CONFIG)
        sim.run()