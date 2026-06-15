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
# 1. HIGH-PERFORMANCE HELPER FUNCTIONS (NUMBA)
# ======================================================================================


@jit(nopython=True)
def update_puffs_numba(
    puffs_array, dt, wind_velocity, meandering_vy, turbulence_intensity, puff_decay_rate, min_mass
):
    """
    Updates the state of all puffs. This function is JIT-compiled by Numba for performance.
    """
    num_puffs = puffs_array.shape[0]
    sqrt_dt = np.sqrt(dt)

    for i in range(num_puffs):
        if puffs_array[i, 3] <= min_mass:
            continue

        puffs_array[i, 0] += wind_velocity[0] * dt
        puffs_array[i, 1] += wind_velocity[1] * dt
        puffs_array[i, 1] += meandering_vy * dt

        # Controlled turbulence: moderate horizontal, tight vertical for excellent shape definition
        turbulence_x = np.random.randn() * (turbulence_intensity * 1.0) * sqrt_dt
        turbulence_y = np.random.randn() * (turbulence_intensity * 0.3) * sqrt_dt
        puffs_array[i, 0] += turbulence_x
        puffs_array[i, 1] += turbulence_y

        puffs_array[i, 2] = 0.1 * np.sqrt(puffs_array[i, 4])
        puffs_array[i, 3] *= 1.0 - puff_decay_rate * dt
        puffs_array[i, 4] += dt

    return puffs_array


@jit(nopython=True)
def calculate_concentration_numba(agent_pos, agent_radius, puffs_array):
    """Calculates concentration using a Top-Hat model."""
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
    """Calculates concentration using a Gaussian profile for each puff."""
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
        timescale = self.model_config.get("timescale", 1.0)
        intensity = self.model_config.get("intensity", 0.2)
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

        self.num_harmonics = self.model_config.get("num_harmonics", 10)
        self.v_var = self.model_config.get("v_var", 0.25)
        self.L = self.model_config.get("integral_length_scale_L", 20.0)
        self.meandering_vy = 0.0

        drift_timescale = self.model_config.get("drift_correction_timescale", 45.0)

        if drift_timescale > 0:
            self.drift_theta = 1.0 / drift_timescale
            print(f"  > Drift correction ENABLED (timescale: {drift_timescale}s)")
        else:
            self.drift_theta = 0.0
            print("  > Drift correction DISABLED.")

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
        # Perfect, clean single sinusoid for a beautiful wave-like plume
        amplitude = self.model_config.get("amplitude", 1.5)
        frequency = self.model_config.get("frequency", 0.08)
        self.meandering_vy = amplitude * np.sin(2 * np.pi * frequency * current_time)
        return self.meandering_vy


# ======================================================================================
# 3. CORE CLASS DEFINITIONS
# ======================================================================================


class DataLogger:
    """Handles saving simulation data to an HDF5 file for a single experiment."""

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
            "custom_osl_data": [],
        }
        self.config = config

    def log_step(self, sim_time, agent_pos, concentration, agent_state, custom_osl_data=None):
        self.data_buffer["time"].append(sim_time)
        self.data_buffer["agent_position"].append(agent_pos)
        self.data_buffer["concentration"].append(concentration)
        self.data_buffer["agent_state"].append(agent_state)
        if custom_osl_data:
            self.data_buffer["custom_osl_data"].append(json.dumps(custom_osl_data))

    def save(self):
        print(f"\nSaving data for experiment to {self.filename}...")
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
        self.source_pos = np.array(config["source_position"])
        
        # Static pre-allocated array initialization
        # A estrutura das 5 colunas do puffs_array é:
        # Coluna 0: Posição X (x)
        # Coluna 1: Posição Y (y)
        # Coluna 2: Raio do Puff (r)
        # Coluna 3: Massa do Puff (mass) [0.0 se estiver inativo]
        # Coluna 4: Idade do Puff (age) em segundos
        self.max_puffs = self.config.get("max_puffs", 2000)
        self.puffs_array = np.zeros((self.max_puffs, 5), dtype=np.float32)
        self.time_since_last_puff = 0.0

        model_name = self.config.get("meander_model", "ou")
        if model_name == "sum_of_sinusoids":
            print("Using Sum of Sinusoids meander model.")
            model_config = self.config.get("sinusoid_meander_config", {})
            self.meander_generator = Sinusoid_MeanderModel(self.config, model_config)
        else:
            print("Using Ornstein-Uhlenbeck (OU) meander model.")
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
                # For the sum_of_sinusoids model, we set the initial vertical offset at release.
                # This creates a perfect spatial sinusoidal plume wave trailing downwind.
                if self.config.get("meander_model", "ou") == "sum_of_sinusoids":
                    self.puffs_array[slots_to_use, 1] = self.source_pos[1] + meandering_vy
                else:
                    self.puffs_array[slots_to_use, 1] = self.source_pos[1]
                
                self.puffs_array[slots_to_use, 2] = 0.0
                self.puffs_array[slots_to_use, 3] = self.config["initial_puff_mass"]
                self.puffs_array[slots_to_use, 4] = 0.0
                
            self.time_since_last_puff = 0.0

        if self.puffs_array.shape[0] > 0:
            # For the sum_of_sinusoids model, we pass 0.0 to update_puffs_numba so that
            # existing puffs do not experience the global vertical meander velocity.
            # This preserves the sinusoidal shape in space.
            meander_val = 0.0 if self.config.get("meander_model", "ou") == "sum_of_sinusoids" else meandering_vy
            self.puffs_array = update_puffs_numba(
                self.puffs_array,
                dt,
                np.array(self.config["mean_wind_velocity"]),
                meander_val,
                self.config["turbulence_intensity"],
                self.config["puff_decay_rate"],
                min_mass,
            )
            # Atualizar inatividade no próprio array (sem realocação de memória)
            dead_mask = (self.puffs_array[:, 3] <= min_mass) | (
                self.puffs_array[:, 4] >= self.config["max_time"]
            )
            self.puffs_array[dead_mask, 3] = 0.0


class Agent:
    """Base class for a searching agent."""

    def __init__(self, config):
        self.config = config
        self.radius = self.config["agent_radius"]
        self.speed = self.config["agent_speed"]
        self.measured_flow_direction = np.array([1.0, 0.0])
        self.reset()

    def reset(self):
        center = np.array(self.config["agent_start_region_center"])
        radius = self.config["agent_start_region_radius"]
        r = radius * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        self.pos = center + np.array([r * np.cos(theta), r * np.sin(theta)])
        self.direction = np.array([1.0, 0.0])
        self.current_concentration = 0.0
        self.measured_flow_direction = np.array([1.0, 0.0])
        if hasattr(self, "internal_state"):
            self.internal_state = "casting"

        self.state = "searching"
        self.first_detection_pos = None
        self.mission_complete = False
        self.circling_step = 0
        self.prev_dist_to_detection = None
        self.skirting_side_ida = None

    def get_circling_target(self):
        source_pos = np.array(self.config["source_position"])
        radius = self.config.get("agent_config", {}).get("success_distance", 2.0) * 1.5
        total_steps = 16
        angle = (self.circling_step / total_steps) * 2 * np.pi
        target = source_pos + np.array([radius * np.cos(angle), radius * np.sin(angle)])
        self.circling_step += 1
        if self.circling_step >= total_steps:
            self.state = "returning"
            print(f"\n[Agent] Circling complete! State changed to RETURNING. Goal: {self.first_detection_pos}")
        return target

    def measure_flow(self, plume):
        mean_wind = np.array(self.config["mean_wind_velocity"])

        if isinstance(plume.meander_generator, OU_MeanderModel):
            vy = plume.meander_generator.meandering_vy
        else:
            vy = plume.meander_generator.meandering_vy

        self.measured_flow_direction = mean_wind + np.array([0.0, vy])
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
    """A more robust agent using decision hysteresis and an expanding cast search pattern."""

    def __init__(self, config):
        self.main_config = config
        super().__init__(config)

    def reset(self):
        super().reset()
        self.state, self.behavior, self.internal_state = "SEARCHING", "SEARCHING", "DECIDING"
        self.target_pos, self.cast_direction, self.search_direction = None, 1, 1
        self.consecutive_hits, self.consecutive_misses = 0, 0
        self.current_cast_length = self.config.get("cast_length", 7.0)

    def _compute_zig_zag_target(self, length, angle_deg, direction_sign):
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
        if not (0 < potential_target[1] < self.main_config["world_height"]):
            print("  > Search target out of bounds. Inverting direction.")
            self.search_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.config["search_length"],
                self.config["search_angle"],
                self.search_direction,
            )
        else:
            self.target_pos = potential_target
        self.internal_state = "MOVING_TO_TARGET"

    def _set_cast_target(self):
        print(
            f"BEHAVIOR: Casting (direction: {self.cast_direction}, length: {self.current_cast_length:.2f})"
        )
        self.behavior = "CASTING"
        potential_target = self._compute_zig_zag_target(
            self.current_cast_length, self.config["cast_angle"], self.cast_direction
        )
        if not (0 < potential_target[1] < self.main_config["world_height"]):
            print("  > Cast target out of bounds. Inverting direction.")
            self.cast_direction *= -1
            self.target_pos = self._compute_zig_zag_target(
                self.current_cast_length, self.config["cast_angle"], self.cast_direction
            )
        else:
            self.target_pos = potential_target
        self.internal_state = "MOVING_TO_TARGET"

    def _set_surge_target(self):
        print("BEHAVIOR: Surging")
        self.behavior = "SURGING"
        self.current_cast_length = self.config.get("cast_length", 7.0)
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
            if (
                self.target_pos is None
                or np.linalg.norm(self.pos - self.target_pos) < 0.5
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

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config):
        """
        Initialize environment

        Args:
            config: Configuration dictionary with simulator parameters
        """
        super().__init__()
        self.config = config

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )
        self.reward_config = {
            "step_penalty": -0.05,
            "concentration_improvement": 1.0,
            "concentration_decrease": -0.05,
            "revisit_threshold": 1.0,  #
            "revisit_penalty": -0.2,  # Mild penalty for revisiting
            "plume_detection_bonus": 0.3,
            "upwind_bonus": 1.0,  # Bonus for moving upwind
            "goal_reward": 500.0,  # Reward for reaching goal
        }

        self.conc_max = 10.0  # Expected maximum concentration
        self.flow_max = 5.0  # Expected maximum flow magnitude

        # State tracking
        self.plume = None
        self.pos = None
        self.time = 0.0
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_pos = None
        self.prev_conc = 0.0
        self.current_conc = 0.0

    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config["world_width"] > 0, "world_width must be positive"
        assert self.config["world_height"] > 0, "world_height must be positive"
        assert self.config["dt"] > 0 and self.config["dt"] < 1, "dt must be in (0, 1)"
        assert self.config["agent_config"]["agent_speed"] > 0, (
            "agent_speed must be positive"
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

        min_margin = 5.0

        self.config["agent_start_region_center"] = (
            np.random.uniform(
                self.config["world_width"] * 0.6,
                self.config["world_width"] - min_margin,
            ),
            np.random.uniform(min_margin, self.config["world_height"] - min_margin),
        )

        self.plume = FilamentPlume(self.config)
        self.time = 0.0

        # Initialize agent position
        c = np.array(self.config["agent_start_region_center"])
        r = self.config["agent_start_region_radius"] * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        self.pos = c + np.array([r * np.cos(theta), r * np.sin(theta)])
        self.prev_pos = self.pos.copy()

        # Reset tracking
        self.visit_map = {}
        self.current_cell = None
        self.steps_in_current_cell = 0
        self.prev_conc = 0.0
        self.current_conc = 0.0

        # Mission state machine properties
        self.state = "searching"
        self.first_detection_pos = None
        self.mission_complete = False
        self.circling_step = 0
        self.prev_dist_to_detection = None

        # Stabilize plume
        for _ in range(
            int(self.config["plume_stabilization_time"] / self.config["dt"])
        ):
            self.plume.update(self.config["dt"], self.time)
            self.time += self.config["dt"]

        return self._get_obs(), {"at_goal": False}

    def step(self, action):
        """
        Execute one step in environment
        """
        dt = self.config["dt"]
        self.prev_pos = self.pos.copy()

        dirs = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0]),   # Right
        }

        # --- High-level state transitions in training env ---
        conc_thresh = self.config["agent_config"]["agent_concentration_threshold"]
        source_pos = np.array(self.config["source_position"])
        dist_to_source = np.linalg.norm(self.pos - source_pos)
        success_distance = self.config["agent_config"]["success_distance"]

        if self.state == "searching":
            self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)
            if self.current_conc > conc_thresh:
                self.state = "following_boundary"
                self.first_detection_pos = self.pos.copy()

        elif self.state == "following_boundary":
            if dist_to_source < success_distance:
                self.state = "circling_source"
                self.circling_step = 0

        # --- Circling source is handled programmatically ---
        if self.state == "circling_source":
            radius = success_distance * 1.5
            total_steps = 16
            angle = (self.circling_step / total_steps) * 2 * np.pi
            self.pos = source_pos + np.array([radius * np.cos(angle), radius * np.sin(angle)])
            self.circling_step += 1
            if self.circling_step >= total_steps:
                self.state = "returning"
                self.prev_dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
            
            # Move plume and increment time
            self.plume.update(dt, self.time)
            self.time += dt
            self.current_conc = calculate_concentration_gaussian_numba(self.pos, self.plume.puffs_array)
            obs = self._get_obs()
            reward = 2.0  # constant small reward for circling
            terminated = False
            truncated = self.time >= self.config["max_time"]
            return obs, reward, terminated, truncated, {"at_goal": False}

        # --- Normal step movement (for searching, following, returning) ---
        target = self.pos + dirs[action] * self.config["agent_config"]["agent_speed"] * dt
        self.pos[0] = np.clip(target[0], 0, self.config["world_width"])
        self.pos[1] = np.clip(target[1], 0, self.config["world_height"])

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

        # Check goal / termination
        if self.state == "returning":
            dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
            at_goal = dist_to_detection < success_distance
            terminated = at_goal
            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(self.current_conc, self.prev_conc, visitas_nesta_celula, at_goal)
        else:
            at_goal = dist_to_source < success_distance
            terminated = False  # Simulation does NOT terminate when reaching source
            truncated = self.time >= self.config["max_time"]
            reward = self._calculate_reward(self.current_conc, self.prev_conc, visitas_nesta_celula, False)

        return self._get_obs(), reward, terminated, truncated, {"at_goal": at_goal}

    def _calculate_reward(self, conc, prev_conc, visitas_nesta_celula, at_goal):
        if self.state == "returning":
            if at_goal:
                return 500.0  # Terminal reward for returning
            
            reward = -0.05  # Step penalty
            dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
            if not hasattr(self, 'prev_dist_to_detection') or self.prev_dist_to_detection is None:
                self.prev_dist_to_detection = dist_to_detection
            delta_dist = dist_to_detection - self.prev_dist_to_detection
            self.prev_dist_to_detection = dist_to_detection
            
            # Reward for moving closer to first_detection_pos
            reward += -delta_dist * 10.0
            
            # Revisit penalty
            if visitas_nesta_celula > self.reward_config["revisit_threshold"]:
                reward += self.reward_config["revisit_penalty"]
            return reward
            
        else:
            # --- Searching / Following Boundary rewards ---
            source_pos = np.array(self.config["source_position"])
            dist_to_source = np.linalg.norm(self.pos - source_pos)
            if dist_to_source < self.config["agent_config"]["success_distance"]:
                return 100.0  # reached source bonus!
                
            reward = self.reward_config["step_penalty"]

            # Boundary following reward:
            if self.state == "following_boundary":
                threshold = self.config["agent_config"]["agent_concentration_threshold"]
                deviation = abs(conc - threshold)
                boundary_reward = 2.0 * np.exp(-deviation)
                reward += boundary_reward
                
                # Wind upwind navigation progress
                mean_wind = np.array(self.config["mean_wind_velocity"])
                if np.linalg.norm(mean_wind) > 1e-6:
                    upwind_dir = -mean_wind / np.linalg.norm(mean_wind)
                    movement = self.pos - self.prev_pos
                    movement_norm = np.linalg.norm(movement)
                    if movement_norm > 1e-6:
                        upwind_progress = np.dot(movement / movement_norm, upwind_dir)
                        if upwind_progress > 0:
                            reward += self.reward_config["upwind_bonus"] * upwind_progress * 1.5
            else:
                # Searching: basic concentration improvement reward
                conc_diff = conc - prev_conc
                if conc_diff > 0:
                    reward += self.reward_config["concentration_improvement"] * min(conc_diff, 1.0)
                elif conc_diff < 0 and conc > 0:
                    reward += self.reward_config["concentration_decrease"]

            if visitas_nesta_celula > self.reward_config["revisit_threshold"]:
                reward += self.reward_config["revisit_penalty"]

            # Plume detection bonus
            threshold = self.config["agent_config"]["agent_concentration_threshold"]
            if conc > threshold:
                reward += self.reward_config["plume_detection_bonus"]

            return reward

    def _get_obs(self):
        conc_normalized = np.clip(self.current_conc / self.conc_max, 0, 1)

        vy = self.plume.meander_generator.meandering_vy
        flow = np.array(self.config["mean_wind_velocity"]) + np.array([0.0, vy])

        flow_x_normalized = np.clip(flow[0] / self.flow_max, -1, 1)
        flow_y_normalized = np.clip(flow[1] / self.flow_max, -1, 1)

        # Centered relative position in returning state, absolute position otherwise
        if self.state == "returning":
            obs_x = (self.pos[0] - self.first_detection_pos[0]) / self.config["world_width"] + 0.5
            obs_y = (self.pos[1] - self.first_detection_pos[1]) / self.config["world_height"] + 0.5
            obs_x = np.clip(obs_x, 0.0, 1.0)
            obs_y = np.clip(obs_y, 0.0, 1.0)
            conc_val = np.clip(np.linalg.norm(self.pos - self.first_detection_pos) / 50.0, 0.0, 1.0)
        else:
            obs_x = self.pos[0] / self.config["world_width"]
            obs_y = self.pos[1] / self.config["world_height"]
            conc_val = conc_normalized

        state_val = 1.0 if self.state == "returning" else 0.0

        return np.array(
            [
                obs_x,
                obs_y,
                conc_val,
                flow_x_normalized,
                flow_y_normalized,
                state_val,
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
        super().__init__(config)
        dummy_env = PlumeRLWrapper(config)

        self.q_agent = RLQLearningAgent(dummy_env)

        chosen_model = self.config.get("model_to_load", "qlearning_plume.pkl")

        try:
            self.q_agent.load(chosen_model)
            print(f"\n[+] Model '{chosen_model}' loaded")
        except:
            print(f"\n[-] Model '{chosen_model}' not found")

    def run_search_algorithm(self, dt):
        # Check if the loaded model is a 4-tuple model (which does not support RL returning)
        is_4tuple = False
        if hasattr(self, 'q_agent') and self.q_agent.q_table:
            first_key = next(iter(self.q_agent.q_table.keys()))
            if len(first_key) == 4:
                is_4tuple = True

        if self.state == "returning" and is_4tuple:
            dist_to_detection = np.linalg.norm(self.pos - self.first_detection_pos)
            if dist_to_detection < 10.0:
                # Close to return point: ignore skirting and go straight to it!
                goal_vec = self.first_detection_pos - self.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.pos + (goal_vec / goal_norm) * self.speed * dt if goal_norm > 1e-6 else self.pos
                return target_pos, {"action": "heuristic_return"}

            # Fallback to boundary-skirting return heuristic
            conc_thresh = self.config.get("agent_config", {}).get("agent_concentration_threshold", 1.0)
            if self.current_concentration > conc_thresh:
                # We are inside the plume: move crosswind to exit it
                center_y = self.config["world_height"] / 2.0
                side = 1.0 if self.pos[1] >= center_y else -1.0
                skirt_vec = np.array([0.0, side])
                target_pos = self.pos + skirt_vec * self.speed * dt
                return target_pos, {"action": "skirting_crosswind"}
            else:
                # Outside or on the boundary: move towards the return point
                goal_vec = self.first_detection_pos - self.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.pos + (goal_vec / goal_norm) * self.speed * dt if goal_norm > 1e-6 else self.pos
                return target_pos, {"action": "skirting_downwind"}

        # Build the exact same observation as PlumeRLWrapper
        if self.state == "returning":
            obs_x = (self.pos[0] - self.first_detection_pos[0]) / self.config["world_width"] + 0.5
            obs_y = (self.pos[1] - self.first_detection_pos[1]) / self.config["world_height"] + 0.5
            obs_x = np.clip(obs_x, 0.0, 1.0)
            obs_y = np.clip(obs_y, 0.0, 1.0)
            conc_val = np.clip(np.linalg.norm(self.pos - self.first_detection_pos) / 50.0, 0.0, 1.0)
        else:
            obs_x = self.pos[0] / self.config["world_width"]
            obs_y = self.pos[1] / self.config["world_height"]
            conc_val = np.clip(self.current_concentration / 10.0, 0, 1)

        vy = self.measured_flow_direction[1]
        flow_x_norm = np.clip(self.measured_flow_direction[0] / 5.0, -1, 1)
        flow_y_norm = np.clip(vy / 5.0, -1, 1)
        state_val = 1.0 if self.state == "returning" else 0.0

        obs = np.array(
            [
                obs_x,
                obs_y,
                conc_val,
                flow_x_norm,
                flow_y_norm,
                state_val
            ],
            dtype=np.float32,
        )

        action = self.q_agent.choose_action(obs, greedy=True)

        dirs = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        target_pos = self.pos + dirs[action] * self.speed * dt

        return target_pos, {"action": action}


AGENT_REGISTRY = {
    "QLearningAgent": QLearningSimAgent,
}
# ======================================================================================
# 4. SIMULATOR (ORCHESTRATOR)
# ======================================================================================


class Simulator:
    """Main orchestrator for running experiments."""

    def __init__(self, config):
        self.config = config
        agent_class_name = config["agent_class"]
        if agent_class_name not in AGENT_REGISTRY:
            raise ValueError(f"Agent class '{agent_class_name}' not found.")
        agent_full_config = {**config, **config.get("agent_config", {})}
        self.agent = AGENT_REGISTRY[agent_class_name](agent_full_config)
        self.plume, self.logger = None, None
        self.simulation_time, self.current_experiment = 0.0, 0
        self.loop_time_ema, self.ema_alpha = 0.0, 0.1
        if config["visualize"]:
            self._init_visualization()

    def reset_experiment(self):
        print(
            "-" * 40
            + f"\nStarting Experiment {self.current_experiment + 1}/{self.config['num_experiments']}"
        )

        min_margin = 5.0
        self.config["agent_start_region_center"] = (
            np.random.uniform(
                self.config["world_width"] * 0.6,
                self.config["world_width"] - min_margin,
            ),
            np.random.uniform(min_margin, self.config["world_height"] - min_margin),
        )

        self.logger = DataLogger(self.config, self.current_experiment)
        self.simulation_time = 0.0
        self.plume = FilamentPlume(self.config)
        self.agent.reset()

    def step(self):
        dt = self.config["dt"]
        self.plume.update(dt, self.simulation_time)
        self.agent.measure_concentration(self.plume)
        self.agent.measure_flow(self.plume)

        conc_thresh = self.config["agent_config"].get("agent_concentration_threshold", 1.0)
        source_pos = self.plume.source_pos
        dist_to_source = np.linalg.norm(self.agent.pos - source_pos)
        success_distance = self.config["agent_config"].get("success_distance", 2.0)

        if self.agent.state == "searching":
            if self.agent.current_concentration > conc_thresh:
                self.agent.state = "following_boundary"
                self.agent.first_detection_pos = self.agent.pos.copy()
                print(f"\n[Simulator] Plume DETECTED at {self.agent.first_detection_pos}! Following boundary to source.")

        elif self.agent.state == "following_boundary":
            if dist_to_source < success_distance:
                self.agent.state = "circling_source"
                self.agent.circling_step = 0
                print(f"\n[Simulator] Source REACHED! Starting circling/contouring maneuver.")

        elif self.agent.state == "circling_source":
            pass

        elif self.agent.state == "returning":
            dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
            if dist_to_detection < success_distance:
                self.agent.mission_complete = True
                print(f"\n[Simulator] Mission complete! Returned to detection point at {self.agent.pos}.")

        if self.agent.state == "circling_source":
            target_pos = self.agent.get_circling_target()
            custom_data = {"state": "circling_source", "step": self.agent.circling_step}
            self.agent.move(target_pos, dt)
        elif self.agent.state == "following_boundary":
            # Skirt the plume on the way to the source (Searching for Origin) for ALL agents!
            if self.agent.current_concentration > conc_thresh:
                # We are inside the plume: move crosswind to exit it
                center_y = self.config["world_height"] / 2.0
                side = 1.0 if self.agent.pos[1] >= center_y else -1.0
                
                # Record the side we used during the upwind trip!
                if self.agent.skirting_side_ida is None:
                    self.agent.skirting_side_ida = side
                
                skirt_vec = np.array([0.0, side])
                target_pos = self.agent.pos + skirt_vec * self.agent.speed * dt
                custom_data = {"state": "following_skirting_crosswind"}
            else:
                # Outside or on the boundary: move towards the source
                goal_vec = source_pos - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                custom_data = {"state": "following_skirting_upwind"}
            self.agent.move(target_pos, dt)
        elif self.agent.state == "returning":
            dist_to_detection = np.linalg.norm(self.agent.pos - self.agent.first_detection_pos)
            if dist_to_detection < 10.0:
                # Close to return point: ignore skirting and go straight to it!
                goal_vec = self.agent.first_detection_pos - self.agent.pos
                goal_norm = np.linalg.norm(goal_vec)
                target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                custom_data = {"state": "returning_heuristic"}
            else:
                # Plume skirting on return for ALL agents
                if self.agent.current_concentration > conc_thresh:
                    # We are inside the plume: move crosswind to exit it
                    if self.agent.skirting_side_ida is not None:
                        # Force the OPPOSITE side of the plume on return!
                        side = -self.agent.skirting_side_ida
                    else:
                        center_y = self.config["world_height"] / 2.0
                        side = 1.0 if self.agent.pos[1] >= center_y else -1.0
                        
                    skirt_vec = np.array([0.0, side])
                    target_pos = self.agent.pos + skirt_vec * self.agent.speed * dt
                    custom_data = {"state": "returning_skirting_crosswind"}
                else:
                    # Outside or on the boundary: move towards the detection point
                    goal_vec = self.agent.first_detection_pos - self.agent.pos
                    goal_norm = np.linalg.norm(goal_vec)
                    target_pos = self.agent.pos + (goal_vec / goal_norm) * self.agent.speed * dt if goal_norm > 1e-6 else self.agent.pos
                    custom_data = {"state": "returning_skirting_downwind"}
            self.agent.move(target_pos, dt)
        else:
            target_pos, custom_data = self.agent.run_search_algorithm(dt)
            self.agent.move(target_pos, dt)

        self.logger.log_step(
            self.simulation_time,
            self.agent.pos.copy(),
            self.agent.current_concentration,
            self.agent.state,
            custom_data,
        )

        self.simulation_time += dt
        if self.agent.mission_complete:
            print(f"\nSimulation complete! Successfully returned to detection point in {self.simulation_time:.2f}s!")
            return True
        return False

    def run(self):
        num_experiments = self.config.get("num_experiments", 1)
        stabilization_time = self.config.get("plume_stabilization_time", 0.0)
        for i in range(num_experiments):
            self.current_experiment = i
            self.reset_experiment()
            print(f"Stabilizing plume for {stabilization_time} seconds...")
            self.simulation_time = 0.0
            while self.simulation_time < stabilization_time:
                self.plume.update(self.config["dt"], self.simulation_time)
                self.simulation_time += self.config["dt"]
                if self.config["visualize"]:
                    self._draw()

            print("Stabilization complete. Starting agent search...")
            self.simulation_time = 0.0
            step_counter = 0
            while self.simulation_time < self.config["max_time"]:
                loop_start_time = time.time()
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
                self._world_to_pixel((0, self.world_h))[1] + 20,
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
        # 1. Create a blank canvas
        canvas = np.zeros((self.viz_h, self.viz_w, 3), dtype=np.uint8)

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

        if self.agent.first_detection_pos is not None:
            detection_px = self._world_to_pixel(self.agent.first_detection_pos)
            cv2.circle(canvas, detection_px, 6, (0, 255, 255), -1, cv2.LINE_AA)  # Yellow solid dot
            cv2.putText(canvas, 'detection', (detection_px[0] + 10, detection_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

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
        font_scale = 0.6
        hz = 1.0 / self.loop_time_ema if self.loop_time_ema > 0 else 0.0
        flow_vec = self.agent.measured_flow_direction
        state_str = self.agent.state.upper()
        info_left = f"Time: {self.simulation_time:.1f}s | State: {state_str} | Conc: {self.agent.current_concentration:.2f} | Flow: ({flow_vec[0]:.1f}, {flow_vec[1]:.1f})"
        info_right = f"Agent Pos: ({self.agent.pos[0]:.1f}, {self.agent.pos[1]:.1f})"

        # This line is correct
        cv2.putText(
            canvas,
            info_left,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )

        t_s_r = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        # The color tuple (255, 255, 255) and thickness integer (1) are now in the correct order.
        cv2.putText(
            canvas,
            info_right,
            (self.viz_w - t_s_r[0] - 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )

        # 6. Display
        cv2.imshow("Odour Simulation", canvas)
        if cv2.waitKey(1) == ord("q"):
            self.simulation_time = self.config["max_time"]


# ======================================================================================
# 5. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    import glob

    CONFIG = {
        "dt": 0.1,
        "max_time": 200.0,
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
        "puff_decay_rate": 0.05,
        "concentration_model": "gaussian",
        "num_experiments": 10,
        "plume_stabilization_time": 1.0,
        "agent_start_region_center": (60.0, 25.0),
        "agent_start_region_radius": 2.0,
        # --- MEANDER MODEL CONFIGURATION ---
        "meander_model": "sum_of_sinusoids",
        "ou_meander_config": {
            "intensity": 0.08,  # Well-controlled meander intensity
            "timescale": 0.1,
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
        "turbulence_intensity": 0.25,  # Moderated base intensity for tight, highly structured plume
        # 'turbulence_intensity': 0.7,
        # --- MEAN WIND VELOCITY ---
        "mean_wind_velocity": (0.2, 0),
        "agent_class": "QLearningAgent",
        "agent_config": {
            "agent_radius": 0.5,
            "agent_speed": 3.0,
            "success_distance": 2.0,
            "agent_concentration_threshold": 1.0,
            "surge_length": 4.0,
            "cast_length": 6.0,
            "cast_angle": 60.0,
            "search_length": 8.0,
            "search_angle": 75.0,
            "cast_length_increase_factor": 0.1,
            "confirmation_steps": 3,
        },
    }

    print("=" * 40)
    print(" OPÇÕES DE TREINO:")
    print(" 1. Treinar Modelo Q-Learning")
    print(" 2. Executar Simulador (com Agente Q-Learning)")
    print("=" * 40)

    escolha = input("Escolha (1-2): ").strip()
    if escolha == "1":
        # TREINO
        env = PlumeRLWrapper(CONFIG)
        agent = RLQLearningAgent(env)

        print("\n--- OPÇÕES DE TREINO ---")
        print(" 1. Iniciar Novo Treino (Do zero)")
        print(" 2. Continuar Treino de um Modelo Existente")
        sub_escolha = input("Escolha (1-2): ").strip()

        if sub_escolha == "2":
            pasta_modelos = "modelos"
            modelos = glob.glob(os.path.join(pasta_modelos, "qlearning_plume_*.pkl"))
            if not modelos:
                print("Nenhum modelo encontrado. A iniciar um novo treino do zero...")
            else:
                modelos.sort(key=os.path.getmtime, reverse=True)
                for i, m in enumerate(modelos):
                    print(f" {i + 1}. {os.path.basename(m)}")

                idx = input(
                    f"\nQual modelo queres continuar a treinar? (1-{len(modelos)}): "
                )
                try:
                    ficheiro_escolhido = modelos[int(idx) - 1]
                    if agent.load(ficheiro_escolhido):
                        print(
                            f"\n[!] A continuar o treino a partir do episódio {len(agent.episode_rewards)}"
                        )
                        print(
                            f"[!] Epsilon atual (taxa de exploração): {agent.epsilon:.4f}"
                        )
                except (ValueError, IndexError):
                    print("Escolha inválida. A iniciar um novo treino do zero...")

        episodios = int(
            input("\nNúmero de episódios adicionais para treinar (ex:3000): ") or "3000"
        )

        agent.train(env, num_episodes=episodios, print_every=50)
        agent.save()

        if CONFIG["visualize"]:
            cv2.destroyAllWindows()

    elif escolha == "2":
        pasta_modelos = "modelos"
        modelos = glob.glob(os.path.join(pasta_modelos, "qlearning_plume_*.pkl"))
        if not modelos:
            print(
                f"Nenhum modelo treinado encontrado na pasta '{pasta_modelos}'. Treina um primeiro (Opção 1)."
            )
        else:
            print("\nModelos disponíveis:")
            modelos.sort(key=os.path.getmtime, reverse=True)

            for i, m in enumerate(modelos):
                nome_ficheiro = os.path.basename(m)
                print(f" {i + 1}. {nome_ficheiro}")

            idx = input(f"\nQual modelo queres testar? (1-{len(modelos)}): ")
            try:
                ficheiro_escolhido = modelos[int(idx) - 1]

                CONFIG["agent_class"] = "QLearningAgent"
                CONFIG["model_to_load"] = ficheiro_escolhido

                print(
                    f"\nIniciando Simulador com: QLearningAgent usando {ficheiro_escolhido}"
                )
                sim = Simulator(CONFIG)
                sim.run()
            except (ValueError, IndexError):
                print("Escolha inválida.")

    else:
        print("Escolha inválida")
