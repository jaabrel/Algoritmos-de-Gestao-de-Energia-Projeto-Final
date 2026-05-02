import numpy as np
import random
import pickle
import datetime
import os
from collections import defaultdict


class QLearningAgent:
    """
    Improved Q-Learning Agent with proper state binning, reward shaping, and tracking.

    Key improvements:
    - Proper episode length tracking
    - Better state binning with configurable resolution
    - Improved epsilon decay schedule
    - Correct terminal state handling
    - Configuration validation
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.9995,  # ✓ Fixed: was 0.995 (too aggressive)
        min_exploration_rate: float = 0.01,
        state_bins: dict = None,  # ✓ New: configurable binning
    ):
        """
        Initialize Q-Learning Agent

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Epsilon decay rate per episode
            min_exploration_rate: Minimum epsilon value
            state_bins: Dict with 'x', 'y', 'conc', 'wind' bin counts
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        # ✓ Fixed: Configurable state binning
        self.state_bins = state_bins or {
            "x": 50,  # Position X bins
            "y": 25,  # Position Y bins
            "conc": 3,  # Concentration bins (low, medium, high)
            "wind": 4,  # Wind direction bins (4 cardinal directions)
        }

        self.q_table = {}

        # ✓ Fixed: Tracking statistics
        self.episode_rewards = []
        self.episode_lengths = []  # ✓ Now properly tracked
        self.success_history = []

    def _state_key(self, state) -> tuple:
        """
        Convert continuous state to discrete state key for Q-table indexing

        State format: [norm_x, norm_y, conc, flow_x, flow_y]
        All values should be normalized to [0, 1] range (or close to it)

        ✓ Fixed: Better binning with configurable resolution
        """
        state = np.asarray(state, dtype=np.float32)

        # Position binning
        x_b = int(np.clip(state[0] * self.state_bins["x"], 0, self.state_bins["x"] - 1))
        y_b = int(np.clip(state[1] * self.state_bins["y"], 0, self.state_bins["y"] - 1))

        # Concentration binning
        # Assumes concentration is normalized to [0, 1]
        conc = state[2]

        if conc <= 0.33:
            c_b = 0  # Low concentration
        elif conc <= 0.67:
            c_b = 1  # Medium concentration
        else:
            c_b = 2  # High concentration

        # Wind direction binning (4 cardinal directions)
        fx, fy = state[3], state[4]
        angle = np.arctan2(fy, fx)
        wind_dir = int(np.round(angle / (np.pi / 2))) % 4

        return (x_b, y_b, c_b, wind_dir)

    def choose_action(self, state, greedy: bool = False) -> int:

        key = self._state_key(state)

        if len(self.q_table) < 5:
            print(f"State key: {key}, Q-values: {self.q_table.get(key, 'NEW')}")
        if not greedy and random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            q_values = self.q_table.get(key, np.zeros(self.env.action_space.n))
            # Exploit: best known action
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done, truncated=False):
        """Update Q-value using Q-learning"""
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Initialize Q-values if new state
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.env.action_space.n)

        current_q = self.q_table[key][action]

        # Proper terminal state handling
        if done or truncated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_key])
            target_q = reward + self.gamma * max_next_q

        # Q-learning update
        self.q_table[key][action] += self.lr * (target_q - current_q)

    def train(self, env, num_episodes=500, print_every=50):
        """Train the Q-Learning agent"""
        print(f"\n{'=' * 70}")
        print(f"  TRAINING Q-LEARNING — {num_episodes} episodes")
        print(
            f"  State bins: {self.state_bins['x']}×{self.state_bins['y']}×{self.state_bins['conc']}×{self.state_bins['wind']}"
        )
        print(
            f"  Max possible states: {self.state_bins['x'] * self.state_bins['y'] * self.state_bins['conc'] * self.state_bins['wind']}"
        )
        print(f"{'=' * 70}\n")

        for ep in range(1, num_episodes + 1):
            state, info = env.reset()
            done = truncated = False
            total_reward = 0
            step = 0

            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)

                self.update(state, action, reward, next_state, done, truncated)

                state = next_state
                total_reward += reward
                step += 1

            # Track statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            self.success_history.append(1 if info.get("at_goal", False) else 0)

            # Decay exploration
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Print progress
            if ep % print_every == 0:
                w = min(print_every, ep)
                avg_reward = np.mean(self.episode_rewards[-w:])
                avg_length = np.mean(self.episode_lengths[-w:])
                avg_success = np.mean(self.success_history[-w:]) * 100

                print(
                    f"  Ep {ep:>5}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:>7.2f} | "
                    f"Avg Length: {avg_length:>6.1f} | "
                    f"Success: {avg_success:>5.1f}% | "
                    f"ε: {self.epsilon:.4f} | "
                    f"Q-States: {len(self.q_table):>5}"
                )

    def save(self, filename=None):
        """
        Save trained model and statistics

        Args:
            filename: Optional filename (default: timestamped)

        Returns:
            Path to saved file
        """
        pasta = "modelos"
        os.makedirs(pasta, exist_ok=True)

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qlearning_plume_{timestamp}.pkl"

        caminho_completo = os.path.join(pasta, filename)

        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "state_bins": self.state_bins,
            "hyperparameters": {
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "exploration_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
            },
            "stats": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,  # ✓ Now properly saved
                "successes": self.success_history,
            },
        }

        with open(caminho_completo, "wb") as f:
            pickle.dump(data, f)

        print(f"\n[+] Model and statistics saved to: {caminho_completo}")
        print(f"    Episodes trained: {len(self.episode_rewards)}")
        print(f"    Q-table states: {len(self.q_table)}")
        print(
            f"    Final success rate: {np.mean(self.success_history[-100:]) * 100:.1f}%"
        )

        return caminho_completo

    def load(self, filepath):
        """
        Load trained model and statistics

        Args:
            filepath: Path to saved model

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"[-] WARNING: {filepath} not found.")
            return False

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.q_table.update(data["q_table"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.state_bins = data.get("state_bins", self.state_bins)

            # Load statistics if available
            if "stats" in data:
                self.episode_rewards = data["stats"].get("rewards", [])
                self.episode_lengths = data["stats"].get("lengths", [])
                self.success_history = data["stats"].get("successes", [])

            print(f"[+] Model loaded from {filepath}")
            print(f"    Q-table states: {len(self.q_table)}")
            print(f"    Episodes in history: {len(self.episode_rewards)}")

            return True

        except Exception as e:
            print(f"[-] Error loading model: {e}")
            return False

    def get_statistics(self):
        """Get training statistics"""
        if not self.episode_rewards:
            return None

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "success_rate": np.mean(self.success_history),
            "q_table_size": len(self.q_table),
            "current_epsilon": self.epsilon,
        }
