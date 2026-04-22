import numpy as np
import random
import pickle 
import os
from collections import defaultdict  

class QLearningAgent:
    def __init__(
            self,
            env,
            learning_rate: float = 0.1,
            discount_factor: float = 0.99,
            exploration_rate: float = 1.0,
            exploration_decay: float = 0.995,
            min_exploration_rate: float = 0.01,
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []

    def _state_key(self, state) -> tuple:
        # state = [norm_x, norm_y, conc, flow_x, flow_y]
        state = np.asarray(state, dtype=np.float32)
        
        # Bins for relative position (we don't use absolute source pos to generalize)
        # But since the user wants fixed source, we can still use normalized coords
        x_b = int(np.clip(state[0] * 20, 0, 19))
        y_b = int(np.clip(state[1] * 10, 0, 9))

        # Concentration bins
        threshold = self.env.config.get('agent_config', {}).get('agent_concentration_threshold', 1.0)
        conc = state[2]
        if conc <= 0.0:
            c_b = 0
        elif conc <= threshold:
            c_b = 1
        else:
            c_b = 2

        # Wind direction bin
        fx, fy = state[3], state[4]
        angle = np.arctan2(fy, fx)
        wind_dir = int(np.round(angle / (np.pi / 2))) % 4

        return (x_b, y_b, c_b, wind_dir)

    def choose_action(self, state, greedy: bool = False) -> int:
        key = self._state_key(state)
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done, truncated=False):
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        current_q = self.q_table[key][action]
        max_next_q = np.max(self.q_table[next_key])

        # Bootstrap from next state unless the episode ended at the TRUE goal
        target_q = reward if done else reward + self.gamma * max_next_q
        self.q_table[key][action] += self.lr * (target_q - current_q)

    def train(self, num_episodes=500, print_every=50):
        print(f"\n{'=' * 55}\n  TRAINING Q-LEARNING — {num_episodes} episodes\n{'=' * 55}")
        
        for ep in range(1, num_episodes + 1):
            state, info = self.env.reset()
            done = truncated = False
            total_reward = step = 0

            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                self.update(state, action, reward, next_state, done, truncated)
                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)
            self.success_history.append(1 if info["at_goal"] else 0)

            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if ep % print_every == 0:
                w = min(print_every, ep)
                avg_reward = np.mean(self.episode_rewards[-w:])
                avg_success = np.mean(self.success_history[-w:]) * 100
                print(f"  Ep {ep:>5}/{num_episodes} | Avg Reward: {avg_reward:>+8.2f} | Success: {avg_success:>5.1f}% | Epsilon: {self.epsilon:.4f}")

    def save(self, filename="q_agent.pkl"):
        data = {"q_table": dict(self.q_table), "epsilon": self.epsilon}
        with open(filename, "wb") as f: pickle.dump(data, f)
        print(f"Model saved to {filename}")

    def load(self, filename="q_agent.pkl"):
        if not os.path.exists(filename): 
            print(f"Warning: {filename} not found.")
            return False
        with open(filename, "rb") as f: data = pickle.load(f)
        self.q_table.update(data["q_table"])
        self.epsilon = data["epsilon"]
        print(f"Model loaded from {filename}")
        return True
