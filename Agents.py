import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import pickle
import os
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from typing import Optional

# ===========
# CLASSE BASE
# ===========

class BaseAgent(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def choose_action(self, state, greedy=False):
        pass

    @abstractmethod
    def train(self, num_episodes, render_every=None, print_every=100):
        pass

    @abstractmethod
    def test(self, num_episodes, render=True, verbose=True):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass


# =========
# Q-LEARNING
# =========
class QLearningAgent(BaseAgent):
    def __init__(
            self,
            env,
            learning_rate: float = 0.15,
            discount_factor: float = 0.95,
            exploration_rate: float = 1.0,
            exploration_decay: float = 0.997,
            min_exploration_rate: float = 0.01,
    ):
        super().__init__(env)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_paths = []
        self.success_history = []

    def _state_key(self, pos) -> tuple:
        if hasattr(pos, "__iter__"):
            return tuple(int(x) for x in pos)
        return pos

    def choose_action(self, agent_pos, greedy: bool = False) -> int:
        key = self._state_key(agent_pos)
        if not greedy and random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[key]))

    def update(self, pos, action, reward, next_pos, done):
        key = self._state_key(pos)
        next_key = self._state_key(next_pos)
        current_q = self.q_table[key][action]
        target_q = reward if done else reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.lr * (target_q - current_q)

    def train(self, num_episodes=2000, render_every=None, print_every=100):
        print(f"\n{'=' * 55}\n  TRAINING Q-LEARNING — {num_episodes} episodes\n{'=' * 55}")
        self.env.print_maze_info()

        for ep in range(1, num_episodes + 1):
            if render_every and ep % render_every == 0:
                self.env.render_mode = "human"
            else:
                self.env.render_mode = None

            _, info = self.env.reset()
            agent_pos = self.env.agent_pos.copy()
            done = truncated = False
            total_reward = step = 0

            while not (done or truncated):
                action = self.choose_action(agent_pos)
                _, reward, done, truncated, info = self.env.step(action)
                next_pos = self.env.agent_pos
                self.update(agent_pos, action, reward, next_pos, done or truncated)
                agent_pos = next_pos
                total_reward += reward
                step += 1

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            self.episode_paths.append(list(self.env.path_history))
            self.success_history.append(1 if info["at_goal"] else 0)

            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if ep % print_every == 0:
                w = min(print_every, ep)
                print(
                    f"  Ep {ep:>5}/{num_episodes} │ Reward: {np.mean(self.episode_rewards[-w:]):>+7.2f} │ Success: {np.mean(self.success_history[-w:]) * 100:>5.1f}% │ ε: {self.epsilon:.4f}")

    def test(self, num_episodes=10, render=True, verbose=True):
        if render: self.env.render_mode = "human"
        results = {"rewards": [], "lengths": [], "successes": 0}
        for ep in range(1, num_episodes + 1):
            _, info = self.env.reset()
            agent_pos = self.env.agent_pos
            done = truncated = False
            total_reward = step = 0
            while not (done or truncated):
                action = self.choose_action(agent_pos, greedy=True)
                _, reward, done, truncated, info = self.env.step(action)
                agent_pos = self.env.agent_pos
                total_reward += reward
                step += 1
            results["rewards"].append(total_reward)
            results["lengths"].append(step)
            if info["at_goal"]: results["successes"] += 1
        return results

    def save(self, filename="q_agent.pkl"):
        data = {"q_table": dict(self.q_table), "epsilon": self.epsilon,
                "stats": {"rewards": self.episode_rewards, "lengths": self.episode_lengths,
                          "successes": self.success_history}}
        with open(filename, "wb") as f: pickle.dump(data, f)

    def load(self, filename="q_agent.pkl"):
        if not os.path.exists(filename): return False
        with open(filename, "rb") as f: data = pickle.load(f)
        self.q_table.update(data["q_table"])
        self.epsilon = data["epsilon"]
        return True

# ==========================================
# 3. SARSA AGENT
# ==========================================

class SARSAAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.15, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.997, min_exploration_rate=0.01):
        super().__init__(env)
        self.lr, self.gamma, self.epsilon = learning_rate, discount_factor, exploration_rate
        self.epsilon_decay, self.min_epsilon = exploration_decay, min_exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.episode_rewards, self.episode_lengths, self.success_history, self.episode_paths = [], [], [], []

    def _state_key(self, state):
        if hasattr(state, "__iter__"):
            state = np.asarray(state, dtype=np.float32)
            x_b = int(np.clip(state[0]* 20, 0, 19))
            y_b = int(np.clip(state[1] * 10, 0 ,9))
            c_b = int(state[2] > self.env.config['agent_config']['agent_concentration_threshold'])

            return (x_b, y_b, c_b)
        return state

    def choose_action(self, agent_pos, greedy=False):
        key = self._state_key(agent_pos)
        if not greedy and random.random() < self.epsilon: return self.env.action_space.sample()
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, next_action, done):
        key, next_key = self._state_key(state), self._state_key(next_state)
        current_q = self.q_table[key][action]
        target = reward if done else reward + self.gamma * self.q_table[next_key][next_action]
        self.q_table[key][action] += self.lr * (target - current_q)

    def train(self, num_episodes=2000, render_every=None, print_every=100):
        print(f"\n{'='*55}\n  TRAINING SARSA — {num_episodes} episodes\n{'='*55}")
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = truncated = False
            total_reward = step = 0
            while not (done or truncated):
                next_state, reward, done, truncated, info = self.env.step(action)
                next_action = self.choose_action(next_state) if not (done or truncated) else None
                self.update(state, action, reward, next_state, next_action, done or truncated)
                state, action = next_state, next_action
                total_reward += reward
                step += 1
            self.episode_rewards.append(total_reward)
            self.success_history.append(1 if info["at_goal"] else 0)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if ep % print_every == 0:
                print(f"  Ep {ep:>5} │ Reward: {np.mean(self.episode_rewards[-print_every:]):>+7.2f} │ Success: {np.mean(self.success_history[-print_every:])*100:.1f}%")

    def test(self, num_episodes=10, render=True, verbose=True):
        results = {"rewards": [], "lengths": [], "successes": 0}
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = truncated = False
            total_reward = step = 0
            while not (done or truncated):
                action = self.choose_action(state, greedy=True)
                state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                step += 1
            results["rewards"].append(total_reward)
            if info["at_goal"]: results["successes"] += 1
        return results

    def save(self, filename):
        with open(filename, "wb") as f: pickle.dump({"q_table": dict(self.q_table), "epsilon": self.epsilon}, f)

    def load(self, filename):
        if not os.path.exists(filename): return False
        with open(filename, "rb") as f: data = pickle.load(f)
        self.q_table.update(data["q_table"]); self.epsilon = data["epsilon"]
        return True

# ==========================================
# 4. PPO (ACTOR-CRITIC, BUFFER, AGENT)
# ==========================================

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x): return self.actor(x), self.critic(x)


class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state);
        self.actions.append(action);
        self.log_probs.append(log_prob)
        self.rewards.append(reward);
        self.dones.append(done);
        self.values.append(value)

    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def get_batch(self):
        return np.array(self.states, dtype=np.float32), np.array(self.actions), np.array(self.log_probs,
                                                                                         dtype=np.float32), \
            np.array(self.rewards, dtype=np.float32), np.array(self.dones, dtype=np.float32), np.array(self.values,
                                                                                                       dtype=np.float32)


class PPOAgent(BaseAgent):
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5, update_epochs=10, mini_batch_size=64, horizon=2048, **kwargs):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma, self.gae_lambda, self.clip_epsilon, self.value_coef, self.entropy_coef = gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef
        self.max_grad_norm, self.update_epochs, self.mini_batch_size, self.horizon = max_grad_norm, update_epochs, mini_batch_size, horizon
        self.network = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.buffer = PPOBuffer()
        self.episode_rewards, self.episode_lengths, self.success_history, self.episode_paths = [], [], [], []

    def _state_to_tensor(self, state):
        return torch.FloatTensor(state).to(self.device).unsqueeze(0) if isinstance(state,
                                                                                   np.ndarray) else torch.FloatTensor(
            state).to(self.device)

    def choose_action(self, state, greedy=False):
        state_t = self._state_to_tensor(state)
        with torch.no_grad():
            probs, value = self.network(state_t)
            dist = Categorical(probs)
            if greedy: return torch.argmax(probs).item()
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def compute_gae(self, rewards, dones, values, next_value):
        advantages, gae = [], 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages, [adv + val for adv, val in zip(advantages, values[:-1])]

    def update(self):
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get_batch()
        with torch.no_grad():
            _, last_value = self.network(self._state_to_tensor(states[-1]))
        advantages, returns = self.compute_gae(rewards.tolist(), dones.tolist(), values.tolist(), last_value.item())
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t, actions_t, old_log_probs_t, returns_t = torch.FloatTensor(states).to(self.device), torch.LongTensor(
            actions).to(self.device), torch.FloatTensor(old_log_probs).to(self.device), torch.FloatTensor(returns).to(
            self.device)

        for _ in range(self.update_epochs):
            probs, values_pred = self.network(states_t)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean() + self.value_coef * nn.MSELoss()(values_pred.squeeze(),
                                                                                    returns_t) - self.entropy_coef * dist.entropy().mean()
            self.optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm);
            self.optimizer.step()
        self.buffer.clear()

    def train(self, num_episodes=2000, render_every=None, print_every=100):
        print(f"\n{'=' * 55}\n  TRAINING PPO — {num_episodes} episodes\n{'=' * 55}")
        episode = 0
        while episode < num_episodes:
            state, _ = self.env.reset()
            done = truncated = False
            ep_reward = step = 0
            while not (done or truncated):
                action, log_prob, val = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.buffer.store(state, action, log_prob, reward, done or truncated, val)
                state, ep_reward, step = next_state, ep_reward + reward, step + 1
                if len(self.buffer.states) >= self.horizon: self.update()
            self.episode_rewards.append(ep_reward);
            self.success_history.append(1 if info["at_goal"] else 0);
            episode += 1
            if episode % print_every == 0:
                print(
                    f"  Ep {episode:>5} │ Reward: {np.mean(self.episode_rewards[-print_every:]):>+7.2f} │ Success: {np.mean(self.success_history[-print_every:]) * 100:.1f}%")
        if len(self.buffer.states) > 0: self.update()

    def test(self, num_episodes=10, render=True, verbose=True):
        results = {"rewards": [], "lengths": [], "successes": 0}
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset();
            done = truncated = False;
            total_reward = step = 0
            while not (done or truncated):
                action = self.choose_action(state, greedy=True)
                state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward;
                step += 1
            results["rewards"].append(total_reward);
            results["lengths"].append(step)
            if info["at_goal"]: results["successes"] += 1
        return results

    def save(self, filename):
        torch.save({"network_state_dict": self.network.state_dict()}, filename)

    def load(self, filename):
        if not os.path.exists(filename): return False
        self.network.load_state_dict(torch.load(filename)["network_state_dict"])
        return True