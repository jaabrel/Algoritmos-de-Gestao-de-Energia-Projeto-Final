from main import PlumeRLWrapper, CONFIG 
from QLearning import QLearningAgent

env = PlumeRLWrapper(CONFIG)
agent = QLearningAgent(env)

agent.train(env, num_episodes=10, print_every=1)

print(f"Avg reward: {np.mean(agent.episode_rewards)}")  # Should be 10-50
print(f"Avg length: {np.mean(agent.episode_lengths)}")
