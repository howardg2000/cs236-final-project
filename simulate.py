import numpy as np
from mdp import MDP
from planner import GreedyExplorer
from tqdm import tqdm
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, num_actions, reward_dists, reward_params, transition_probs, planner, explore_freq) -> None:
        self.env = MDP(num_actions, reward_dists,
                       reward_params, transition_probs)
        self.reward_means = [(param[0] + param[1]) /
                             2 for param in reward_params]
        self.planner = planner(
            num_actions, self.reward_means, transition_probs, explore_freq)

    def simulate(self, n_iter, verbose=False, output_steps=False):
        total_reward = 0.0
        optimal_reward = 0.0
        rewards_per_step = []
        for t in range(n_iter):
            action = self.planner.get_action(t)
            reward, all_rewards = self.env.get_reward(action, verbose=True)
            if verbose:
                print(all_rewards, action)
            self.planner.record_reward(action, t, reward)

            total_reward += reward
            rewards_per_step.append(reward)
            optimal_reward += max(all_rewards)
        if output_steps:
            return total_reward, optimal_reward, rewards_per_step
        return total_reward, optimal_reward


if __name__ == "__main__":
    min_freq = 1
    max_freq = 200
    step = 5
    explore_freqs = list(range(min_freq, max_freq, step)) + [100000000]
    explore_rewards = []
    optimal_explore_rewards = []
    transition_probs = [0.05, 0.01]
    print(transition_probs)
    for explore_freq in tqdm(explore_freqs):
        total_rewards = []
        optimal_rewards = []
        for _ in tqdm(range(100)):
            num_actions = 2
            reward_dists = [np.random.uniform, np.random.uniform]
            reward_params = [(-1, 5), (-5, 5)]

            planner = GreedyExplorer

            sim = Simulation(num_actions, reward_dists,
                             reward_params, transition_probs, planner, explore_freq=explore_freq)

            num_steps = 10000
            total, optimal = sim.simulate(num_steps)
            total_rewards.append(total)
            optimal_rewards.append(optimal)
        print(np.mean(total_rewards))
        explore_rewards.append(total_rewards)
        optimal_explore_rewards.append(optimal_rewards)

    explore_rewards = np.array(explore_rewards)
    optimal_explore_rewards = np.array(optimal_explore_rewards)
    params = f'{str(min_freq)}to{str(max_freq)}_{str(transition_probs[0])}x{str(transition_probs[1])}'
    np.save(f"data/total_rewards/explore_rewards_{params}.npy",
            explore_rewards)
    np.save(f"data/optimal_rewards/optimal_explore_rewards_{params}.npy",
            optimal_explore_rewards)
