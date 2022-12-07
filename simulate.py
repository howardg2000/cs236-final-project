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
    explore_freqs = list(range(1, 60)) + [100000000]
    explore_rewards = []
    optimal_explore_rewards = []
    transition_probs = [0.1, 0.1]
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
    np.save("explore_rewards_1to60_0.1x0.1.npy", explore_rewards)
    np.save("optimal_explore_rewards_1to60_0.1x0.1.npy",
            optimal_explore_rewards)

    # plt.plot([1, 5, 10, 20, 50, 100, 10000000000], explore_rewards)
