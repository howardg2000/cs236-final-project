import numpy as np
from mdp import MDP
from planner import GreedyPlanner


class Simulation:
    def __init__(self, num_actions, reward_dists, reward_params, transition_probs, planner) -> None:
        self.env = MDP(num_actions, reward_dists,
                       reward_params, transition_probs)
        self.reward_means = [(param[0] + param[1]) /
                             2 for param in reward_params]
        self.planner = planner(
            num_actions, self.reward_means, transition_probs)

    def simulate(self, n_iter, verbose=False):
        total_reward = 0.0
        optimal_reward = 0.0
        for t in range(n_iter):
            action = self.planner.get_action(t)
            reward, all_rewards = self.env.get_reward(action, verbose=True)
            if verbose:
                print(all_rewards, action)
            self.planner.record_reward(action, t, reward)

            total_reward += reward
            optimal_reward += max(all_rewards)

        print(total_reward, optimal_reward)


if __name__ == "__main__":
    np.random.seed(50)

    num_actions = 2
    reward_dists = [np.random.uniform, np.random.uniform]
    reward_params = [(-1, 5), (-5, 5)]
    transition_probs = [0.1, 0.1]
    planner = GreedyPlanner

    sim = Simulation(num_actions, reward_dists,
                     reward_params, transition_probs, planner)

    sim.simulate(25, verbose=True)
