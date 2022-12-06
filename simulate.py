import numpy as np
from mdp import MDP


class Planner:
    def __init__(self, num_actions, action_means, transition_probs) -> None:
        self.num_actions = num_actions
        self.action_means = np.array(action_means)
        self.transition_probs = np.array(transition_probs)

        self.last_explored = np.array([-np.inf] * num_actions)
        self.rewards = np.array([-np.inf] * num_actions)

    def get_action(self, time_step):
        """Get next action to explore."""
        if time_step < self.num_actions:
            return time_step
        time_since_explored = time_step - self.last_explored
        prob_of_reward_change = transition_probs ** time_since_explored
        expected_rewards = prob_of_reward_change * self.action_means + \
            (1 - prob_of_reward_change) * self.rewards
        return np.argmax(expected_rewards)

    def record_reward(self, action, timestep, reward):
        """Record reward for action."""
        self.last_explored[action] = timestep
        self.rewards[action] = reward


if __name__ == "__main__":
    np.random.seed(50)

    num_actions = 2
    reward_dists = [np.random.uniform, np.random.uniform]
    reward_params = [(-1, 5), (-5, 5)]
    transition_probs = [0.1, 0.1]
    env = MDP(num_actions, reward_dists, reward_params, transition_probs)

    reward_means = [(param[0] + param[1]) / 2 for param in reward_params]
    planner = Planner(num_actions, reward_means, transition_probs)

    n_iter = 25
    total_reward = 0.0
    optimal_reward = 0.0
    for t in range(n_iter):
        action = planner.get_action(t)
        reward, all_rewards = env.get_reward(action, verbose=True)
        # print(all_rewards, action)
        planner.record_reward(action, t, reward)

        total_reward += reward
        optimal_reward += max(all_rewards)

    print(total_reward, optimal_reward)
