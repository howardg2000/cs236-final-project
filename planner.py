import numpy as np


class Planner:
    def __init__(self, num_actions, action_means, transition_probs) -> None:
        self.num_actions = num_actions
        self.action_means = np.array(action_means)
        self.transition_probs = np.array(transition_probs)

        self.last_explored = np.array([-np.inf] * num_actions)
        self.rewards = np.array([-np.inf] * num_actions)

    def get_action(self, time_step):
        """Get next action to explore."""
        raise NotImplementedError

    def record_reward(self, action, timestep, reward):
        """Record reward for action."""
        raise NotImplementedError


class GreedyPlanner(Planner):
    def __init__(self, num_actions, action_means, transition_probs) -> None:
        super().__init__(num_actions, action_means, transition_probs)

    def get_action(self, time_step):
        """Get next action to explore."""
        if time_step < self.num_actions:
            return time_step
        time_since_explored = time_step - self.last_explored
        prob_of_same_reward = (1-self.transition_probs) ** time_since_explored
        expected_rewards = prob_of_same_reward * self.rewards + \
            (1 - prob_of_same_reward) * self.action_means
        return np.argmax(expected_rewards)

    def record_reward(self, action, timestep, reward):
        """Record reward for action."""
        self.last_explored[action] = timestep
        self.rewards[action] = reward
