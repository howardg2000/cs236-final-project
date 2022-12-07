import numpy as np


class Planner:
    def __init__(self, num_actions, action_means, transition_probs) -> None:
        self.num_actions = num_actions
        self.action_means = np.array(action_means)
        self.transition_probs = np.array(transition_probs)

        self.last_explored = np.array([-np.inf] * num_actions)
        self.rewards = np.array([-np.inf] * num_actions)
        self.explored_consecutively = np.zeros(num_actions)

    def get_action(self, time_step):
        """Get next action to explore."""
        raise NotImplementedError

    def record_reward(self, action, timestep, reward):
        """Record reward for action."""
        raise NotImplementedError


class GreedyExplorer(Planner):
    def __init__(self, num_actions, action_means, transition_probs, explore_freq) -> None:
        super().__init__(num_actions, action_means, transition_probs)

        self.explore_freq = explore_freq

    def get_action(self, time_step):
        """Get next action to explore."""
        if time_step < self.num_actions:
            return time_step
        time_since_explored = time_step - self.last_explored
        prob_of_same_reward = (1-self.transition_probs) ** time_since_explored
        expected_rewards = prob_of_same_reward * self.rewards + \
            (1 - prob_of_same_reward) * self.action_means

        best_action = np.argmax(expected_rewards)

        if self.explored_consecutively[best_action] >= self.explore_freq:
            selected_action = np.random.randint(self.num_actions)
        else:
            selected_action = best_action

        # Update the number of times the action has been explored consecutively
        prev_explores = self.explored_consecutively[selected_action]
        self.explored_consecutively = np.zeros(self.num_actions)
        self.explored_consecutively[selected_action] = prev_explores + 1
        return selected_action

    def record_reward(self, action, timestep, reward):
        """Record reward for action."""
        self.last_explored[action] = timestep
        self.rewards[action] = reward
