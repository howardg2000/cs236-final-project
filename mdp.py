import numpy as np


class MDP:
    def __init__(self, num_actions, r_dists, r_dist_params, transition_probs) -> None:
        """Create new MDP object.

        Args:
            num_actions (int): Number of actions available.
            r_dists (List[np distribution]): List of reward distribution objects, i.e. [np.random.uniform, np.random.normal]
            r_dist_params (List[Tuple]): List of reward distribution parameters, i.e. [(0, 1), (0, 1)]
            transition_probs (List[float]): List of transition probabilities for each action, i.e. [0.8, 0.1, 0.1]
        """
        assert num_actions == len(r_dists)
        assert num_actions == len(r_dist_params)
        assert num_actions == len(transition_probs)

        self.num_actions = num_actions
        self.r_dists = r_dists
        self.r_dist_params = r_dist_params
        self.transition_probs = np.array(transition_probs)

        self.rewards = self._draw_rewards()

    def _draw_rewards(self):
        """Draw new rewards from distributions."""
        return [dist(*self.r_dist_params[i])
                for i, dist in enumerate(self.r_dists)]

    def get_reward(self, action, verbose=False):
        """Get next reward for action (int)."""
        assert action < self.num_actions

        rewards_to_change = 1 * (np.random.random(
            size=self.num_actions) <= self.transition_probs)
        rewards_to_keep = 1 - rewards_to_change
        self.new_rewards = self._draw_rewards()
        self.rewards = rewards_to_change * \
            self.new_rewards + rewards_to_keep * self.rewards
        if verbose:
            return self.rewards[action], self.rewards
        return self.rewards[action]
