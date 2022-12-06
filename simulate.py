import numpy as np
from mdp import MDP

if __name__ == "__main__":
    np.random.seed(42)

    env = MDP(2, [np.random.uniform, np.random.normal],
              [(-1, 5), (-5, 5)], [0.1, 0.1])

    n_iter = 10
    for i in range(n_iter):
        reward, all_rewards = env.get_reward(0, verbose=True)
        print(all_rewards)
