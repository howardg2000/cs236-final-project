import numpy as np
import matplotlib.pyplot as plt

file_name = "explore_rewards_1to60_0.1x0.1.npy"
explore_freqs = list(range(1, 60)) + [100000000]

explore_rewards = np.load(file_name)
print(explore_rewards.shape)

means = explore_rewards.mean(axis=1)
stds = explore_rewards.std(axis=1)

print(means[-1])

plt.plot(explore_freqs[:-1], means[:-1])
plt.axhline(y=means[-1], color='r', linestyle='-', label="No exploration")
plt.title("Mean reward vs exploration frequency p=(0.9,0.1)")
plt.xlabel("Exploration frequency")
plt.ylabel("Mean reward (10k steps)")
plt.legend()
