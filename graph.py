import matplotlib.pyplot as plt
import numpy as np

# Sample data
group_labels = ['3 robots', '4 robots', '5 robots']
robot1 = [23, 28, 52]
robot2 = [27, 33, 42]
robot3 = [22, 30, 45]
robot4 = [0, 28, 36]
robot5 = [0, 0, 51]

n_groups = len(group_labels)
n_bars_per_group = 5
bar_width = 0.15

# Control spacing between groups (reduce this for tighter groups)
group_spacing = 0.5  # Reduce this to make groups closer together (default ~1.0)
group_x = np.arange(n_groups) * (n_bars_per_group * bar_width + group_spacing)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(group_x - 2*bar_width, robot1, bar_width, label='Robot 1')
ax.bar(group_x - bar_width, robot2, bar_width, label='Robot 2')
ax.bar(group_x, robot3, bar_width, label='Robot 3')
ax.bar(group_x + bar_width, robot4, bar_width, label='Robot 4')
ax.bar(group_x + 2*bar_width, robot5, bar_width, label='Robot 5')

# Labels and ticks
ax.set_xticks(group_x)
ax.set_xticklabels(group_labels)
ax.set_title('Average traveled distance by robots in test case 1')
ax.set_ylabel('Average Distance')
ax.legend()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()
