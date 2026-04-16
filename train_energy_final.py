"""
Train 60%, 70%, 80% energy budgets using SB3 verbose logging.
Captures ep_rew_mean from SB3's internal rolling average (same as original graph).
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import random
import time
import re
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from map_gen_v8 import MapGen
from stable_baselines3 import PPO
from chemical_v3_no_visual import ChemicalClean

REWARDS = {
    "clean_spill": 50,
    "consecutive_bonus": 150,
    "revisit_penalty": -10,
    "explore_penalty": -10,
    "obstacle_penalty": -5,
}

BATTERY_LEVELS = [0.60, 0.70, 0.80]
TOTAL_TIMESTEPS = 500_000


def make_env(battery_level):
    grid_maps = MapGen()
    grid_map = np.array(grid_maps.generate_main_with_subclusters_map(
        20, 20, (30, 60),
        {"num_range": (4, 8), "size_range": (5, 15),
         "distance_from_main": (9, 10), "distance_between_subclusters": 8}
    ))
    free_cells = np.argwhere(grid_map == 0)
    idx = random.sample(range(len(free_cells)), 1)
    robot_pos = np.array([tuple(free_cells[i]) for i in idx])
    env = ChemicalClean(grid_map, num_robots=1, robot_positions=robot_pos,
                        field_of_view=0, chem_fov=1, rewards=REWARDS,
                        battery_level=battery_level)
    return env


class StdoutCapture:
    """Captures stdout to extract ep_rew_mean and total_timesteps from SB3 output."""
    def __init__(self, original):
        self.original = original
        self.data = []  # (timesteps, ep_rew_mean)
        self._buffer = ""
        self._current_timesteps = None
        self._current_reward = None

    def write(self, text):
        self.original.write(text)
        self._buffer += text
        # Parse SB3 output lines
        for line in self._buffer.split("\n"):
            line = line.strip()
            if "total_timesteps" in line and "|" in line:
                try:
                    val = line.split("|")[-2].strip()
                    self._current_timesteps = int(val)
                except:
                    pass
            if "ep_rew_mean" in line and "|" in line:
                try:
                    val = line.split("|")[-2].strip()
                    self._current_reward = float(val)
                except:
                    pass
            if self._current_timesteps is not None and self._current_reward is not None:
                self.data.append((self._current_timesteps, self._current_reward))
                self._current_timesteps = None
                self._current_reward = None
        # Keep only last incomplete line in buffer
        if "\n" in self._buffer:
            self._buffer = self._buffer.rsplit("\n", 1)[-1]

    def flush(self):
        self.original.flush()


results = {}

for bl in BATTERY_LEVELS:
    label = f"{int(bl*100)}%"
    print(f"\n  Training {label}...")

    env = make_env(bl)
    model = PPO("MultiInputPolicy", env, verbose=1,
                ent_coef=0.05, device="cuda", learning_rate=3e-4)

    # Capture SB3's ep_rew_mean output
    capture = StdoutCapture(sys.stdout)
    sys.stdout = capture

    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1)

    sys.stdout = capture.original
    results[label] = capture.data
    print(f"  {label} done: {len(capture.data)} data points")

# Plot - same style as original pink graph
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"60%": "#E53935", "70%": "#1E88E5", "80%": "#43A047"}

for label, data in results.items():
    if not data:
        continue
    ts = [d[0] for d in data]
    rw = [d[1] for d in data]
    ax.plot(ts, rw, color=colors[label], linewidth=1.5, label=label)

ax.set_xlabel("Time steps", fontsize=12)
ax.set_ylabel("Mean episode reward", fontsize=12)
ax.set_title("Training Convergence Under Varying Energy Budgets", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sensitivity_results/energy_budget_convergence.png", dpi=150)
plt.close()
print(f"\nPlot saved: sensitivity_results/energy_budget_convergence.png")
