"""
GPU-optimized training for 60%, 70%, 80% energy budgets.
Uses SubprocVecEnv with 16 parallel environments to maximize GPU utilization.
5 seeds per budget, plots median +/- std.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import random
import time
import sys
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from map_gen_v8 import MapGen
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from chemical_v3_no_visual import ChemicalClean

REWARDS = {
    "clean_spill": 50,
    "consecutive_bonus": 150,
    "revisit_penalty": -10,
    "explore_penalty": -10,
    "obstacle_penalty": -5,
}

BATTERY_LEVELS = [0.60, 0.70, 0.80]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 500_000
N_ENVS = 16  # parallel environments to feed the GPU

os.makedirs("sensitivity_results", exist_ok=True)


def make_env_fn(battery_level, seed, env_id):
    """Returns a function that creates an environment (needed for SubprocVecEnv)."""
    def _init():
        random.seed(seed + env_id)
        np.random.seed(seed + env_id)
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
    return _init


class StdoutCapture:
    def __init__(self, original):
        self.original = original
        self.data = []
        self._buffer = ""
        self._current_timesteps = None
        self._current_reward = None

    def write(self, text):
        self.original.write(text)
        self._buffer += text
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
        if "\n" in self._buffer:
            self._buffer = self._buffer.rsplit("\n", 1)[-1]

    def flush(self):
        self.original.flush()


results = {}
total_runs = len(BATTERY_LEVELS) * len(SEEDS)
run = 0

for bl in BATTERY_LEVELS:
    label = f"{int(bl*100)}%"
    results[label] = []

    for seed in SEEDS:
        run += 1
        print(f"\n[{run}/{total_runs}] Battery={label}, Seed={seed}, Envs={N_ENVS}")

        # Create parallel environments
        env_fns = [make_env_fn(bl, seed, i) for i in range(N_ENVS)]
        vec_env = SubprocVecEnv(env_fns)

        model = PPO("MultiInputPolicy", vec_env, verbose=1,
                    ent_coef=0.05, device="cuda", learning_rate=3e-4,
                    seed=seed,
                    n_steps=2048,       # steps per env before update
                    batch_size=2048,    # larger batch for GPU
                    n_epochs=10)

        capture = StdoutCapture(sys.stdout)
        sys.stdout = capture
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1)
        sys.stdout = capture.original

        vec_env.close()
        results[label].append(capture.data)
        print(f"  Done: {len(capture.data)} data points")

# Save raw data
with open("sensitivity_results/energy_budget_data.pkl", "wb") as f:
    pickle.dump(results, f)

# ---- Plot: median +/- std across seeds ----
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"60%": "#E53935", "70%": "#1E88E5", "80%": "#43A047"}

for label, seed_curves in results.items():
    all_ts = set()
    for curve in seed_curves:
        for ts, rw in curve:
            all_ts.add(ts)
    grid = np.array(sorted(all_ts))

    interpolated = []
    for curve in seed_curves:
        if not curve:
            continue
        ts_arr = np.array([d[0] for d in curve])
        rw_arr = np.array([d[1] for d in curve])
        interp = np.interp(grid, ts_arr, rw_arr)
        interpolated.append(interp)

    if not interpolated:
        continue

    stacked = np.array(interpolated)
    median = np.median(stacked, axis=0)
    std = np.std(stacked, axis=0)

    ax.plot(grid, median, color=colors[label], linewidth=2, label=label)
    ax.fill_between(grid, median - std, median + std,
                    color=colors[label], alpha=0.15)

ax.set_xlabel("Time steps", fontsize=12)
ax.set_ylabel("Mean episode reward", fontsize=12)
ax.set_title("Training Convergence Under Varying Energy Budgets\n(median +/- std across 5 seeds)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sensitivity_results/energy_budget_convergence.png", dpi=150)
plt.close()
print(f"\nPlot saved: sensitivity_results/energy_budget_convergence.png")
