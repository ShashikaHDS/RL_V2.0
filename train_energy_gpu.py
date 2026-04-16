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
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from map_gen_v8 import MapGen
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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


from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    """Logs ep_rew_mean using SB3's internal ep_info_buffer."""
    def __init__(self, log_freq=2048):
        super().__init__()
        self.data = []
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                self.data.append((self.num_timesteps, mean_rew))
        return True


if __name__ == '__main__':

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
            vec_env = DummyVecEnv(env_fns)

            model = PPO("MultiInputPolicy", vec_env, verbose=1,
                        ent_coef=0.05, device="cuda", learning_rate=3e-4,
                        seed=seed,
                        n_steps=2048,
                        batch_size=2048,
                        n_epochs=10)

            callback = RewardLogger()
            model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=1, callback=callback)

            vec_env.close()
            results[label].append(callback.data)
            print(f"  Done: {len(callback.data)} data points")

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
