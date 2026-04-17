"""
Train 60%, 70%, 80% energy budgets with 5 seeds each.
Runs all 15 jobs in parallel using multiprocessing.
Plots median +/- std bands.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARDS = {
    "clean_spill": 5,
    "consecutive_bonus": 40,
    "revisit_penalty": -2,
    "explore_penalty": -5,
    "obstacle_penalty": -5,
}

BATTERY_LEVELS = [0.60, 0.70, 0.80]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 2_000_000


def train_one(args):
    bl, seed = args
    label = f"{int(bl*100)}%"

    import os, random, numpy as np
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame
    pygame.init()
    pygame.display.set_mode((1, 1))

    from map_gen_v8 import MapGen
    from stable_baselines3 import PPO
    from chemical_v3_no_visual import ChemicalClean

    random.seed(seed)
    np.random.seed(seed)

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
                        battery_level=bl)

    model = PPO("MultiInputPolicy", env, verbose=0,
                ent_coef=0.05, device="cpu", learning_rate=3e-4, seed=seed)

    # Collect ep_rew_mean
    data = []
    steps_per_learn = 2048
    total = 0
    while total < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=steps_per_learn, log_interval=0, reset_num_timesteps=False)
        total += steps_per_learn
        if len(model.ep_info_buffer) > 0:
            mean_rew = np.mean([ep["r"] for ep in model.ep_info_buffer])
            data.append((total, mean_rew))

    print(f"  {label} seed={seed}: {len(data)} points")
    return (label, seed, data)


if __name__ == '__main__':
    from multiprocessing import Pool

    os.makedirs("sensitivity_results", exist_ok=True)

    jobs = [(bl, seed) for bl in BATTERY_LEVELS for seed in SEEDS]
    print(f"Running {len(jobs)} jobs in parallel across 15 cores...")
    print(f"Timesteps per job: {TOTAL_TIMESTEPS:,}")

    with Pool(15) as pool:
        all_results = pool.map(train_one, jobs)

    # Organize results
    results = {}
    for label, seed, data in all_results:
        if label not in results:
            results[label] = []
        results[label].append(data)

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
