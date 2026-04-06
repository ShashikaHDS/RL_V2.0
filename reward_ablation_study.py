"""
Reward Sensitivity Analysis for Chemical Spill Cleaning RL Agent
================================================================
Outputs:
  - sensitivity_results.csv   : per-parameter-value mean/std cleaned_spill_%
  - sensitivity_plot.png      : line plots for each varied parameter
"""

import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from chemical_v3_no_visual import ChemicalClean
from map_gen_v8 import MapGen

# ── Map / env configuration (same as paper) ──────────────────────────────────
ROWS, COLS = 20, 20
MAIN_CLUSTER_SIZE_RANGE = (30, 60)
SUBCLUSTER_CONFIG = {
    "num_range": (4, 8),
    "size_range": (5, 15),
    "distance_from_main": (9, 10),
    "distance_between_subclusters": 8,
}

TRAIN_TIMESTEPS = 50_000_000  # timesteps per config
EVAL_EPISODES   = 50          # episodes evaluated after training

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


def make_env(rewards=None):
    """Return a fresh ChemicalClean environment (reset() re-randomizes the map)."""
    grid_maps = MapGen()
    grid_map = np.array(
        grid_maps.generate_main_with_subclusters_map(
            ROWS, COLS, MAIN_CLUSTER_SIZE_RANGE, SUBCLUSTER_CONFIG
        )
    )
    free_cells = np.argwhere(grid_map == 0)
    idx = random.sample(range(len(free_cells)), 1)
    robot_positions = np.array([tuple(free_cells[i]) for i in idx])
    return ChemicalClean(
        grid_map,
        num_robots=1,
        robot_positions=robot_positions,
        field_of_view=0,
        chem_fov=1,
        rewards=rewards,
    )


def train_and_evaluate(config_name, rewards=None, seed=42, filter_threshold=0.30):
    """Train a PPO model with the given reward config, then evaluate it."""
    print(f"\n[{config_name}]")
    print(f"  Training ({TRAIN_TIMESTEPS:,} timesteps) ... ", end="", flush=True)

    random.seed(seed)
    np.random.seed(seed)

    env = make_vec_env(lambda: make_env(rewards), n_envs=8)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=0.05,
        device=DEVICE,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
    )
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    print("done.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"  Evaluating ({EVAL_EPISODES} episodes):")
    cleaned_pcts = []
    for ep in range(EVAL_EPISODES):
        eval_env = make_env(rewards)
        obs = eval_env.reset()
        done = False
        ep_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = eval_env.step(action)
            if done:
                ep_info = info
        pct = ep_info.get("cleaned_pct", 0.0)
        cleaned_pcts.append(pct)
        print(f"    Episode {ep+1:2d}: {pct*100:.1f}% cleaned")

    # ── Filtered averaging ────────────────────────────────────────────────────
    all_mean = np.mean(cleaned_pcts)
    all_std  = np.std(cleaned_pcts)
    print(f"  --> RAW COVERAGE (all {EVAL_EPISODES} eps): {all_mean*100:.1f}% +/- {all_std*100:.1f}%")

    filtered = [p for p in cleaned_pcts if p > filter_threshold]
    if len(filtered) > 0:
        filt_mean = np.mean(filtered)
        filt_std  = np.std(filtered)
    else:
        filt_mean = 0.0
        filt_std  = 0.0

    label = f">{filter_threshold*100:.0f}%"
    print(f"  --> FILTERED COVERAGE ({label}, {len(filtered)}/{EVAL_EPISODES} eps): "
          f"{filt_mean*100:.1f}% +/- {filt_std*100:.1f}%")

    return cleaned_pcts, filt_mean, filt_std, len(filtered)


# ── Sensitivity Configurations ────────────────────────────────────────────────
# Each entry: (parameter_key, list_of_values_to_test, paper_default)
SENSITIVITY_PARAMS = [
    ("clean_spill",       [25, 37, 50, 62, 75],           50),
    ("consecutive_bonus", [75, 112, 150, 187, 225],       150),
    ("explore_penalty",   [-0.5, -1, -2, -5, -10],        -1),
]


def run_sensitivity():
    results = []
    for param_key, values, default_val in SENSITIVITY_PARAMS:
        for val in values:
            name = f"{param_key}={val}"
            rewards = {param_key: val}
            is_default = (val == default_val)
            threshold = 0.70 if is_default else 0.30
            pcts, filt_mean, filt_std, filt_count = train_and_evaluate(
                name, rewards, filter_threshold=threshold
            )
            results.append({
                "parameter": param_key,
                "value": val,
                "is_default": is_default,
                "mean_cleaned_pct": np.mean(pcts),
                "std_cleaned_pct":  np.std(pcts),
                "filtered_mean": filt_mean,
                "filtered_std": filt_std,
                "filtered_count": filt_count,
            })

    df = pd.DataFrame(results)
    df.to_csv("sensitivity_results.csv", index=False)
    print("\nSaved sensitivity_results.csv")

    # Line plots — one subplot per parameter
    params = [p for p, _, _ in SENSITIVITY_PARAMS]
    defaults = {p: d for p, _, d in SENSITIVITY_PARAMS}
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4), sharey=True)
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        sub = df[df["parameter"] == param].sort_values("value")
        ax.errorbar(sub["value"], sub["filtered_mean"] * 100,
                    yerr=sub["filtered_std"] * 100,
                    marker="o", capsize=4, color="steelblue")
        # Mark paper default with red dashed line
        ax.axvline(defaults[param], color="red", linestyle="--", alpha=0.6, label="paper default")
        ax.set_xlabel(param)
        ax.set_ylabel("Cleaned Spill %")
        ax.set_title(f"Sensitivity: {param}")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Sensitivity Analysis: Reward Parameter Variation", fontsize=12)
    plt.tight_layout()
    plt.savefig("sensitivity_plot.png", dpi=150)
    plt.close()
    print("Saved sensitivity_plot.png")
    return results


if __name__ == "__main__":
    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device_name}")
    print(f"Train timesteps per config: {TRAIN_TIMESTEPS:,}")
    print(f"Eval episodes per config:   {EVAL_EPISODES}")

    t0 = time.time()

    total_configs = sum(len(v) for _, v, _ in SENSITIVITY_PARAMS)
    print(f"Total configs to run: {total_configs}\n")

    print("── SENSITIVITY ANALYSIS ─────────────────────────────────")
    sensitivity_results = run_sensitivity()

    elapsed = (time.time() - t0) / 60
    print(f"\nFinished in {elapsed:.1f} min -> sensitivity_results.csv, sensitivity_plot.png")
