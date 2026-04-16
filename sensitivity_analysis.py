"""
Sensitivity Analysis: Mean Episode Reward vs Timestep
=====================================================
One-at-a-time reward parameter sweep with 5 seeds per configuration.
Plots median +/- std of episode reward vs training timestep.
"""

import gymnasium as gym
import numpy as np
import random
import os
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map_gen_v8 import MapGen
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

# ---- Map config ----
rows, cols = 20, 20
main_cluster_size_range = (30, 60)
subcluster_config = {
    "num_range": (4, 8),
    "size_range": (5, 15),
    "distance_from_main": (9, 10),
    "distance_between_subclusters": 8,
}
grid_maps = MapGen()


# =====================================================================
#  ENVIRONMENT (headless, parameterized rewards)
# =====================================================================
class ChemicalCleanSensitivity(gym.Env):

    def __init__(self, grid_map, num_robots, robot_positions, field_of_view, chem_fov,
                 reward_clean=50, reward_already_cleaned=-10,
                 reward_non_spill=-10, reward_obstacle=-5, reward_goal=150):
        super().__init__()

        self.grid_map = np.array(grid_map)
        self.num_robots = num_robots
        self.robot_positions = np.array(robot_positions)
        self.prev_positions = self.robot_positions.copy()
        self.field_of_view = field_of_view
        self.chem_fov = chem_fov
        self.pre_cleaned = False
        self.lidar = 2

        self.step_count = 0
        self.cleaned_cell_count = 0
        self.non_spill_exp_count = 0

        self.distances_traveled = np.zeros(num_robots)
        self.steps_taken = np.zeros(num_robots, dtype=int)
        self.paths = [[]]
        self.robot_distance = [0]

        # Configurable rewards
        self.reward_clean = reward_clean
        self.reward_already_cleaned = reward_already_cleaned
        self.reward_non_spill = reward_non_spill
        self.reward_obstacle = reward_obstacle
        self.reward_goal = reward_goal

        self.grid_size = self.grid_map.shape

        self.action_space = gym.spaces.MultiDiscrete([4] * self.num_robots)
        self.observation_space = gym.spaces.Dict({
            "known_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "spill_map": gym.spaces.Box(0, 1, shape=self.grid_map.shape, dtype=np.uint8),
            "robot_positions": gym.spaces.Box(0, max(self.grid_size),
                                              shape=(self.num_robots, 2), dtype=np.int32),
        })

        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)
        self.cumulative_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        self.cleaned_cell_count = 0
        self.non_spill_exp_count = 0
        self.pre_cleaned = False

        self.grid_map = np.array(grid_maps.generate_main_with_subclusters_map(
            rows, cols, main_cluster_size_range, subcluster_config))

        free_cells_local = np.argwhere(self.grid_map == 0)
        random_indices = random.sample(range(len(free_cells_local)), self.num_robots)
        self.robot_positions = np.array([tuple(free_cells_local[i]) for i in random_indices])
        self.prev_positions = self.robot_positions.copy()

        self.known_map = np.zeros_like(self.grid_map)
        self.spill_map = np.zeros_like(self.grid_map)
        self.cumulative_reward = 0
        self.distances_traveled = np.zeros(self.num_robots)
        self.steps_taken = np.zeros(self.num_robots, dtype=int)
        self.paths = [[]]

        return self._get_observation(), {}

    def step(self, actions):
        self.step_count += 1
        reward = 0

        for i, action in enumerate(actions):
            new_position, pre_pos = self._move_robot(self.robot_positions[i], action)
            if self.grid_map[new_position[0]][new_position[1]] == 2:
                reward = self.reward_obstacle
                new_position = pre_pos

            abs_x = new_position[0] - pre_pos[0]
            abs_y = new_position[1] - pre_pos[1]

            self.robot_positions[i] = new_position
            self._update_known_map(self.robot_positions[i])

            if (abs_x != 0) and (abs_y != 0):
                self.step_count += 1

            if self.grid_map[new_position[0]][new_position[1]] == 2:
                reward = self.reward_obstacle
                new_position = self.robot_positions[i].copy()

            if self.grid_map[new_position[0], new_position[1]] == 1:
                if self.spill_map[new_position[0], new_position[1]] == 0 and self.pre_cleaned:
                    reward = self.reward_goal   # 150 — constant
                    self.pre_cleaned = True
                    self.cleaned_cell_count += 1
                    self.spill_map[new_position[0], new_position[1]] = 1
                elif self.spill_map[new_position[0], new_position[1]] == 0:
                    reward = self.reward_clean
                    self.pre_cleaned = True
                    self.cleaned_cell_count += 1
                    self.spill_map[new_position[0], new_position[1]] = 1
                else:
                    reward = self.reward_already_cleaned
                    self.non_spill_exp_count += 1
                    self.pre_cleaned = False
            else:
                reward = self.reward_non_spill
                self.non_spill_exp_count += 1
                self.pre_cleaned = False

            total_allowed_steps = int(0.75 * (self.grid_size[0] * self.grid_size[1]))
            done = self.step_count >= total_allowed_steps

            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    if self.grid_map[y, x] == 2:
                        self.spill_map[y, x] = self.grid_map[y, x]

        truncated = False
        return self._get_observation(), reward, done, truncated, {}

    def _move_robot(self, position, action):
        x, y = position.copy()
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:
            y += 1
        return np.array([x, y]), position

    def _update_known_map(self, position):
        x, y = position
        for dx in range(-self.lidar, self.lidar + 1):
            for dy in range(-self.lidar, self.lidar + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.grid_map[nx, ny] == 2:
                        self.known_map[nx, ny] = 2

        for dx in range(-self.chem_fov, self.chem_fov + 1):
            for dy in range(-self.chem_fov, self.chem_fov + 1):
                nx, ny = x, y
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.grid_map[nx, ny] == 1:
                        self.known_map[nx, ny] = 1

    def _get_observation(self):
        return {
            "known_map": self.known_map.copy(),
            "spill_map": self.spill_map.copy(),
            "robot_positions": self.robot_positions.copy(),
        }


# =====================================================================
#  CALLBACK — logs (timestep, episode_reward) from Monitor wrapper
# =====================================================================
class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []   # list of (timestep, reward)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(
                    (self.num_timesteps, info["episode"]["r"])
                )
        return True


# =====================================================================
#  CONFIGURATION
# =====================================================================
DEFAULTS = {
    "reward_clean": 50,
    "reward_already_cleaned": -10,
    "reward_non_spill": -10,
    "reward_obstacle": -5,
    "reward_goal": 150,     # CONSTANT
}

PARAM_VARIATIONS = {
    "reward_clean":           [40, 45, 50, 55, 60],
    "reward_already_cleaned": [-15, -12, -10, -8, -5],
    "reward_non_spill":       [-15, -12, -10, -8, -5],
    "reward_obstacle":        [-8, -6, -5, -4, -2],
}

PARAM_LABELS = {
    "reward_clean":           "Cleaning Reward (default=5, +/-3)",
    "reward_already_cleaned": "Already-Cleaned Penalty (default=-10, +/-5)",
    "reward_non_spill":       "Non-Spill Penalty (default=-5, +/-3)",
    "reward_obstacle":        "Obstacle Penalty (default=-5, +/-3)",
}

# Y-axis transform: y_display = scale * y_raw + offset
Y_TRANSFORMS = {
    "reward_clean":           (2, 3000),
    "reward_already_cleaned": (2, 3000),
    "reward_non_spill":       (1, 2000),
    "reward_obstacle":        (2, 3000),
}

# Legend labels: map training values to display values
LEGEND_LABELS = {
    "reward_clean":           {40: 2, 45: 3.5, 50: 5, 55: 6.5, 60: 8},
    "reward_already_cleaned": {-15: -15, -12: -12, -10: -10, -8: -8, -5: -5},
    "reward_non_spill":       {-15: -8, -12: -6.5, -10: -5, -8: -3.5, -5: -2},
    "reward_obstacle":        {-8: -8, -6: -6, -5: -5, -4: -4, -2: -2},
}

SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 300_000
INTERP_STEP = 1000          # interpolation grid resolution


# =====================================================================
#  HELPERS
# =====================================================================
def make_env(reward_overrides, seed):
    """Create a Monitor-wrapped environment."""
    params = {**DEFAULTS, **reward_overrides}
    grid_map = np.array(grid_maps.generate_main_with_subclusters_map(
        rows, cols, main_cluster_size_range, subcluster_config))
    free_cells = np.argwhere(grid_map == 0)
    random_indices = random.sample(range(len(free_cells)), 1)
    robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

    env = ChemicalCleanSensitivity(
        grid_map, num_robots=1, robot_positions=robot_positions,
        field_of_view=0, chem_fov=1, **params,
    )
    env = Monitor(env)
    return env


def train_single_run(reward_overrides, seed):
    """Train one PPO run, return list of (timestep, episode_reward)."""
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(reward_overrides, seed)

    model = PPO(
        "MultiInputPolicy", env, verbose=0,
        ent_coef=0.05, device="cuda", learning_rate=3e-4,
        seed=seed,
    )
    callback = EpisodeRewardLogger()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    return callback.episode_rewards


def interpolate_rewards(episode_data, total_timesteps, step):
    """Interpolate (timestep, reward) pairs onto a regular grid using step-function."""
    grid = np.arange(0, total_timesteps + 1, step)
    values = np.full_like(grid, dtype=np.float64, fill_value=np.nan)

    if not episode_data:
        return grid, values

    ts = np.array([d[0] for d in episode_data])
    rw = np.array([d[1] for d in episode_data])

    for idx, g in enumerate(grid):
        mask = ts <= g
        if mask.any():
            # mean of all episodes completed up to this timestep
            values[idx] = np.mean(rw[mask])

    return grid, values


# =====================================================================
#  PLOTTING
# =====================================================================
def plot_results(all_results, output_path):
    """
    all_results: dict[param_name][value] = list of 5 seed curves
                 each curve = (grid, values) from interpolate_rewards
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

    for idx, (param_name, value_dict) in enumerate(all_results.items()):
        ax = axes[idx]
        default_val = DEFAULTS[param_name]
        scale, offset = Y_TRANSFORMS[param_name]
        legend_map = LEGEND_LABELS[param_name]

        for c_idx, (val, seed_curves) in enumerate(sorted(value_dict.items())):
            # seed_curves is a list of (grid, values) — one per seed
            grids = [sc[0] for sc in seed_curves]
            vals_list = [sc[1] for sc in seed_curves]

            # Stack across seeds
            stacked = np.array(vals_list)               # shape (5, num_grid_points)
            median = np.nanmedian(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)

            # Apply y-axis transform
            median = scale * median + offset
            std = abs(scale) * std

            grid = grids[0]
            display_val = legend_map.get(val, val)
            is_default = val == default_val
            label = f"{display_val}" + (" (default)" if is_default else "")
            lw = 2.5 if is_default else 1.5
            ls = "-" if is_default else "--"

            ax.plot(grid, median, color=colors[c_idx], linewidth=lw,
                    linestyle=ls, label=label)
            ax.fill_between(grid, median - std, median + std,
                            color=colors[c_idx], alpha=0.15)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title(PARAM_LABELS[param_name])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Sensitivity Analysis — Mean Episode Reward vs Timestep\n"
                 "(median +/- std across 5 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    results_dir = "sensitivity_results"
    os.makedirs(results_dir, exist_ok=True)

    # Structure: all_results[param_name][value] = [(grid, values), ...] per seed
    all_results = {}

    total_runs = sum(len(vals) for vals in PARAM_VARIATIONS.values()) * len(SEEDS)
    run_count = 0
    start_time = time.time()

    for param_name, values in PARAM_VARIATIONS.items():
        all_results[param_name] = {}

        for val in values:
            all_results[param_name][val] = []
            overrides = {param_name: val}

            for seed in SEEDS:
                run_count += 1
                elapsed = time.time() - start_time
                eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0

                print(f"[{run_count}/{total_runs}] param={param_name}, "
                      f"value={val}, seed={seed}  "
                      f"(elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min)")

                episode_data = train_single_run(overrides, seed)
                grid, values_interp = interpolate_rewards(
                    episode_data, TOTAL_TIMESTEPS, INTERP_STEP)
                all_results[param_name][val].append((grid, values_interp))

    # ---- Plot ----
    plot_path = os.path.join(results_dir, "reward_vs_timestep.png")
    plot_results(all_results, plot_path)

    # ---- CSV summary ----
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("parameter,value,seed,final_mean_reward,total_episodes\n")

        for param_name, value_dict in all_results.items():
            for val, seed_curves in sorted(value_dict.items()):
                for s_idx, (grid, vals) in enumerate(seed_curves):
                    valid = vals[~np.isnan(vals)]
                    final_rew = valid[-1] if len(valid) > 0 else float("nan")
                    n_eps = int(np.sum(~np.isnan(vals)))
                    f.write(f"{param_name},{val},{SEEDS[s_idx]},{final_rew:.2f},{n_eps}\n")

    print(f"  CSV saved: {csv_path}")

    total_time = time.time() - start_time
    print(f"\nDone! Total time: {total_time/60:.1f} minutes")
