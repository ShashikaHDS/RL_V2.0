"""
Train a robust PPO model for ~13M timesteps (matching paper results).
Headless, no pygame. Saves checkpoints every 500k steps.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import random
import time
import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from map_gen_v8 import MapGen
from stable_baselines3 import PPO
from chemical_v3_no_visual import ChemicalClean

# Reward structure (matches chemical_v3.py - large magnitudes for stronger signal)
REWARDS = {
    "clean_spill": 50,
    "consecutive_bonus": 150,
    "revisit_penalty": -10,
    "explore_penalty": -10,
    "obstacle_penalty": -5,
}

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
grid_map = grid_maps.generate_main_with_subclusters_map(
    rows, cols, main_cluster_size_range, subcluster_config
)
grid_map = np.array(grid_map)

free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), 1)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = ChemicalClean(grid_map, num_robots=1, robot_positions=robot_positions,
                    field_of_view=0, chem_fov=1, rewards=REWARDS)

# ---- Directories ----
timestamp = int(time.time())
models_dir = f"models/robust_{timestamp}/"
logdir = f"logs/robust_{timestamp}/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# ---- Training ----
TOTAL_TIMESTEPS = 3_000_000
SAVE_EVERY = 200_000

print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
print(f"Saving checkpoints every {SAVE_EVERY:,} steps to {models_dir}")
print(f"Logs: {logdir}")

# Larger MLP for better representation learning
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

model = PPO("MultiInputPolicy", env, verbose=1,
            tensorboard_log=logdir, ent_coef=0.05,
            device="cuda", learning_rate=3e-4,
            n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=policy_kwargs)

start_time = time.time()
steps_done = 0

while steps_done < TOTAL_TIMESTEPS:
    model.learn(total_timesteps=SAVE_EVERY, log_interval=4, reset_num_timesteps=False)
    steps_done += SAVE_EVERY
    model.save(os.path.join(models_dir, f"{steps_done}"))
    elapsed = (time.time() - start_time) / 60
    print(f"  Checkpoint: {steps_done:,} steps | {elapsed:.1f} min elapsed")

total_time = (time.time() - start_time) / 60
print(f"\nTraining complete! {total_time:.1f} minutes total")
print(f"Final model: {models_dir}{TOTAL_TIMESTEPS}")
