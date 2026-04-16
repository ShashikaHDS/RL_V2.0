"""
Continue training the robust PPO model from the 2M checkpoint.
Saves checkpoints every 500k steps to the same directory.
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

# Same rewards as original training
REWARDS = {
    "clean_spill": 50,
    "consecutive_bonus": 150,
    "revisit_penalty": -10,
    "explore_penalty": -10,
    "obstacle_penalty": -5,
}

# Map config
rows, cols = 20, 20
main_cluster_size_range = (30, 60)
subcluster_config = {
    "num_range": (4, 8),
    "size_range": (5, 15),
    "distance_from_main": (9, 10),
    "distance_between_subclusters": 8,
}

grid_maps = MapGen()
grid_map = np.array(grid_maps.generate_main_with_subclusters_map(
    rows, cols, main_cluster_size_range, subcluster_config))

free_cells = np.argwhere(grid_map == 0)
random_indices = random.sample(range(len(free_cells)), 1)
robot_positions = np.array([tuple(free_cells[i]) for i in random_indices])

env = ChemicalClean(grid_map, num_robots=1, robot_positions=robot_positions,
                    field_of_view=0, chem_fov=1, rewards=REWARDS)

# Continue from existing checkpoint
RESUME_FROM = "models/robust_1775803416/2000000.zip"
models_dir = "models/robust_1775803416/"
logdir = "logs/robust_continued/"
os.makedirs(logdir, exist_ok=True)

print(f"Resuming from: {RESUME_FROM}")
model = PPO.load(RESUME_FROM, env=env, device="cuda", tensorboard_log=logdir)

# Continue training
ADDITIONAL_TIMESTEPS = 8_000_000
SAVE_EVERY = 500_000

print(f"Training for {ADDITIONAL_TIMESTEPS:,} more timesteps...")

start_time = time.time()
steps_done = 2_000_000  # already done

while steps_done < (2_000_000 + ADDITIONAL_TIMESTEPS):
    model.learn(total_timesteps=SAVE_EVERY, log_interval=4, reset_num_timesteps=False)
    steps_done += SAVE_EVERY
    model.save(os.path.join(models_dir, f"{steps_done}"))
    elapsed = (time.time() - start_time) / 60
    print(f"  Checkpoint: {steps_done:,} steps | {elapsed:.1f} min elapsed")

total_time = (time.time() - start_time) / 60
print(f"\nDone! {total_time:.1f} minutes")
