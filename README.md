# RL Chemical Spill Cleaning

Multi-robot reinforcement learning system for autonomous chemical spill cleaning on grid-based environments using PPO (Proximal Policy Optimization) with Stable-Baselines3.

## Overview

A robot navigates a 2D grid containing chemical spills, obstacles, and free cells. The goal is to clean as many spill cells as possible within a step budget. The environment uses a custom Gym interface with dictionary observations (known map, spill map, robot positions).

## Reward Structure

| Event | Reward |
|-------|--------|
| Cleaning a spill (first time) | +50 |
| Consecutive spill cleaning | +150 |
| Moving to already-cleaned cell | -10 |
| Moving to non-spill cell | -10 |
| Hitting an obstacle | -5 |

## Key Files

- `chemical_v3.py` — Main environment with pygame visualization
- `chemical_v3_no_visual.py` — Headless environment for faster training
- `sensitivity_analysis.py` — One-at-a-time reward sensitivity analysis (5 values per parameter, 5 seeds each, plots median +/- std of episode reward vs timestep)
- `reward_ablation_study.py` — Reward ablation study
- `map_gen_v8.py` — Procedural grid map generator with clustered spills and obstacles

## Requirements

- Python 3.9+
- stable-baselines3
- gymnasium
- pygame
- numpy
- matplotlib

## Usage

### Training
```bash
python chemical_v3.py
```

### Sensitivity Analysis
```bash
python sensitivity_analysis.py
```
Outputs plots and CSV to `sensitivity_results/`.

### Prediction / Evaluation
```bash
python chemical_predict.py
```

## Map Generation

Maps are procedurally generated using `MapGen` with configurable parameters:
- Grid size (default 20x20)
- Main cluster size range
- Subcluster count, size, and spacing

Cell types: `0` = free, `1` = chemical spill, `2` = obstacle
