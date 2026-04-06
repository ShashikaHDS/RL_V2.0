#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame visualization extracted & refactored for your research paper figures.

• Matches your color palette
• Clean API and single display (no re-init each step)
• Multi-robot support + path overlays
• Toggle pause/step, adjustable speed
• Export high-res PNGs and record frames for GIF/MP4 (optional)

How to integrate with your env (sketch):

    viz = PygameGridViz(grid_shape=env.grid_map.shape, cell_size=20, margin=1)
    viz.set_grid(env.grid_map)

    obs = env.reset()
    recording = False  # set True to save frames

    running = True
    while running:
        # 1) choose actions ...
        obs, reward, done, info = env.step(actions)

        viz.update(
            known_map=obs["known_map"],
            spill_map=obs["spill_map"],
            robot_positions=obs["robot_positions"],
            paths=getattr(env, "paths", None)  # list[list[(y,x),(y,x)]] segments or None
        )
        viz.render(record_frame=recording)

        if viz.should_quit or done:
            break

    # Export a crisp figure for the paper
    viz.export_png("final_frame.png", upscale=4)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import os
import numpy as np
import pygame

# ------------------------ Colors (your palette) ------------------------
pygame.init()
bg_color = (255, 255, 255)
grid_color = pygame.Color("grey")
obs_color = (0, 255, 0)  # obstacles

normal_area_color = (255, 255, 0)   # visited/known area (exploring)
robot_color = (255, 0, 0)

spill_color = (165, 42, 42)         # uncleaned spill
spill_cleaned_color = (0, 128, 0)   # cleaned spill

Coord = Tuple[int, int]

@dataclass
class VizOptions:
    show_grid_lines: bool = True
    show_known_overlay: bool = True
    show_paths: bool = True
    show_hud: bool = True
    fps_limit: int = 60
    title: str = "Boustrophedon / RL Coverage (Pygame)"

class PygameGridViz:
    def __init__(self,
                 grid_shape: Tuple[int, int],
                 cell_size: int = 20,
                 margin: int = 1,
                 options: VizOptions | None = None):
        """
        grid_shape: (rows, cols)
        cell_size: pixel size per cell in the on-screen window
        margin: pixel gap between cells (grid lines drawn on top)
        """
        self.H, self.W = grid_shape
        self.cell = cell_size
        self.margin = margin
        self.options = options or VizOptions()

        # Surfaces / window
        self.screen_w = self.W * (self.cell + self.margin) + self.margin
        self.screen_h = self.H * (self.cell + self.margin) + self.margin + (28 if self.options.show_hud else 0)
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(self.options.title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 18)

        # State
        self.grid_map: Optional[np.ndarray] = None  # 0 free, 1 spill, 2 obstacle
        self.known_map = np.zeros((self.H, self.W), dtype=np.uint8)  # >0 means known/visited
        self.spill_map = np.zeros((self.H, self.W), dtype=np.uint8)  # 1 means cleaned/detected
        self.robot_positions: Optional[np.ndarray] = None            # (N,2) array of (y,x)
        self.paths: Optional[List[List[Tuple[Coord, Coord]]]] = None # list of segments per robot

        # UI state
        self.should_quit = False
        self.paused = False
        self.ms_per_step = 60  # for auto-play stepping visuals (if you call .tick_step())
        self._elapsed = 0

        # Recording
        self._frame_idx = 0
        self._record_dir = None

        # Pre-compute unique robot colors (first is your red)
        self.robot_colors = self._make_robot_colors()

    # -------------------- Public API --------------------
    def set_grid(self, grid_map: np.ndarray) -> None:
        assert grid_map.ndim == 2, "grid_map must be 2D"
        self.grid_map = grid_map.astype(int)
        self.H, self.W = self.grid_map.shape

    def update(self,
               known_map: Optional[np.ndarray] = None,
               spill_map: Optional[np.ndarray] = None,
               robot_positions: Optional[np.ndarray] = None,
               paths: Optional[List[List[Tuple[Coord, Coord]]]] = None) -> None:
        if known_map is not None:
            self.known_map = known_map.astype(np.uint8)
        if spill_map is not None:
            self.spill_map = spill_map.astype(np.uint8)
        if robot_positions is not None:
            self.robot_positions = np.array(robot_positions, dtype=int)
        if paths is not None:
            self.paths = paths

    def render(self, record_frame: bool = False) -> None:
        """Draw current state and flip the display. Optionally saves the frame."""
        self._handle_events()
        self._draw_scene()
        pygame.display.flip()
        if record_frame:
            self._save_frame()
        self.clock.tick(self.options.fps_limit)

    def tick_step(self) -> None:
        """If you want time-gated stepping (e.g., for demos), call each loop."""
        dt = self.clock.tick(self.options.fps_limit)
        self._elapsed += dt
        if not self.paused and self._elapsed >= self.ms_per_step:
            self._elapsed = 0  # you can hook a step trigger here if needed

    def export_png(self, path: str, upscale: int = 3) -> None:
        """Render off-screen at higher resolution for crisp figures."""
        assert self.grid_map is not None, "Call set_grid() first"
        big_cell = self.cell * upscale
        big_margin = self.margin * upscale
        h = self.H * (big_cell + big_margin) + big_margin
        w = self.W * (big_cell + big_margin) + big_margin
        surf = pygame.Surface((w, h))
        self._draw_grid(surf, big_cell, big_margin)
        pygame.image.save(surf, path)

    def start_recording(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._record_dir = directory
        self._frame_idx = 0

    def stop_recording(self) -> None:
        self._record_dir = None

    # -------------------- Internals --------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.should_quit = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    # If you hook env stepping to RIGHT, handle it outside using paused flag
                    pass
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.ms_per_step = max(10, self.ms_per_step - 10)
                elif event.key == pygame.K_MINUS:
                    self.ms_per_step = min(1000, self.ms_per_step + 10)

    def _draw_scene(self) -> None:
        # background outside grid
        self.screen.fill((30, 30, 30))
        # main grid
        self._draw_grid(self.screen, self.cell, self.margin)
        # HUD
        if self.options.show_hud:
            cov = self._coverage_stats()
            hud_text = (
                f"Known: {cov['known_cells']}/{cov['total_cells']}  "
                f"Spills cleaned: {cov['cleaned_spills']}  "
                f"Coverage: {cov['coverage_pct']:.1f}%  "
                f"{'PAUSED' if self.paused else ''}"
            )
            surf = self.font.render(hud_text, True, (240, 240, 240))
            self.screen.blit(surf, (10, self.screen_h - 22))

    def _draw_grid(self, surface: pygame.Surface, cell: int, margin: int) -> None:
        assert self.grid_map is not None, "grid_map not set"
        H, W = self.grid_map.shape

        # layer 1: base cells
        for y in range(H):
            for x in range(W):
                r = self._cell_rect(surface, x, y, cell, margin)
                val = int(self.grid_map[y, x])
                if val == 2:
                    color = obs_color
                elif val == 1:
                    # show cleaned vs uncleaned using spill_map (1 means cleaned)
                    color = spill_cleaned_color if self.spill_map[y, x] == 1 else spill_color
                else:
                    # free cell, overlay known area later
                    color = bg_color