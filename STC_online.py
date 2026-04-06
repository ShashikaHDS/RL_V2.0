"""
Online Spanning Tree Coverage (STC) — Pygame Demo
-------------------------------------------------
Now supports: (a) **custom fixed maps** and (b) **step budget stop**.

How to use a custom map:
1) Paste your `grid_map` (0=free, 1=obstacle) into the CONFIG section below.
2) Set `USE_CUSTOM_MAP = True` and `MAX_STEPS = 340` (already set).
3) Run: `python online_stc_pygame.py`

Controls
- R : reset (keeps same custom map if provided)
- SPACE : pause/resume
- ESC / Q : quit

Tested with: Python 3.9+, pygame >= 2.1, numpy
"""
from __future__ import annotations
import pygame
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

# =====================
# CONFIG
# =====================
CELL = 16                 # pixels per grid cell (visual scale)
FPS = 60

# Robot & sensing
START = (1, 1)            # robot start cell (col, row)
SENSOR_RADIUS = 3         # reveal radius in cells (Chebyshev distance)
MOVES_PER_SECOND = 16     # how fast robot steps between cells
MAX_STEPS = 340           # <-- step budget (will stop when reached)

# Random map fallback (ignored when using custom map)
DEFAULT_W, DEFAULT_H = 40, 28
OBSTACLE_PROB = 0.18
SEED = None               # set to an int for reproducibility

# === Custom map (paste yours here) ===
USE_CUSTOM_MAP = True
CUSTOM_MAP_ARRAY = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# Colors
COLORS = {
    'bg': (245, 246, 250),
    'grid': (220, 223, 230),
    'ob': (40, 40, 40),
    'free_unknown': (230, 235, 245),
    'free_known': (210, 230, 255),
    'visited': (170, 220, 255),
    'frontier': (140, 200, 255),
    'robot': (255, 120, 120),
    'edge': (120, 170, 220),
    'text': (20, 20, 20),
}

# 4-neighborhood moves (E, S, W, N) — prefer to keep heading when possible
DIRS = [(1,0),(0,1),(-1,0),(0,-1)]

# =====================
# Utilities
# =====================
Coord = Tuple[int, int]

def in_bounds(c: Coord, w: int, h: int) -> bool:
    x, y = c
    return 0 <= x < w and 0 <= y < h

# Chebyshev distance for a square sensor footprint
cheby = lambda a,b: max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# =====================
# Map Representations
# =====================
class GridMap:
    """Ground-truth map (unknown to robot initially) and robot's knowledge map."""
    def __init__(self, w: int, h: int, p_ob: float=0.0, seed: Optional[int]=None, truth: Optional[np.ndarray]=None):
        if truth is not None:
            # Use provided truth map directly
            assert truth.ndim == 2, "truth must be a 2D numpy array"
            self.w, self.h = truth.shape[1], truth.shape[0]
            # store as [x][y]
            self.truth = [[int(truth[y, x]) for y in range(self.h)] for x in range(self.w)]
        else:
            if seed is not None:
                random.seed(seed)
            self.w, self.h = w, h
            self.truth = [[0]*h for _ in range(w)]    # 0 free, 1 obstacle
            for x in range(w):
                for y in range(h):
                    # Outer border obstacles to keep robot inside
                    if x==0 or y==0 or x==w-1 or y==h-1:
                        self.truth[x][y] = 1
                    else:
                        self.truth[x][y] = 1 if random.random() < p_ob else 0
        # Ensure start is free in truth
        sx, sy = START
        if in_bounds((sx,sy), self.w, self.h):
            self.truth[sx][sy] = 0

        # Knowledge map (what the robot has discovered)
        self.known_state = [[-1]*self.h for _ in range(self.w)]  # -1 unknown, 0 free, 1 obstacle
        self.visited = [[False]*self.h for _ in range(self.w)]   # visited free cells

    def reveal_with_sensor(self, center: Coord, radius: int):
        """Reveal cells within sensor radius in the knowledge map."""
        cx, cy = center
        for x in range(cx-radius, cx+radius+1):
            for y in range(cy-radius, cy+radius+1):
                if in_bounds((x,y), self.w, self.h) and cheby((x,y),(cx,cy)) <= radius:
                    self.known_state[x][y] = self.truth[x][y]

    def is_free_truth(self, c: Coord) -> bool:
        x, y = c
        return in_bounds(c, self.w, self.h) and self.truth[x][y] == 0

    def is_free_known(self, c: Coord) -> bool:
        x, y = c
        return in_bounds(c, self.w, self.h) and self.known_state[x][y] == 0

# =====================
# Online STC Robot
# =====================
class OnlineSTCRobot:
    def __init__(self, gmap: GridMap, start: Coord=(1,1)):
        self.map = gmap
        self.pos = start
        self.heading_idx = 0  # index into DIRS
        self.stack: List[Coord] = [start]  # DFS stack over discovered free cells
        self.parent = {start: None}        # tree parent pointers
        self.edges = set()                 # for drawing the tree
        self.steps_taken = 0
        self.done_reason = None
        # Initialize sensing at start
        if in_bounds(self.pos, self.map.w, self.map.h):
            self.map.reveal_with_sensor(self.pos, SENSOR_RADIUS)
            self.map.visited[self.pos[0]][self.pos[1]] = True

    def neighbors_heading_order(self, c: Coord) -> List[Coord]:
        """Prefer continuing same heading, then right, left, reverse.
        Returns neighbor list filtered to in-bounds.
        """
        # Order indices: forward, right, left, back
        forward = self.heading_idx
        right = (self.heading_idx + 1) % 4
        left  = (self.heading_idx + 3) % 4
        back  = (self.heading_idx + 2) % 4
        order = [forward, right, left, back]
        res = []
        for idx in order:
            dx, dy = DIRS[idx]
            nx, ny = c[0]+dx, c[1]+dy
            if in_bounds((nx,ny), self.map.w, self.map.h):
                res.append(((nx,ny), idx))
        return res

    def step(self) -> bool:
        """Perform one online STC step. Returns False if coverage is complete or budget hit."""
        if self.steps_taken >= MAX_STEPS:
            self.done_reason = "BUDGET"
            return False
        if not self.stack:
            self.done_reason = "COMPLETE"
            return False
        c = self.stack[-1]

        # Sense at current position
        self.map.reveal_with_sensor(c, SENSOR_RADIUS)

        # Try to expand to a new discovered FREE & not visited neighbor
        moved = False
        for (n, idx) in self.neighbors_heading_order(c):
            if self.map.is_free_known(n) and not self.map.visited[n[0]][n[1]]:
                # Move to n, add edge to tree
                self.heading_idx = idx
                self.edges.add((min(c,n), max(c,n)))
                self.pos = n
                self.map.visited[n[0]][n[1]] = True
                self.parent[n] = c
                self.stack.append(n)
                self.steps_taken += 1
                moved = True
                break

        if not moved:
            # No unvisited neighbor in known free space → backtrack along the tree
            self.stack.pop()
            if self.stack:
                prev = self.stack[-1]
                # Update heading to face the movement direction
                dx, dy = prev[0]-c[0], prev[1]-c[1]
                if (dx,dy) in DIRS:
                    self.heading_idx = DIRS.index((dx,dy))
                self.pos = prev
                self.steps_taken += 1
            else:
                # Finished: nowhere to backtrack
                self.done_reason = "COMPLETE"
                return False
        return True

# =====================
# Visualization
# =====================
class Viewer:
    def __init__(self, gmap: GridMap, robot: OnlineSTCRobot):
        pygame.init()
        self.gmap = gmap
        self.robot = robot
        self.w_px = self.gmap.w*CELL
        self.h_px = self.gmap.h*CELL + 40  # status bar
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Online STC — Unknown Map Coverage")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND

    def draw_grid(self):
        for x in range(self.gmap.w):
            for y in range(self.gmap.h):
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                st = self.gmap.known_state[x][y]
                if st == -1:
                    pygame.draw.rect(self.screen, COLORS['free_unknown'], rect)
                elif st == 1:
                    pygame.draw.rect(self.screen, COLORS['ob'], rect)
                else:  # known free
                    if self.gmap.visited[x][y]:
                        pygame.draw.rect(self.screen, COLORS['visited'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['free_known'], rect)

        # Draw grid lines
        for x in range(self.gmap.w+1):
            pygame.draw.line(self.screen, COLORS['grid'], (x*CELL, 0), (x*CELL, self.gmap.h*CELL))
        for y in range(self.gmap.h+1):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y*CELL), (self.gmap.w*CELL, y*CELL))

        # Draw spanning tree edges
        for (a,b) in self.robot.edges:
            ax = a[0]*CELL + CELL//2
            ay = a[1]*CELL + CELL//2
            bx = b[0]*CELL + CELL//2
            by = b[1]*CELL + CELL//2
            pygame.draw.line(self.screen, COLORS['edge'], (ax,ay), (bx,by), 2)

        # Draw robot
        rx = self.robot.pos[0]*CELL + CELL//2
        ry = self.robot.pos[1]*CELL + CELL//2
        pygame.draw.circle(self.screen, COLORS['robot'], (rx, ry), CELL//3)

    def draw_status(self, done: bool):
        bar = pygame.Rect(0, self.gmap.h*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, (255,255,255), bar)
        pygame.draw.line(self.screen, COLORS['grid'], (0, self.gmap.h*CELL), (self.w_px, self.gmap.h*CELL))
        known_free = sum(1 for x in range(self.gmap.w) for y in range(self.gmap.h)
                         if self.gmap.known_state[x][y] == 0)
        visited = sum(1 for x in range(self.gmap.w) for y in range(self.gmap.h)
                      if self.gmap.visited[x][y])
        status = self.robot.done_reason if done else 'RUNNING'
        txt = (
            f"steps: {self.robot.steps_taken}/{MAX_STEPS}  "
            f"known_free: {known_free}  visited: {visited}  "
            f"paused: {self.paused}  status: {status}"
        )
        surf = self.font.render(txt, True, COLORS['text'])
        self.screen.blit(surf, (8, self.gmap.h*CELL + 10))

    def loop(self):
        running = True
        done = False
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_r:
                        # reset world & robot (keep same map)
                        self.reset_world()
                        done = False

            self.screen.fill(COLORS['bg'])
            self.draw_grid()

            if not self.paused and not done:
                self.accumulator += dt
                while self.accumulator >= self.step_interval:
                    cont = self.robot.step()
                    if not cont:
                        done = True
                        break
                    self.accumulator -= self.step_interval

            self.draw_status(done)
            pygame.display.flip()
        pygame.quit()

    def reset_world(self):
        if USE_CUSTOM_MAP:
            truth = CUSTOM_MAP_ARRAY
            gmap = GridMap(0, 0, truth=truth)
        else:
            gmap = GridMap(DEFAULT_W, DEFAULT_H, OBSTACLE_PROB, seed=SEED)
        robot = OnlineSTCRobot(gmap, start=START)
        self.gmap = gmap
        self.robot = robot
        self.w_px = self.gmap.w*CELL
        self.h_px = self.gmap.h*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))

# =====================
# Main
# =====================

def generate_world() -> Tuple[GridMap, OnlineSTCRobot]:
    if USE_CUSTOM_MAP:
        truth = CUSTOM_MAP_ARRAY
        g = GridMap(0, 0, truth=truth)
        r = OnlineSTCRobot(g, start=START)
        return g, r
    else:
        # Random world with at least one free neighbor around start
        attempts = 0
        while True:
            attempts += 1
            g = GridMap(DEFAULT_W, DEFAULT_H, OBSTACLE_PROB, seed=SEED)
            r = OnlineSTCRobot(g, start=START)
            sx, sy = START
            free_neighbors = 0
            for dx,dy in DIRS:
                if g.is_free_truth((sx+dx, sy+dy)):
                    free_neighbors += 1
            if free_neighbors > 0 or attempts > 50:
                return g, r


def main():
    g, r = generate_world()
    viewer = Viewer(g, r)
    viewer.loop()

if __name__ == "__main__":
    main()
