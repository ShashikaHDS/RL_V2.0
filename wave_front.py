"""
Wavefront (BFS) Online Area Coverage — Sensor radius = 1
========================================================

Map values (input truth):
- 0 = free
- 1 = (ignored) previously "spill", now treated as FREE
- 2 = obstacle (blocked)

Planner is ONLINE: only moves through cells revealed by a radius-1 sensor.
Visualization can show the full ground-truth from the start (display-only).

Controls:
- SPACE : pause/resume
- N     : single step
- +/-   : speed up / slow down (cells per second)
- R     : reset (same map)
- S     : screenshot
- ESC/Q : quit
"""
from __future__ import annotations
import pygame
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

# =====================
# CONFIG
# =====================
CELL = 16                   # pixels per grid cell
FPS = 60
MOVES_PER_SECOND = 30       # animation speed (cells/sec)
START = (1, 1)              # (x, y) must be traversable (0 or 1 in truth)
SENSOR_RADIUS = 1           # Chebyshev radius
STEP_BUDGET: Optional[int] = None  # e.g., 800; None = unlimited

# Show full map from the start (visual only; planner still uses sensed knowledge)
SHOW_FULL_TRUTH_MAP = True

# Map: 0=free, 1=treated-as-free, 2=obstacle
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
    'bg': (255, 255, 255),
    'ob': (50, 50, 55),
    'free': (255, 255, 255),
    'text': (20, 20, 20),
    'path': (0, 0, 0),
}

# =====================
# World / sensing
# =====================
def cheby(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

class GridWorld:
    def __init__(self, truth: np.ndarray, start: Tuple[int,int]):
        self.truth = truth.copy()           # HxW with 0/1 (free) and 2 (obstacle)
        self.H, self.W = self.truth.shape
        sx, sy = start
        if not (0 <= sx < self.W and 0 <= sy < self.H):
            raise ValueError("START out of bounds.")
        if self.truth[sy, sx] == 2:
            raise ValueError("START is inside an obstacle.")

        # known map: -1 unknown, 0 free, 2 obstacle  (note: truth==1 is treated as FREE -> 0)
        self.known = np.full_like(self.truth, -1)
        self.visited = np.zeros_like(self.truth, dtype=bool)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def reveal(self, center: Tuple[int,int], radius: int=SENSOR_RADIUS):
        cx, cy = center
        for y in range(cy-radius, cy+radius+1):
            for x in range(cx-radius, cx+radius+1):
                if self.in_bounds(x,y) and cheby((x,y),(cx,cy)) <= radius:
                    self.known[y, x] = 2 if self.truth[y, x] == 2 else 0  # 1 -> 0 (free)

    def is_known_traversable(self, x: int, y: int) -> bool:
        return self.in_bounds(x,y) and (self.known[y, x] == 0)

    def mark_touch(self, x: int, y: int):
        self.visited[y, x] = True

    def frontier_mask(self) -> np.ndarray:
        """Known free cells touching any unknown neighbor (4-neighborhood)."""
        fm = np.zeros_like(self.known, dtype=bool)
        for y in range(self.H):
            for x in range(self.W):
                if self.known[y, x] == 0:
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if self.in_bounds(nx, ny) and self.known[ny, nx] == -1:
                            fm[y, x] = True
                            break
        return fm

# =====================
# BFS helpers (on known free)
# =====================
def bfs_to_target(world: GridWorld, start: Tuple[int,int], target_pred) -> List[Tuple[int,int]]:
    """
    4-neighbor BFS on known-free cells until target_pred(x,y) is True.
    Returns the path (start..goal). Empty if not found.
    """
    if target_pred(*start):
        return [start]
    H, W = world.H, world.W
    Q = deque([start])
    par = {start: None}
    while Q:
        x, y = Q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H:
                if world.is_known_traversable(nx, ny) and (nx,ny) not in par:
                    par[(nx,ny)] = (x,y)
                    if target_pred(nx, ny):
                        # reconstruct
                        path = [(nx,ny)]
                        cur = (x,y)
                        while cur is not None:
                            path.append(cur)
                            cur = par[cur]
                        path.reverse()
                        return path
                    Q.append((nx,ny))
    return []

def nearest_frontier_path(world: GridWorld, start: Tuple[int,int]) -> List[Tuple[int,int]]:
    fm = world.frontier_mask()
    def is_frontier(x: int, y: int) -> bool:
        return fm[y, x]
    return bfs_to_target(world, start, is_frontier)

def nearest_unvisited_known_path(world: GridWorld, start: Tuple[int,int]) -> List[Tuple[int,int]]:
    def is_unvisited(x: int, y: int) -> bool:
        return world.is_known_traversable(x, y) and not world.visited[y, x]
    return bfs_to_target(world, start, is_unvisited)

# =====================
# Online wavefront planner (area coverage only)
# =====================
class WavefrontOnline:
    def __init__(self, world: GridWorld, start: Tuple[int,int]):
        self.world = world
        self.pos = start
        self.queue: deque[Tuple[int,int]] = deque()
        self.route: List[Tuple[int,int]] = [start]
        self.steps_taken = 0
        self.done = False

        # initial sense + mark
        self.world.reveal(self.pos, SENSOR_RADIUS)
        self.world.mark_touch(*self.pos)

    def _enqueue(self, pts: List[Tuple[int,int]]):
        for p in pts:
            self.queue.append(p)

    def _tick_move(self) -> bool:
        if not self.queue:
            return False
        nx, ny = self.queue.popleft()
        if not self.world.is_known_traversable(nx, ny):
            return False
        self.pos = (nx, ny)
        self.world.reveal(self.pos, SENSOR_RADIUS)
        self.world.mark_touch(nx, ny)
        self.route.append(self.pos)
        self.steps_taken += 1
        return True

    def _plan_next(self) -> bool:
        """Wavefront priority: nearest frontier; else nearest unvisited known; else nothing."""
        # 1) go to nearest frontier to expand the known region
        path = nearest_frontier_path(self.world, self.pos)
        if len(path) > 1:
            self._enqueue(path[1:])  # skip current
            return True

        # 2) if no frontier left, cover any remaining known-but-unvisited cells
        path = nearest_unvisited_known_path(self.world, self.pos)
        if len(path) > 1:
            self._enqueue(path[1:])
            return True

        # 3) done
        return False

    def step(self) -> bool:
        if self.done:
            return False

        # Move if we have queued steps
        if self._tick_move():
            return True

        # Otherwise plan something new (based on current knowledge)
        if self._plan_next():
            return True

        self.done = True
        return False

# =====================
# Visualization
# =====================
class Viewer:
    def __init__(self, truth: np.ndarray):
        pygame.init()
        self.truth = truth
        self.H, self.W = truth.shape

        self.world = GridWorld(truth, START)
        self.world.reveal(START, SENSOR_RADIUS)

        self.alg = WavefrontOnline(self.world, START)

        self.w_px = self.W*CELL
        self.h_px = self.H*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Wavefront (BFS) Online Area Coverage")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND
        self.budget = STEP_BUDGET or 10**12

    def draw_world(self):
        # Draw cells (no grid lines). If SHOW_FULL_TRUTH_MAP, draw truth map from the start.
        for y in range(self.H):
            for x in range(self.W):
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                if SHOW_FULL_TRUTH_MAP:
                    v = self.truth[y, x]
                    if v == 2:
                        pygame.draw.rect(self.screen, COLORS['ob'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['free'], rect)
                else:
                    v = self.world.known[y, x]
                    if v == 2:
                        pygame.draw.rect(self.screen, COLORS['ob'], rect)
                    elif v == 0:
                        pygame.draw.rect(self.screen, COLORS['free'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['free'], rect)  # unknown shown as white

        # Path polyline (no robot glyph)
        if len(self.alg.route) >= 2:
            pts = [(x*CELL + CELL//2, y*CELL + CELL//2) for (x,y) in self.alg.route]
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

    def draw_status(self):
        bar = pygame.Rect(0, self.H*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)

        visited = int(self.world.visited.sum())
        known_free = int(np.sum(self.world.known == 0))
        total_free_truth = int(np.sum(self.truth != 2))  # everything non-obstacle is free
        status = "DONE" if self.alg.done else "RUNNING"
        budget_txt = "∞" if STEP_BUDGET is None else str(STEP_BUDGET)

        txt = (
            f"steps {self.alg.steps_taken} (budget {budget_txt}) | "
            f"visited {visited} | known_free {known_free} | "
            f"total_free_truth {total_free_truth} | status {status}"
        )
        self.screen.blit(self.font.render(txt, True, COLORS['text']), (8, self.H*CELL + 10))

    def loop(self):
        running = True
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
                    elif e.key == pygame.K_n:
                        self.alg.step()
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.step_interval = max(1/240, self.step_interval - 0.01)
                    elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.step_interval = min(1/1, self.step_interval + 0.01)
                    elif e.key == pygame.K_s:
                        fname = f"wavefront_area_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.world = GridWorld(self.truth, START)
                        self.world.reveal(START, SENSOR_RADIUS)
                        self.alg = WavefrontOnline(self.world, START)
                        self.accumulator = 0.0
                        self.paused = False

            if not self.paused and not self.alg.done:
                self.accumulator += dt
                while self.accumulator >= self.step_interval:
                    if self.alg.steps_taken >= self.budget:
                        self.alg.done = True
                        break
                    self.alg.step()
                    self.accumulator -= self.step_interval

            self.screen.fill(COLORS['bg'])
            self.draw_world()
            self.draw_status()
            pygame.display.flip()
        pygame.quit()

# =====================
# Main
# =====================
def main():
    truth = CUSTOM_MAP_ARRAY.copy() if USE_CUSTOM_MAP else np.zeros((40, 60), dtype=int)
    viewer = Viewer(truth)
    viewer.loop()

if __name__ == "__main__":
    main()
