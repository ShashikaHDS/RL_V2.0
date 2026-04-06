"""
Online Spanning Tree Coverage (STC) with Pygame

Visualization:
- White background
- Spill colors: red (uncleaned), green (cleaned)
- Robot path: black
- Robot: blue cell (no tail/arrow)
- No grid lines displayed
- Lidar/sensing not drawn (used internally only when full reveal is off)

Encoding:
- 0 = free
- 1 = chemical spill (TRAVERSABLE to clean)
- 2 = obstacle (BLOCKED)

Controls:
- SPACE : pause/resume
- N     : single step
- +/-   : speed up / slow down (cells per second)
- R     : reset (keeps the same custom map if enabled)
- T     : toggle STC tree edges on/off
- S     : screenshot
- ESC/Q : quit
"""
from __future__ import annotations
import pygame
import random
import numpy as np
from typing import List, Tuple, Optional

# =====================
# CONFIG
# =====================
CELL = 16                 # pixels per grid cell (visual scale)
FPS = 60

# Robot & sensing
START = (1, 1)            # robot start cell (x=col, y=row)
SENSOR_RADIUS = 3         # used for revealing, not drawn
MOVES_PER_SECOND = 18     # cells per second
MAX_STEPS = 300           # step budget

# Reveal the entire map (spills/obstacles) at t=0
REVEAL_FULL_MAP_AT_START = True

# Random map fallback (ignored if USE_CUSTOM_MAP=True)
DEFAULT_W, DEFAULT_H = 40, 28
P_OBS = 0.18
P_SPILL = 0.08
SEED = None

# === Custom map (paste yours here) ===
USE_CUSTOM_MAP = True
CUSTOM_MAP_ARRAY = np.array([
    # 0=free, 1=SPILL, 2=OBSTACLE
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

# Colors (per request)
COLORS = {
    'bg': (255, 255, 255),        # white background
    'ob': (50, 50, 55),           # obstacle (dark grey) so black path is distinct
    'spill': (255, 0, 0),         # RED (uncleaned)
    'spill_cleaned': (0, 180, 0), # GREEN (cleaned)
    'free': (255, 255, 255),      # free/unknown cells drawn white
    'robot': (0, 120, 255),       # BLUE robot cell
    'edge': (160, 200, 255),      # (optional) tree edges
    'text': (20, 20, 20),
    'path': (0, 0, 0),            # BLACK path line
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

cheby = lambda a,b: max(abs(a[0]-b[0]), abs(a[1]-b[1]))  # for sensing

# =====================
# Map Representations
# =====================
class GridMap:
    """Ground-truth map (unknown to robot initially) and robot's knowledge map.
       Encoding: 0=free, 1=spill (traversable), 2=obstacle (blocked)."""
    def __init__(self, w: int, h: int, p_obs: float=0.0, p_spill: float=0.0,
                 seed: Optional[int]=None, truth: Optional[np.ndarray]=None):
        if truth is not None:
            assert truth.ndim == 2, "truth must be a 2D numpy array"
            self.w, self.h = truth.shape[1], truth.shape[0]
            self.truth = [[int(truth[y, x]) for y in range(self.h)] for x in range(self.w)]
        else:
            if seed is not None:
                random.seed(seed)
            self.w, self.h = w, h
            self.truth = [[0]*h for _ in range(w)]
            for x in range(w):
                for y in range(h):
                    if x == 0 or y == 0 or x == w-1 or y == h-1:
                        self.truth[x][y] = 2
                    else:
                        r = random.random()
                        if r < p_obs:
                            self.truth[x][y] = 2
                        elif r < p_obs + p_spill:
                            self.truth[x][y] = 1
                        else:
                            self.truth[x][y] = 0

        # Ensure start isn't an obstacle
        sx, sy = START
        if in_bounds((sx,sy), self.w, self.h) and self.truth[sx][sy] == 2:
            self.truth[sx][sy] = 0

        # Knowledge map:
        # -1 unknown, 0 free, 1 spill, 2 obstacle
        if REVEAL_FULL_MAP_AT_START:
            # Deep copy the ground truth so everything is known from t=0
            self.known_state = [col[:] for col in self.truth]
        else:
            self.known_state = [[-1]*self.h for _ in range(self.w)]

        self.visited = [[False]*self.h for _ in range(self.w)]

    def reveal_with_sensor(self, center: Coord, radius: int):
        """Reveal cells within sensor radius in the knowledge map (not drawn)."""
        cx, cy = center
        for x in range(cx-radius, cx+radius+1):
            for y in range(cy-radius, cy+radius+1):
                if in_bounds((x,y), self.w, self.h) and cheby((x,y),(cx,cy)) <= radius:
                    self.known_state[x][y] = self.truth[x][y]

    def is_traversable_known(self, c: Coord) -> bool:
        x, y = c
        return in_bounds(c, self.w, self.h) and self.known_state[x][y] in (0, 1)

# =====================
# Online STC Robot
# =====================
class OnlineSTCRobot:
    def __init__(self, gmap: GridMap, start: Coord=(1,1)):
        self.map = gmap
        self.pos = start
        self.heading_idx = 0
        self.stack: List[Coord] = [start]
        self.parent = {start: None}
        self.edges = set()
        self.steps_taken = 0
        self.done_reason = None

        # Spills
        self.cleaned_spills = 0
        self.cleaned_mask = [[False]*self.map.h for _ in range(self.map.w)]

        # Path trace (for black path line)
        self.trace: List[Coord] = [start]

        # Initialize sensing & state
        if in_bounds(self.pos, self.map.w, self.map.h):
            self._sense(self.pos)
            self.map.visited[self.pos[0]][self.pos[1]] = True
            self._try_clean(self.pos)

    def _sense(self, c: Coord):
        if not REVEAL_FULL_MAP_AT_START:
            self.map.reveal_with_sensor(c, SENSOR_RADIUS)

    def _try_clean(self, c: Coord):
        x, y = c
        if self.map.known_state[x][y] == 1 and not self.cleaned_mask[x][y]:
            self.cleaned_mask[x][y] = True
            self.cleaned_spills += 1

    def neighbors_heading_order(self, c: Coord) -> List[Coord]:
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
        if self.steps_taken >= MAX_STEPS:
            self.done_reason = "BUDGET"
            return False
        if not self.stack:
            self.done_reason = "COMPLETE"
            return False

        c = self.stack[-1]
        self._sense(c)

        # advance if any neighbor is known traversable and unvisited
        for (n, idx) in self.neighbors_heading_order(c):
            if self.map.is_traversable_known(n) and not self.map.visited[n[0]][n[1]]:
                self.heading_idx = idx
                self.edges.add((min(c,n), max(c,n)))
                self.pos = n
                self.map.visited[n[0]][n[1]] = True
                self.parent[n] = c
                self.stack.append(n)
                self._try_clean(n)
                self.steps_taken += 1
                self.trace.append(n)     # add to path
                return True

        # otherwise backtrack
        self.stack.pop()
        if self.stack:
            prev = self.stack[-1]
            dx, dy = prev[0]-c[0], prev[1]-c[1]
            if (dx,dy) in DIRS:
                self.heading_idx = DIRS.index((dx,dy))
            self.pos = prev
            self.steps_taken += 1
            self.trace.append(prev)      # backtrack path is still part of travel
            return True

        self.done_reason = "COMPLETE"
        return False

# =====================
# Visualization
# =====================
class Viewer:
    def __init__(self, gmap: GridMap, robot: OnlineSTCRobot):
        pygame.init()
        self.gmap = gmap
        self.robot = robot
        self.w_px = self.gmap.w*CELL
        self.h_px = self.gmap.h*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Online STC — Coverage")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND

        self.show_tree = True  # toggle STC edges (not grid lines)

    def draw_grid(self):
        # base cells (no grid lines at all)
        for x in range(self.gmap.w):
            for y in range(self.gmap.h):
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                st = self.gmap.known_state[x][y]   # 0 free, 1 spill, 2 obstacle (no -1 when full reveal)
                if st == 0:
                    pygame.draw.rect(self.screen, COLORS['free'], rect)
                elif st == 2:
                    pygame.draw.rect(self.screen, COLORS['ob'], rect)
                elif st == 1:
                    if self.robot.cleaned_mask[x][y]:
                        pygame.draw.rect(self.screen, COLORS['spill_cleaned'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['spill'], rect)
                else:
                    # unknown (only possible if REVEAL_FULL_MAP_AT_START=False)
                    pygame.draw.rect(self.screen, COLORS['free'], rect)

        # optional spanning tree edges (not grid lines)
        if self.show_tree:
            for (a,b) in self.robot.edges:
                ax = a[0]*CELL + CELL//2
                ay = a[1]*CELL + CELL//2
                bx = b[0]*CELL + CELL//2
                by = b[1]*CELL + CELL//2
                pygame.draw.line(self.screen, COLORS['edge'], (ax,ay), (bx,by), 2)

        # robot path (black polyline)
        if len(self.robot.trace) >= 2:
            pts = [(cx*CELL + CELL//2, cy*CELL + CELL//2) for (cx, cy) in self.robot.trace]
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

        # robot as a blue cell
        rx, ry = self.robot.pos
        pygame.draw.rect(self.screen, COLORS['robot'], pygame.Rect(rx*CELL, ry*CELL, CELL, CELL))

    def draw_status(self, done: bool):
        # No separator/grid line above the status bar
        bar = pygame.Rect(0, self.gmap.h*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)

        visited = sum(1 for x in range(self.gmap.w) for y in range(self.gmap.h)
                      if self.gmap.visited[x][y])
        spills_total = sum(1 for x in range(self.gmap.w) for y in range(self.gmap.h)
                           if self.gmap.truth[x][y] == 1)
        spills_cleaned = self.robot.cleaned_spills

        status = self.robot.done_reason if done else 'RUNNING'
        txt = (
            f"steps {self.robot.steps_taken}/{MAX_STEPS} | "
            f"visited {visited} | spills cleaned {spills_cleaned}/{spills_total} | "
            f"status {status}"
        )
        self.screen.blit(self.font.render(txt, True, COLORS['text']), (8, self.gmap.h*CELL + 10))

    def loop(self):
        running = True
        done = False
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                    elif e.key == pygame.K_SPACE: self.paused = not self.paused
                    elif e.key == pygame.K_n:
                        if not done:
                            cont = self.robot.step()
                            if not cont: done = True
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.step_interval = max(1/240, self.step_interval - 0.01)
                    elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.step_interval = min(1/1, self.step_interval + 0.01)
                    elif e.key == pygame.K_t:
                        self.show_tree = not self.show_tree
                    elif e.key == pygame.K_s:
                        fname = f"stc_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.reset_world()
                        done = False

            if not self.paused and not done:
                self.accumulator += dt
                while self.accumulator >= self.step_interval:
                    cont = self.robot.step()
                    if not cont:
                        done = True
                        break
                    self.accumulator -= self.step_interval

            # draw
            self.screen.fill(COLORS['bg'])   # white
            self.draw_grid()
            self.draw_status(done)
            pygame.display.flip()
        pygame.quit()

    def reset_world(self):
        if USE_CUSTOM_MAP:
            truth = CUSTOM_MAP_ARRAY
            gmap = GridMap(0, 0, truth=truth)
        else:
            gmap = GridMap(DEFAULT_W, DEFAULT_H, P_OBS, P_SPILL, seed=SEED)
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
        g = GridMap(0, 0, truth=CUSTOM_MAP_ARRAY)
        r = OnlineSTCRobot(g, start=START)
        return g, r
    else:
        attempts = 0
        while True:
            attempts += 1
            g = GridMap(DEFAULT_W, DEFAULT_H, P_OBS, P_SPILL, seed=SEED)
            r = OnlineSTCRobot(g, start=START)
            if attempts > 0:
                return g, r

def main():
    g, r = generate_world()
    viewer = Viewer(g, r)
    viewer.loop()

if __name__ == "__main__":
    main()
