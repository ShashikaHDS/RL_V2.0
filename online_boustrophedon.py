"""
Online Boustrophedon Coverage (Sensor-based, Radius=1)
=====================================================

- Algorithm is ONLINE: it only plans over cells revealed by a radius-1 sensor.
- Visualization shows the FULL ground-truth map from the start (optional flag below).
- Values: 0=free, 1=spill (cleaned -> green), 2=obstacle.

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
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set

# =====================
# CONFIG
# =====================
CELL = 16                   # pixels per grid cell
FPS = 60
MOVES_PER_SECOND = 24       # animation speed (cells/sec)
START = (1, 1)              # (x, y) in base-grid coords (must be traversable in truth)
SENSOR_RADIUS = 1           # Chebyshev radius = 1 (8-neighborhood)
STEP_BUDGET: Optional[int] = 328  # e.g. 800 to stop early; None = unlimited

# Visualization: show the FULL ground-truth map from the start (visual only)
SHOW_FULL_TRUTH_MAP = True

# Map: 0=free, 1=spill, 2=obstacle
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
    'spill': (255, 0, 0),
    'spill_cleaned': (0, 180, 0),
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
        self.truth = truth.copy()           # HxW with 0/1/2
        self.H, self.W = self.truth.shape
        sx, sy = start
        if not (0 <= sx < self.W and 0 <= sy < self.H):
            raise ValueError("START out of bounds.")
        if self.truth[sy, sx] == 2:
            raise ValueError("START is inside an obstacle.")

        # known map: -1 unknown, 0 free, 1 spill, 2 obstacle
        self.known = np.full_like(self.truth, -1)
        self.visited = np.zeros_like(self.truth, dtype=bool)
        self.cleaned = np.zeros_like(self.truth, dtype=bool)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def reveal(self, center: Tuple[int,int], radius: int=SENSOR_RADIUS):
        cx, cy = center
        for y in range(cy-radius, cy+radius+1):
            for x in range(cx-radius, cx+radius+1):
                if self.in_bounds(x,y) and cheby((x,y),(cx,cy)) <= radius:
                    self.known[y, x] = self.truth[y, x]

    def is_known_traversable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x,y): return False
        return self.known[y, x] in (0, 1)

    def mark_touch(self, x: int, y: int):
        self.visited[y, x] = True
        if self.truth[y, x] == 1:
            self.cleaned[y, x] = True

    def frontier_mask(self) -> np.ndarray:
        """Known traversable cells that touch any unknown neighbor (4-neigh)."""
        fm = np.zeros_like(self.known, dtype=bool)
        for y in range(self.H):
            for x in range(self.W):
                if self.known[y, x] in (0,1):
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if self.in_bounds(nx, ny) and self.known[ny, nx] == -1:
                            fm[y, x] = True
                            break
        return fm

# =====================
# Online BCD on the known map
# =====================
Interval = Tuple[int,int]  # y0,y1 inclusive
CellID = int

class BCD:
    """Boustrophedon decomposition on a binary traversability mask.
       Unknown cells are treated as blocked (not part of any cell)."""
    def __init__(self, known: np.ndarray):
        self.known = known
        self.H, self.W = known.shape
        self.free_mask = (known == 0) | (known == 1)

        self.cells: Dict[CellID, Dict[int, Interval]] = {}  # id -> {x: (y0,y1)}
        self._decompose()

    def _col_intervals(self, x: int) -> List[Interval]:
        runs: List[Interval] = []
        in_run = False
        y0 = 0
        for y in range(self.H):
            if self.free_mask[y, x]:
                if not in_run:
                    in_run = True
                    y0 = y
            else:
                if in_run:
                    runs.append((y0, y-1))
                    in_run = False
        if in_run: runs.append((y0, self.H-1))
        return runs

    @staticmethod
    def _overlap(a: Interval, b: Interval) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])

    def _decompose(self):
        next_id = 0
        prev: List[Tuple[Interval, CellID]] = []

        for x in range(self.W):
            curr = self._col_intervals(x)
            # build overlap maps
            prev_to_curr: Dict[int, List[int]] = defaultdict(list)
            curr_to_prev: Dict[int, List[int]] = defaultdict(list)
            for i, (py, pid) in enumerate(prev):
                for j, cy in enumerate(curr):
                    if self._overlap(py, cy):
                        prev_to_curr[i].append(j)
                        curr_to_prev[j].append(i)

            assigned: Dict[int, CellID] = {}

            # merges
            for j, plist in curr_to_prev.items():
                if len(plist) >= 2:
                    cid = next_id; next_id += 1
                    assigned[j] = cid

            # splits
            for i, clist in prev_to_curr.items():
                if len(clist) >= 2:
                    pid = prev[i][1]
                    for j in clist:
                        if j not in assigned:
                            cid = next_id; next_id += 1
                            assigned[j] = cid

            # one-to-one
            for j, cy in enumerate(curr):
                if j in assigned: continue
                plist = curr_to_prev.get(j, [])
                if len(plist) == 1:
                    i = plist[0]
                    if len(prev_to_curr.get(i, [])) == 1:
                        assigned[j] = prev[i][1]

            # fresh ones
            for j, cy in enumerate(curr):
                if j not in assigned:
                    cid = next_id; next_id += 1
                    assigned[j] = cid

            # record
            for j, cy in enumerate(curr):
                cid = assigned[j]
                self.cells.setdefault(cid, {})[x] = cy

            prev = [(curr[j], assigned[j]) for j in range(len(curr))]

    def find_cell_containing(self, x: int, y: int) -> Optional[CellID]:
        for cid, cols in self.cells.items():
            if x in cols:
                y0,y1 = cols[x]
                if y0 <= y <= y1:
                    return cid
        return None

# =====================
# Sweep path inside a cell (on known traversable)
# =====================
def sweep_cell(cell_cols: Dict[int, Interval],
               start_xy: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Lawn-mower sweep of a single BCD cell, starting near start_xy (inside the cell)."""
    path: List[Tuple[int,int]] = []
    xs = sorted(cell_cols.keys())
    if not xs: return path

    sx, sy = start_xy
    # pick closest column as start
    start_col = min(xs, key=lambda xx: abs(xx - sx))
    start_idx = xs.index(start_col)

    # Column order: start -> right, then left
    order = [start_col]
    for k in range(start_idx+1, len(xs)): order.append(xs[k])
    for k in range(start_idx-1, -1, -1): order.append(xs[k])

    # pick closer end on start_col
    y0, y1 = cell_cols[start_col]
    cur_y = y0 if abs(sy - y0) <= abs(sy - y1) else y1

    # sweep start_col
    if cur_y == y0:
        for y in range(y0, y1+1):
            path.append((start_col, y))
    else:
        for y in range(y1, y0-1, -1):
            path.append((start_col, y))

    last_y = path[-1][1]
    # remaining columns
    for x in order[1:]:
        y0, y1 = cell_cols[x]

        # horizontal hop to x at closest end (y0 or y1)
        start_y, end_y = (y0, y1) if abs(last_y - y0) <= abs(last_y - y1) else (y1, y0)

        # lateral step into column x at last_y row
        prev_x = path[-1][0]
        if x != prev_x:
            path.append((x, last_y))

        # climb to start_y if needed
        if path[-1][1] != start_y:
            step = 1 if start_y > path[-1][1] else -1
            for yy in range(path[-1][1] + step, start_y + step, step):
                path.append((x, yy))

        # vertical sweep to end_y
        step = 1 if end_y >= start_y else -1
        for yy in range(start_y, end_y + step, step):
            if not path or (x,yy) != path[-1]:
                path.append((x, yy))

        last_y = path[-1][1]

    return path

# =====================
# BFS on known-traversable for connectors / frontier
# =====================
def bfs_known(world: GridWorld, s: Tuple[int,int], t: Tuple[int,int]) -> List[Tuple[int,int]]:
    """4-neighbor BFS limited to *known traversable* cells."""
    if s == t: return [s]
    H, W = world.H, world.W
    Q = deque([s])
    par = {s: None}
    while Q:
        x,y = Q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H:
                if world.is_known_traversable(nx, ny) and (nx,ny) not in par:
                    par[(nx,ny)] = (x,y)
                    if (nx,ny) == t:
                        # reconstruct
                        path = [(nx,ny)]
                        cur = (x,y)
                        while cur is not None:
                            path.append(cur)
                            cur = par[cur]
                        path.reverse()
                        return path
                    Q.append((nx,ny))
    return [s]

def nearest_frontier(world: GridWorld, from_xy: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    fm = world.frontier_mask()
    if not fm.any(): return None
    # BFS outward over known traversable until hitting a frontier tile
    H, W = world.H, world.W
    Q = deque([from_xy])
    seen = {from_xy}
    while Q:
        x,y = Q.popleft()
        if fm[y, x]:
            return (x,y)
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H:
                if world.is_known_traversable(nx, ny) and (nx,ny) not in seen:
                    seen.add((nx,ny))
                    Q.append((nx,ny))
    return None

# =====================
# Online planner
# =====================
class OnlineBoustro:
    def __init__(self, world: GridWorld, start: Tuple[int,int]):
        self.world = world
        self.pos = start
        self.queue: deque[Tuple[int,int]] = deque()  # pending cell steps
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
        """Move one cell if queue has items. Returns True if moved."""
        if not self.queue:
            return False
        nx, ny = self.queue.popleft()
        # only step on known traversable
        if not self.world.is_known_traversable(nx, ny):
            return False
        self.pos = (nx, ny)
        self.world.reveal(self.pos, SENSOR_RADIUS)
        self.world.mark_touch(nx, ny)
        self.route.append(self.pos)
        self.steps_taken += 1
        return True

    def _plan_within_current_cell(self):
        """Plan/continue a boustrophedon sweep inside the current known cell (if any left to cover)."""
        known = self.world.known
        bcd = BCD(known)
        x,y = self.pos
        cid = bcd.find_cell_containing(x, y)
        if cid is None:
            return False  # not inside any known cell
        cols = bcd.cells[cid]

        # If everything in this cell already visited, do nothing
        any_unvisited = False
        for cx, (y0,y1) in cols.items():
            for yy in range(y0, y1+1):
                if self.world.visited[yy, cx] == False:
                    any_unvisited = True
                    break
            if any_unvisited: break
        if not any_unvisited:
            return False

        # Clamp start inside the cell if needed
        if x not in cols:
            xs = sorted(cols.keys())
            x = min(xs, key=lambda xx: abs(xx - x))
            y0, y1 = cols[x]
            y = y0 if abs(y - y0) <= abs(y - y1) else y1
        else:
            y0, y1 = cols[x]
            y = min(max(y, y0), y1)

        cell_path = sweep_cell(cols, (x,y))

        # Trim prefix up to current position to avoid backtracking
        try:
            start_idx = cell_path.index(self.pos)
        except ValueError:
            start_idx = 0
        trimmed = cell_path[start_idx+1:]  # exclude current pos

        # Enqueue steps
        self._enqueue(trimmed)
        return len(trimmed) > 0

    def _plan_to_frontier(self):
        """Route to nearest frontier tile (known traversable touching unknown)."""
        f = nearest_frontier(self.world, self.pos)
        if f is None:
            return False
        path = bfs_known(self.world, self.pos, f)
        if len(path) <= 1:
            return False
        self._enqueue(path[1:])  # skip current pos
        return True

    def _plan_to_unvisited_known(self):
        """If any known traversable tile remains unvisited, go to nearest."""
        H, W = self.world.H, self.world.W
        targets = [(x,y) for y in range(H) for x in range(W)
                   if self.world.is_known_traversable(x,y) and not self.world.visited[y, x]]
        if not targets:
            return False
        # BFS outward to nearest unvisited known
        Q = deque([self.pos])
        seen = {self.pos}
        par = {self.pos: None}
        goal = None
        while Q:
            u = Q.popleft()
            if u in targets:
                goal = u; break
            ux, uy = u
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                vx, vy = ux+dx, uy+dy
                if 0 <= vx < W and 0 <= vy < H and (vx,vy) not in seen and self.world.is_known_traversable(vx,vy):
                    seen.add((vx,vy))
                    par[(vx,vy)] = u
                    Q.append((vx,vy))
        if goal is None:
            return False
        # reconstruct
        path = [goal]
        cur = par[goal]
        while cur is not None:
            path.append(cur)
            cur = par[cur]
        path.reverse()
        if len(path) <= 1:
            return False
        self._enqueue(path[1:])
        return True

    def step(self) -> bool:
        """One online step: move if queue has actions, otherwise (re)plan."""
        if self.done:
            return False

        # Move one cell if we have a plan ready
        if self._tick_move():
            return True

        # No queued motion -> (re)sense and plan
        self.world.reveal(self.pos, SENSOR_RADIUS)

        # 1) try to continue/plan sweep inside current cell
        if self._plan_within_current_cell():
            return True

        # 2) move to frontier to discover new area
        if self._plan_to_frontier():
            return True

        # 3) fall back: if any known traversable remains unvisited, go there
        if self._plan_to_unvisited_known():
            return True

        # 4) nothing left to do
        self.done = True
        return False

# =====================
# Viewer / Runner
# =====================
class Viewer:
    def __init__(self, truth: np.ndarray):
        pygame.init()
        self.truth = truth
        self.H, self.W = truth.shape

        self.world = GridWorld(truth, START)
        # sense at start (so cleaned overlay can appear correctly)
        self.world.reveal(START, SENSOR_RADIUS)

        self.alg = OnlineBoustro(self.world, START)

        self.w_px = self.W*CELL
        self.h_px = self.H*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Online Boustrophedon (Sensor radius = 1)")
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
                else:
                    v = self.world.known[y, x]

                if v == 2:
                    pygame.draw.rect(self.screen, COLORS['ob'], rect)
                elif v == 1:
                    # If spill and already cleaned, draw green
                    if self.world.cleaned[y, x]:
                        pygame.draw.rect(self.screen, COLORS['spill_cleaned'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['spill'], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['free'], rect)

        # Draw path polyline (no robot glyph)
        if len(self.alg.route) >= 2:
            pts = [(x*CELL + CELL//2, y*CELL + CELL//2) for (x,y) in self.alg.route]
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

    def draw_status(self):
        bar = pygame.Rect(0, self.H*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)

        visited = int(self.world.visited.sum())
        known_trav = int(np.sum((self.world.known == 0) | (self.world.known == 1)))
        spills_known = int(np.sum(self.world.known == 1))
        spills_clean = int(np.sum(self.world.cleaned))
        status = "DONE" if self.alg.done else "RUNNING"
        budget_txt = "∞" if STEP_BUDGET is None else str(STEP_BUDGET)
        used = self.alg.steps_taken

        txt = (
            f"steps {used} (budget {budget_txt}) | "
            f"visited {visited} | known_trav {known_trav} | "
            f"spills cleaned {spills_clean}/{spills_known} | status {status}"
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
                        fname = f"online_boustro_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.world = GridWorld(self.truth, START)
                        self.world.reveal(START, SENSOR_RADIUS)
                        self.alg = OnlineBoustro(self.world, START)
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
