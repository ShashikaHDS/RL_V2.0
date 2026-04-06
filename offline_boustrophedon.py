"""
Boustrophedon Cellular Decomposition (BCD) + Lawn-mower Coverage
================================================================
Offline coverage (full map known). Values:
- 0 = free
- 1 = chemical spill (traversable; turns green when cleaned)
- 2 = obstacle (blocked)

Visualization:
- White background
- Obstacles dark gray
- Spills red -> turn green when traversed
- Path only (black polyline), no robot glyph, no grid lines

Controls:
- SPACE : pause/resume
- N     : single step
- +/-   : speed up / slow down (cells per second)
- R     : rebuild route (same map)
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
CELL = 20              # pixels per grid cell
FPS = 60
MOVES_PER_SECOND = 30     # animation speed (cells/sec)
START = (13, 19)            # (x, y) in base-grid coords

# Stop early after N cell-steps; set to None for full coverage
STEP_BUDGET: Optional[int] = 228  # e.g., 900

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
    'spill_cleaned': (57,255,20),
    'free': (255, 255, 255),
    'text': (20, 20, 20),
    'path': (0, 0, 0),
}

# =====================
# Boustrophedon decomposition on a grid
# =====================
Interval = Tuple[int,int]            # (y0, y1) inclusive
CellID = int

class BCD:
    """Boustrophedon cellular decomposition on a binary traversability grid."""
    def __init__(self, grid: np.ndarray):
        """
        grid: HxW, traversable if grid != 2
        Cells are stored as mapping: cell_id -> dict[x] = (y0,y1)
        """
        self.grid = grid
        self.H, self.W = grid.shape
        self.free_mask = (grid != 2)

        self.cells: Dict[CellID, Dict[int, Interval]] = {}
        self.adj: Dict[CellID, Set[CellID]] = defaultdict(set)
        self._decompose()

    def _col_intervals(self, x: int) -> List[Interval]:
        """Contiguous free runs along column x (inclusive y0..y1)."""
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
        next_id: CellID = 0
        prev: List[Tuple[Interval, CellID]] = []  # (interval, id)

        for x in range(self.W):
            curr_intervals = self._col_intervals(x)
            # Build overlap maps
            prev_to_curr: Dict[int, List[int]] = defaultdict(list)  # prev_idx -> [curr_idx...]
            curr_to_prev: Dict[int, List[int]] = defaultdict(list)  # curr_idx -> [prev_idx...]

            for i, (py, pid) in enumerate(prev):
                for j, cy in enumerate(curr_intervals):
                    if self._overlap(py, cy):
                        prev_to_curr[i].append(j)
                        curr_to_prev[j].append(i)

            # Prepare assignments for current intervals
            assigned: Dict[int, CellID] = {}

            # 1) Handle merges: curr overlapped by multiple prev -> new cell
            for j, prev_list in curr_to_prev.items():
                if len(prev_list) >= 2:
                    cid = next_id; next_id += 1
                    assigned[j] = cid
                    # adjacency from all prev cells to this merged cell
                    for i in prev_list:
                        pid = prev[i][1]
                        self.adj[pid].add(cid)
                        self.adj[cid].add(pid)

            # 2) Handle splits: prev overlapped by multiple curr -> each curr gets new cell
            for i, curr_list in prev_to_curr.items():
                if len(curr_list) >= 2:
                    pid = prev[i][1]
                    for j in curr_list:
                        if j in assigned:  # already assigned due to merge
                            self.adj[pid].add(assigned[j])
                            self.adj[assigned[j]].add(pid)
                        else:
                            cid = next_id; next_id += 1
                            assigned[j] = cid
                            self.adj[pid].add(cid)
                            self.adj[cid].add(pid)

            # 3) One-to-one continuations
            for j, cy in enumerate(curr_intervals):
                if j in assigned:
                    continue
                plist = curr_to_prev.get(j, [])
                if len(plist) == 1:
                    i = plist[0]
                    # ensure that prev interval also maps only to this curr (true continuation)
                    if len(prev_to_curr.get(i, [])) == 1:
                        assigned[j] = prev[i][1]

            # 4) Fresh components (no overlap at all)
            for j, cy in enumerate(curr_intervals):
                if j not in assigned:
                    cid = next_id; next_id += 1
                    assigned[j] = cid

            # Record current intervals into cells
            for j, cy in enumerate(curr_intervals):
                cid = assigned[j]
                self.cells.setdefault(cid, {})[x] = cy

            # Prepare prev for next column
            prev = [ (curr_intervals[j], assigned[j]) for j in range(len(curr_intervals)) ]

# =====================
# Routing inside and between cells
# =====================
def bfs_grid(trav: np.ndarray, s: Tuple[int,int], t: Tuple[int,int]) -> List[Tuple[int,int]]:
    """4-neighbor shortest path on traversable mask (True=free)."""
    if s == t: return [s]
    H, W = trav.shape
    Q = deque([s])
    par = {s: None}
    while Q:
        x, y = Q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and trav[ny, nx] and (nx,ny) not in par:
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
    return [s]  # fallback

def choose_cell_entry(cell_cols: Dict[int, Interval], prev_xy: Tuple[int,int]) -> Tuple[int,int]:
    """Pick an entry point (x,y) in this cell near prev_xy."""
    # closest column in L1; then clamp y into that column's interval to nearest end
    px, py = prev_xy
    xs = sorted(cell_cols.keys())
    x = min(xs, key=lambda xx: abs(xx - px))
    y0, y1 = cell_cols[x]
    # snap to closest of the two ends to avoid mid-interval start duplicates
    y = y0 if abs(py - y0) <= abs(py - y1) else y1
    return (x, y)

def sweep_cell(cell_cols: Dict[int, Interval], start_xy: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Produce boustrophedon (lawn-mower) coverage inside a single cell."""
    path: List[Tuple[int,int]] = []
    xs = sorted(cell_cols.keys())
    if not xs: return path

    # Start from the closest column to start_xy; then sweep outward across all columns
    sx, sy = start_xy
    start_col = min(xs, key=lambda xx: abs(xx - sx))
    start_idx = xs.index(start_col)

    # Order columns: start_col -> right, then start_col-1 -> left (zig)
    order = [start_col]
    # append to the right
    for k in range(start_idx+1, len(xs)): order.append(xs[k])
    # then to the left
    for k in range(start_idx-1, -1, -1): order.append(xs[k])

    # Start y: pick closer end at start_col
    y0, y1 = cell_cols[start_col]
    cur_y = y0 if abs(sy - y0) <= abs(sy - y1) else y1

    # Visit start column first
    if cur_y == y0:
        # go up to y1
        for y in range(y0, y1+1):
            path.append((start_col, y))
    else:
        for y in range(y1, y0-1, -1):
            path.append((start_col, y))

    # For remaining columns, always start at the end (y0 or y1) closest to previous y
    last_y = path[-1][1]
    for x in order[1:]:
        y0, y1 = cell_cols[x]
        # horizontal hop to column x at the closest end
        if abs(last_y - y0) <= abs(last_y - y1):
            start_y, end_y = y0, y1
        else:
            start_y, end_y = y1, y0

        # move horizontally one step to (x, last_y) if needed (grid BFS will add connectors between cells,
        # but inside a cell adjacent columns are guaranteed traversable at both ends)
        # ensure we horizontally connect at start_y
        # If last_y != start_y, we’ll just “start” the vertical sweep at start_y; the last point was (prev_x,last_y)
        # add a direct horizontal step if they are adjacent columns:
        prev_x = path[-1][0]
        if x != prev_x:
            path.append((x, last_y))  # lateral move at constant y (shared overlap is guaranteed)

        # If lateral y differs from the chosen start_y, walk vertically to start_y
        if path[-1][1] != start_y:
            step = 1 if start_y > path[-1][1] else -1
            for y in range(path[-1][1]+step, start_y+step, step):
                path.append((x, y))

        # Now sweep full segment to end_y
        step = 1 if end_y >= start_y else -1
        for y in range(start_y, end_y+step, step):
            # Avoid duplicating the very first point if already present
            if not path or (x, y) != path[-1]:
                path.append((x, y))

        last_y = path[-1][1]

    return path

# =====================
# Planner: cells + ordering + connectors
# =====================
class BoustrophedonPlanner:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        self.grid = grid
        self.H, self.W = grid.shape
        self.trav = (grid != 2)

        self.decomp = BCD(grid)
        self.cells = self.decomp.cells
        self.adj = self.decomp.adj

        # find start cell if any
        sx, sy = start
        self.start_cell = None
        for cid, cols in self.cells.items():
            if sx in cols:
                y0, y1 = cols[sx]
                if y0 <= sy <= y1:
                    self.start_cell = cid
                    break
        if self.start_cell is None:
            # pick the nearest cell entry if start lies in obstacle
            best = None
            for cid, cols in self.cells.items():
                entry = choose_cell_entry(cols, start)
                if best is None or abs(entry[0]-sx)+abs(entry[1]-sy) < best[0]:
                    best = (abs(entry[0]-sx)+abs(entry[1]-sy), cid)
            self.start_cell = best[1] if best else None

        # order cells using BFS over the adjacency graph from start cell; if disconnected leftovers, append them
        self.order: List[CellID] = []
        seen: Set[CellID] = set()
        if self.start_cell is not None:
            Q = deque([self.start_cell]); seen.add(self.start_cell)
            while Q:
                u = Q.popleft()
                self.order.append(u)
                for v in self.adj.get(u, []):
                    if v not in seen:
                        seen.add(v); Q.append(v)
        # append any remaining isolated cells (rare)
        for cid in self.cells:
            if cid not in seen:
                self.order.append(cid)

        # build full route: connectors (BFS) + cell sweeps
        self.route: List[Tuple[int,int]] = self._make_route(start)

    def _make_route(self, start: Tuple[int,int]) -> List[Tuple[int,int]]:
        route: List[Tuple[int,int]] = []
        cur = start
        for k, cid in enumerate(self.order):
            cols = self.cells[cid]
            entry = choose_cell_entry(cols, cur)
            # connector from cur to entry (if needed)
            if not route:
                if cur != entry:
                    route += bfs_grid(self.trav, cur, entry)[:-1]  # exclude duplicate entry
            else:
                if route[-1] != entry:
                    conn = bfs_grid(self.trav, route[-1], entry)
                    route += conn[1:]  # skip duplicate

            # sweep this cell
            cell_path = sweep_cell(cols, entry)
            if route:
                # avoid duplicate first
                if cell_path and cell_path[0] == route[-1]:
                    route += cell_path[1:]
                else:
                    route += cell_path
            else:
                route += cell_path

            cur = route[-1] if route else start
        return route

# =====================
# Runner (simulate traversal, update cleaned/visited)
# =====================
class Runner:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        self.grid = grid.copy()
        self.H, self.W = grid.shape
        self.trav = (grid != 2)

        self.plan = BoustrophedonPlanner(self.grid, start)
        self.route = self.plan.route

        self.idx = 0
        self.done = False
        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.cleaned = np.zeros_like(self.grid, dtype=bool)

        # mark start
        if self.route:
            x, y = self.route[0]
            self._touch((x, y))

        self.budget = STEP_BUDGET
        self.used = 0

    def _touch(self, p: Tuple[int,int]):
        x, y = p
        self.visited[y, x] = True
        if self.grid[y, x] == 1:
            self.cleaned[y, x] = True

    def step(self, k: int=1):
        if self.done:
            return
        for _ in range(k):
            if self.budget is not None and self.used >= self.budget:
                self.done = True
                break
            if self.idx+1 < len(self.route):
                self.idx += 1
                self._touch(self.route[self.idx])
                self.used += 1
            else:
                self.done = True
                break

# =====================
# Visualization (Pygame)
# =====================
class Viewer:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        pygame.init()
        self.grid = grid
        self.H, self.W = grid.shape
        self.runner = Runner(grid, start)

        self.w_px = self.W*CELL
        self.h_px = self.H*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Boustrophedon Coverage (Offline)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND

    def draw_world(self):
        # cells (no grid lines)
        for y in range(self.H):
            for x in range(self.W):
                v = self.grid[y, x]
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                if v == 2:
                    pygame.draw.rect(self.screen, COLORS['ob'], rect)
                elif v == 1:
                    if self.runner.cleaned[y, x]:
                        pygame.draw.rect(self.screen, COLORS['spill_cleaned'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['spill'], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['free'], rect)

        # path polyline
        if self.runner.idx >= 1:
            pts = []
            for i in range(0, self.runner.idx+1):
                x, y = self.runner.route[i]
                cx = x*CELL + CELL//2
                cy = y*CELL + CELL//2
                pts.append((cx, cy))
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

    def draw_status(self):
        bar = pygame.Rect(0, self.H*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)
        visited = int(self.runner.visited.sum())
        total_trav = int(np.sum(self.grid != 2))
        spills_total = int(np.sum(self.grid == 1))
        spills_clean = int(np.sum(self.runner.cleaned))
        status = "DONE" if self.runner.done else "RUNNING"
        budget_txt = "∞" if STEP_BUDGET is None else str(STEP_BUDGET)
        txt = (
            f"steps {self.runner.idx+1}/{len(self.runner.route)} "
            f"(budget {budget_txt}, used {self.runner.used}) | "
            f"visited {visited}/{total_trav} | "
            f"spills cleaned {spills_clean}/{spills_total} | "
            f"status {status}"
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
                    if e.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                    elif e.key == pygame.K_SPACE: self.paused = not self.paused
                    elif e.key == pygame.K_n: self.runner.step(1)
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.step_interval = max(1/240, self.step_interval - 0.01)
                    elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.step_interval = min(1/1, self.step_interval + 0.01)
                    elif e.key == pygame.K_s:
                        fname = f"boustro_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.runner = Runner(self.grid, START)
                        self.accumulator = 0.0
                        self.paused = False

            if not self.paused and not self.runner.done:
                self.accumulator += dt
                while self.accumulator >= self.step_interval:
                    self.runner.step(1)
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
    grid = CUSTOM_MAP_ARRAY.copy() if USE_CUSTOM_MAP else np.zeros((40, 60), dtype=int)
    viewer = Viewer(grid, START)
    viewer.loop()

if __name__ == "__main__":
    main()
