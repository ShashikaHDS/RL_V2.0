"""
Offline STC (Supercell Decomposition + Spanning Tree + Euler Tour)
==================================================================

Now supports a STEP_BUDGET to stop after a fixed number of steps.

Visualization:
- White background
- Spill (uncleaned)=red, Spill (cleaned)=green, Obstacles=dark gray, Free=white
- Path only (robot not drawn)
- No grid lines
- Optional STC tree edges toggle (press T)

Controls:
- SPACE : pause/resume
- N     : single step
- +/-   : speed up / slow down (cells per second)
- R     : reset (rebuild route for the same map)
- T     : toggle STC tree edges
- S     : screenshot
- ESC/Q : quit
"""

from __future__ import annotations
import pygame
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

# =====================
# CONFIG
# =====================
CELL = 20                # pixels per grid cell (visual scale)
FPS = 60
MOVES_PER_SECOND = 30     # path animation speed (cells per second)
START = (13, 19)            # (x, y) in base grid coordinates

# Limit how many steps to animate/travel.
# - None  -> no limit (cover entire route)
# - int   -> stop after that many steps (cells traversed)
STEP_BUDGET: Optional[int] = 220
# e.g., 800

# Map: 0=free, 1=spill (traversable), 2=obstacle (blocked)
USE_CUSTOM_MAP = True
CUSTOM_MAP_ARRAY = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# Colors
COLORS = {
    'bg': (255, 255, 255),        # white background
    'ob': (50, 50, 55),           # obstacle
    'spill': (255, 0, 0),         # red (uncleaned)
    'spill_cleaned': (57,255,20), # green (cleaned)
    'free': (255, 255, 255),      # free cells white
    'text': (20, 20, 20),
    'edge': (160, 200, 255),      # STC tree edges
    'path': (0, 0, 0),            # black path
}

DIRS4 = {
    'L': (-1, 0), 'R': (1, 0), 'U': (0, -1), 'D': (0, 1)
}
OPPOSITE = {'L':'R','R':'L','U':'D','D':'U'}

# =====================
# Helpers (supercell / micro)
# =====================
def super_dims(W: int, H: int) -> Tuple[int,int]:
    return W//2, H//2  # ignore last row/col if odd

def super_top_left(sx: int, sy: int) -> Tuple[int,int]:
    return 2*sx, 2*sy

def super_cells(x0: int, y0: int):
    # order: 0 TL, 1 TR, 2 BL, 3 BR
    return [(x0, y0), (x0+1, y0), (x0, y0+1), (x0+1, y0+1)]

MICRO_NEI = {
    0: [1,2], 1: [0,3], 2: [0,3], 3: [1,2]
}
ANCHOR = {'L': [0,2], 'R': [1,3], 'U': [0,1], 'D': [2,3]}
CROSS_MAP = {
    'R': {1:0, 3:2},
    'L': {0:1, 2:3},
    'U': {0:2, 1:3},
    'D': {2:0, 3:1},
}

def dir_between(a: Tuple[int,int], b: Tuple[int,int]) -> Optional[str]:
    dx = b[0]-a[0]; dy = b[1]-a[1]
    if dx == 1 and dy == 0: return 'R'
    if dx ==-1 and dy == 0: return 'L'
    if dx == 0 and dy == 1: return 'D'
    if dx == 0 and dy ==-1: return 'U'
    return None

def super_free(grid: np.ndarray, sx: int, sy: int) -> bool:
    x0,y0 = super_top_left(sx,sy)
    if x0+1 >= grid.shape[1] or y0+1 >= grid.shape[0]:
        return False
    block = grid[y0:y0+2, x0:x0+2]
    return np.all(block != 2)

def micro_plan(start_idx: int, exit_dir: Optional[str]) -> List[int]:
    """BFS on 2x2 micro-cells to cover all 4 and end on exit side if provided."""
    goal_set = set(range(4)) if exit_dir is None else set(ANCHOR[exit_dir])
    start_mask = 1<<start_idx
    from collections import deque
    Q = deque()
    Q.append((start_idx, start_mask))
    parent = {(start_idx, start_mask): None}
    while Q:
        idx, mask = Q.popleft()
        if mask == 0b1111 and (exit_dir is None or idx in goal_set):
            path = []
            cur = (idx, mask)
            while cur is not None:
                i, m = cur
                path.append(i)
                cur = parent[cur]
            path.reverse()
            return path
        for j in MICRO_NEI[idx]:
            m2 = mask | (1<<j)
            state = (j, m2)
            if state not in parent:
                parent[state] = (idx, mask)
                Q.append(state)
    return [start_idx, *(k for k in range(4) if k!=start_idx)]

# =====================
# STC planner (offline)
# =====================
class OfflineSTCPlanner:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        self.grid = grid
        self.H, self.W = grid.shape
        self.SW, self.SH = super_dims(self.W, self.H)
        self.free_super: Set[Tuple[int,int]] = set()
        self.graph: Dict[Tuple[int,int], List[Tuple[int,int]]] = defaultdict(list)

        for sy in range(self.SH):
            for sx in range(self.SW):
                if super_free(grid, sx, sy):
                    self.free_super.add((sx,sy))

        for (sx,sy) in self.free_super:
            for d,(dx,dy) in DIRS4.items():
                nx, ny = sx+dx, sy+dy
                if (nx,ny) in self.free_super:
                    self.graph[(sx,sy)].append((nx,ny))

        ssx, ssy = start[0]//2, start[1]//2
        self.start_super = self._nearest_free_super((ssx, ssy))

        self.tree_adj, self.tree_edges = self._build_spanning_tree(self.start_super)
        self.euler_nodes = self._euler_walk(self.start_super)
        self.route = self._build_cell_route()

    def _nearest_free_super(self, s: Tuple[int,int]) -> Tuple[int,int]:
        if s in self.free_super:
            return s
        from collections import deque
        Q = deque([s]); seen = {s}
        while Q:
            u = Q.popleft()
            for d,(dx,dy) in DIRS4.items():
                v = (u[0]+dx, u[1]+dy)
                if v in seen or v[0]<0 or v[1]<0 or v[0]>=self.SW or v[1]>=self.SH:
                    continue
                if v in self.free_super:
                    return v
                seen.add(v); Q.append(v)
        return next(iter(self.free_super))

    def _build_spanning_tree(self, root: Tuple[int,int]):
        tree_adj: Dict[Tuple[int,int], List[Tuple[int,int]]] = defaultdict(list)
        tree_edges: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
        seen = set([root]); stack = [root]
        while stack:
            u = stack.pop()
            for v in self.graph[u]:
                if v not in seen:
                    seen.add(v)
                    tree_adj[u].append(v)
                    tree_adj[v].append(u)
                    tree_edges.add(tuple(sorted([u,v])))
                    stack.append(v)
        return tree_adj, tree_edges

    def _euler_walk(self, root: Tuple[int,int]) -> List[Tuple[int,int]]:
        seq: List[Tuple[int,int]] = [root]
        def dfs(u: Tuple[int,int], p: Optional[Tuple[int,int]]):
            for v in self.tree_adj.get(u, []):
                if v == p: 
                    continue
                seq.append(v)
                dfs(v, u)
                seq.append(u)
        dfs(root, None)
        return seq

    def _build_cell_route(self) -> List[Tuple[int,int]]:
        if not self.euler_nodes:
            return []
        covered: Dict[Tuple[int,int], int] = defaultdict(int)
        route: List[Tuple[int,int]] = []

        def append_micro(s: Tuple[int,int], start_idx: int, exit_dir: Optional[str]):
            x0,y0 = super_top_left(*s)
            mpath_idx = micro_plan(start_idx, exit_dir)
            for k, idx in enumerate(mpath_idx):
                cx, cy = super_cells(x0,y0)[idx]
                if route and (cx,cy) == route[-1]:
                    continue
                route.append((cx,cy))
                covered[s] |= (1<<idx)

        cur_s = self.euler_nodes[0]
        next_dir = dir_between(cur_s, self.euler_nodes[1]) if len(self.euler_nodes) > 1 else None
        start_idx = ANCHOR[next_dir][0] if next_dir else 0
        append_micro(cur_s, start_idx, next_dir)

        for i in range(1, len(self.euler_nodes)):
            prev_s = self.euler_nodes[i-1]
            cur_s  = self.euler_nodes[i]
            move_dir = dir_between(prev_s, cur_s)
            enter_dir = OPPOSITE[move_dir]

            px0, py0 = super_top_left(*prev_s)
            last = route[-1]
            prev_cells = super_cells(px0, py0)
            try:
                last_idx_prev = prev_cells.index(last)
            except ValueError:
                last_idx_prev = ANCHOR[move_dir][0]

            if last_idx_prev not in set(ANCHOR[move_dir]):
                for nb in MICRO_NEI[last_idx_prev]:
                    if nb in set(ANCHOR[move_dir]):
                        nbx, nby = prev_cells[nb]
                        route.append((nbx,nby))
                        last_idx_prev = nb
                        break

            cx0, cy0 = super_top_left(*cur_s)
            entry_idx_cur = CROSS_MAP[move_dir][last_idx_prev]
            cur_cells = super_cells(cx0, cy0)
            route.append(cur_cells[entry_idx_cur])

            next_dir = dir_between(cur_s, self.euler_nodes[i+1]) if i < len(self.euler_nodes)-1 else None
            append_micro(cur_s, entry_idx_cur, next_dir)

        return route

# =====================
# Runner (animates the route)
# =====================
class OfflineSTCRunner:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        self.grid = grid.copy()
        self.H, self.W = grid.shape
        self.planner = OfflineSTCPlanner(self.grid, start)
        self.route = self.planner.route
        self.idx = 0
        self.done = False

        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.cleaned = np.zeros_like(self.grid, dtype=bool)

        # Step budget (None = unlimited)
        self.budget = STEP_BUDGET
        self.steps_taken = 0

        if self.route:
            x,y = self.route[0]
            self._touch((x,y))

    def reset(self):
        self.__init__(self.grid, START)

    def step(self, k: int=1):
        if self.done:
            return
        for _ in range(k):
            # Respect budget if set
            if self.budget is not None and self.steps_taken >= self.budget:
                self.done = True
                break
            if self.idx+1 < len(self.route):
                self.idx += 1
                self._touch(self.route[self.idx])
                self.steps_taken += 1
            else:
                self.done = True
                break

    def _touch(self, p: Tuple[int,int]):
        x,y = p
        self.visited[y, x] = True
        if self.grid[y, x] == 1 and not self.cleaned[y, x]:
            self.cleaned[y, x] = True

# =====================
# Visualization (Pygame)
# =====================
class Viewer:
    def __init__(self, grid: np.ndarray, start: Tuple[int,int]):
        pygame.init()
        self.grid = grid
        self.H, self.W = grid.shape
        self.runner = OfflineSTCRunner(grid, start)

        self.w_px = self.W*CELL
        self.h_px = self.H*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Offline STC (Supercell + Euler Tour)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND

        self.show_tree = False  # toggle STC tree edges

    def draw_world(self):
        # draw cells (no grid lines)
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

        # STC tree edges between supercell centers
        if self.show_tree:
            edges = self.runner.planner.tree_edges
            for (a,b) in edges:
                ax, ay = a; bx, by = b
                acx = (2*ax+1)*CELL
                acy = (2*ay+1)*CELL
                bcx = (2*bx+1)*CELL
                bcy = (2*by+1)*CELL
                pygame.draw.line(self.screen, COLORS['edge'], (acx,acy), (bcx,bcy), 2)

        # path polyline (up to current index)
        if self.runner.idx >= 1:
            pts = []
            for i in range(0, self.runner.idx+1):
                x,y = self.runner.route[i]
                cx = x*CELL + CELL//2
                cy = y*CELL + CELL//2
                pts.append((cx,cy))
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

    def draw_status(self):
        bar = pygame.Rect(0, self.H*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)

        visited = int(self.runner.visited.sum())
        total_trav = int(np.sum(self.grid != 2))
        spills_total = int(np.sum(self.grid == 1))
        spills_clean = int(np.sum(self.runner.cleaned))

        status = "DONE" if self.runner.done else "RUNNING"
        budget_txt = "∞" if self.runner.budget is None else str(self.runner.budget)
        txt = (
            f"steps {self.runner.idx+1}/{len(self.runner.route)} "
            f"(budget {budget_txt}, used {self.runner.steps_taken}) | "
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
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_n:
                        self.runner.step(1)
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.step_interval = max(1/240, self.step_interval - 0.01)
                    elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.step_interval = min(1/1, self.step_interval + 0.01)
                    elif e.key == pygame.K_t:
                        self.show_tree = not self.show_tree
                    elif e.key == pygame.K_s:
                        fname = f"stc_offline_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.runner = OfflineSTCRunner(self.grid, START)
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
