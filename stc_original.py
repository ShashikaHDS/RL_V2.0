"""
Original Online STC (boundary-following over the dual graph)
===========================================================

- Unknown map; robot reveals locally (Chebyshev SENSOR_RADIUS)
- Decomposition into 2x2 "supercells" (dual-graph nodes)
- Online DFS with a boundary-following / right-hand rule ordering on the dual graph
- On first entry to a supercell, micro-sweep its 2x2 to cover/clean and return to the entry anchor
- Chooses the next neighbor supercell using only currently known cells
- Backtracks on the dual graph when no unvisited adjacent supercell is safely enterable

Visualization:
- White background
- Spill (uncleaned) = red, Spill (cleaned) = green, Obstacles = dark gray, Free/unknown = white
- Path only (black polyline), no robot glyph
- No grid lines
- (Optional) toggle to show discovered dual-tree edges

Controls:
- SPACE : pause/resume
- N     : single step
- +/-   : speed up / slow down (cells per second)
- T     : toggle dual-tree edge overlay
- S     : screenshot
- R     : reset (same map)
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
MOVES_PER_SECOND = 24       # animation speed (cells per second)
START = (1, 1)              # (x, y) base-grid coordinate (must be traversable)
SENSOR_RADIUS = 3           # Chebyshev; not drawn, only used internally
STEP_BUDGET: Optional[int] = None  # None for unlimited; else stop after N traveled cells

# Map encoding: 0=free, 1=spill (traversable), 2=obstacle (blocked)
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
    'edge': (160, 200, 255),
    'path': (0, 0, 0),
}

# Directions on the dual graph (supercells)
DIRS = [(1,0),(0,1),(-1,0),(0,-1)]        # R, D, L, U
RIGHT = lambda h: (h+1) % 4
LEFT  = lambda h: (h+3) % 4
BACK  = lambda h: (h+2) % 4

# 2x2 micro indices: 0 TL, 1 TR, 2 BL, 3 BR
MICRO_NEI = {0:[1,2], 1:[0,3], 2:[0,3], 3:[1,2]}
ANCHOR = {  # which micro cells lie on each side of a 2x2
    0: [1,3],   # heading R -> right side of current (east boundary) -> TR/BR
    1: [2,3],   # heading D -> bottom boundary -> BL/BR
    2: [0,2],   # heading L -> left boundary -> TL/BL
    3: [0,1],   # heading U -> top boundary -> TL/TR
}
# If you leave via boundary micro idx 'a' in direction h, you enter neighbor at:
CROSS_MAP = {
    0: {1:0, 3:2},   # move R: current 1->neighbor 0, 3->2
    2: {0:1, 2:3},   # move L
    3: {0:2, 1:3},   # move U
    1: {2:0, 3:1},   # move D
}

def cheby(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# =====================
# Map & Knowledge
# =====================
class GridWorld:
    def __init__(self, truth: np.ndarray, start: Tuple[int,int]):
        self.truth = truth.copy()              # HxW (0 free, 1 spill, 2 obstacle)
        self.H, self.W = truth.shape
        sx, sy = start
        if self.truth[sy, sx] == 2:
            raise ValueError("START is inside an obstacle.")
        # -1 unknown, 0 free, 1 spill, 2 obstacle
        self.known = np.full_like(self.truth, -1)
        # housekeeping
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
        if not self.in_bounds(x,y): return
        self.visited[y, x] = True
        if self.truth[y, x] == 1:
            self.cleaned[y, x] = True

# =====================
# Supercell helpers
# =====================
def super_of(x: int, y: int) -> Tuple[int,int]:
    return x//2, y//2

def super_top_left(sx: int, sy: int) -> Tuple[int,int]:
    return 2*sx, 2*sy

def micro_coords(sx: int, sy: int, idx: int) -> Tuple[int,int]:
    x0, y0 = super_top_left(sx, sy)
    if   idx == 0: return (x0,   y0)
    elif idx == 1: return (x0+1, y0)
    elif idx == 2: return (x0,   y0+1)
    else:          return (x0+1, y0+1)

def micro_idx_of(sx: int, sy: int, x: int, y: int) -> int:
    x0, y0 = super_top_left(sx, sy)
    if x == x0   and y == y0:   return 0
    if x == x0+1 and y == y0:   return 1
    if x == x0   and y == y0+1: return 2
    return 3

def super_in_bounds(world: GridWorld, sx: int, sy: int) -> bool:
    x0, y0 = super_top_left(sx, sy)
    return (world.in_bounds(x0, y0) and world.in_bounds(x0+1, y0+1))

def free_micro_set(world: GridWorld, sx: int, sy: int) -> Set[int]:
    """Return set of micro indices (0..3) known to be traversable."""
    S: Set[int] = set()
    if not super_in_bounds(world, sx, sy):
        return S
    for idx in range(4):
        x, y = micro_coords(sx, sy, idx)
        if world.is_known_traversable(x,y):
            S.add(idx)
    return S

def micro_move_shortest(allowed: Set[int], a: int, b: int) -> List[int]:
    """Shortest path of micro indices from a to b staying within allowed."""
    if a == b: return [a]
    Q = deque([a])
    parent = {a: None}
    while Q:
        u = Q.popleft()
        for v in MICRO_NEI[u]:
            if v in allowed and v not in parent:
                parent[v] = u
                if v == b:
                    path = [v]
                    while path[-1] is not None:
                        path.append(parent[path[-1]])
                    path.pop()     # drop None
                    path.reverse()
                    return [a] + path[1:]
                Q.append(v)
    return [a]  # unreachable (shouldn't happen if both in allowed)

def micro_cover_return(allowed: Set[int], start_idx: int) -> List[int]:
    """Cover all allowed micro cells and return to start_idx (shortest)."""
    # state: (idx, mask) over allowed indices packed to 4 bits
    idx_list = sorted(list(allowed))
    pos_of = {idx:i for i,idx in enumerate(idx_list)}
    start_state = (start_idx, 1 << pos_of[start_idx])
    goal_mask = (1 << len(idx_list)) - 1

    Q = deque([start_state])
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start_state: None}
    while Q:
        idx, mask = Q.popleft()
        if mask == goal_mask and idx == start_idx:
            # reconstruct sequence of micro indices
            seq: List[int] = []
            cur = (idx, mask)
            while cur is not None:
                i, m = cur
                seq.append(i)
                cur = parent[cur]
            seq.reverse()
            return seq
        for j in MICRO_NEI[idx]:
            if j not in allowed: 
                continue
            m2 = mask | (1 << pos_of[j])
            s2 = (j, m2)
            if s2 not in parent:
                parent[s2] = (idx, mask)
                Q.append(s2)
    return [start_idx]

# =====================
# Online STC (dual-graph boundary following)
# =====================
class OnlineSTCClassic:
    def __init__(self, world: GridWorld, start_xy: Tuple[int,int]):
        self.world = world
        self.route: List[Tuple[int,int]] = []
        self.queue: deque[Tuple[int,int]] = deque()  # motion queue of base cells
        self.tree_edges: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()

        # start position & sensing
        self.pos = start_xy
        self.world.reveal(self.pos)
        self.world.mark_touch(*self.pos)
        self.route.append(self.pos)

        # current supercell, micro idx, heading (0=R,1=D,2=L,3=U)
        self.sx, self.sy = super_of(*self.pos)
        self.heading = 0
        self.cur_micro = micro_idx_of(self.sx, self.sy, *self.pos)

        # DFS structures over dual graph
        self.visited_super: Set[Tuple[int,int]] = set()
        self.parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
        self.stack: List[Tuple[int,int]] = []

        # bookkeeping
        self.steps_taken = 0
        self.done = False

        # enter initial supercell: cover micro (2x2) we already know around us
        self._ensure_micro_coverage_on_entry()

    # --- sensing & motion ---
    def _enqueue_cells(self, cells: List[Tuple[int,int]]):
        for c in cells:
            self.queue.append(c)

    def _tick_move(self) -> bool:
        """Advance by one base cell along queue; update sensing & cleaning. Returns False if no motion pending."""
        if not self.queue:
            return False
        nx, ny = self.queue.popleft()
        # refuse stepping into known obstacle
        if self.world.known[ny, nx] == 2:
            return False
        self.pos = (nx, ny)
        self.world.reveal(self.pos)
        self.world.mark_touch(nx, ny)
        self.route.append(self.pos)
        self.steps_taken += 1
        return True

    # --- micro coverage inside a supercell ---
    def _ensure_micro_coverage_on_entry(self):
        """On first entry to a supercell, cover all currently known-free micro cells and return to entry anchor."""
        sc = (self.sx, self.sy)
        if sc in self.visited_super:
            return
        self.visited_super.add(sc)

        # With SENSOR_RADIUS=3, the whole 2x2 and its neighbors are revealed on entry.
        allowed = free_micro_set(self.world, self.sx, self.sy)
        if self.cur_micro not in allowed:
            # if entry anchor got marked blocked (shouldn't if we stood there), just keep it
            allowed.add(self.cur_micro)

        micro_seq = micro_cover_return(allowed, self.cur_micro)  # micro indices
        cells: List[Tuple[int,int]] = []
        x0, y0 = super_top_left(self.sx, self.sy)
        for idx in micro_seq[1:]:  # skip the very first (we are already there)
            cx, cy = micro_coords(self.sx, self.sy, idx)
            cells.append((cx, cy))
        self._enqueue_cells(cells)

    # --- neighbor selection using boundary following on the dual graph ---
    def _neighbor_super(self, sx: int, sy: int, h: int) -> Tuple[int,int]:
        dx, dy = DIRS[h]
        return sx+dx, sy+dy

    def _available_dirs(self) -> List[int]:
        """Check directions in boundary-following order: right, straight, left, back.
        A direction is 'available to traverse now' if we can physically cross the shared boundary safely,
        using currently known cells on both sides."""
        order = [RIGHT(self.heading), self.heading, LEFT(self.heading), BACK(self.heading)]
        avail: List[int] = []
        allowed_here = free_micro_set(self.world, self.sx, self.sy)
        for h in order:
            # choose the closest boundary anchor on this side
            boundary = ANCHOR[h]
            # pick whichever is closer in micro graph
            if not allowed_here:
                continue
            # choose candidate anchor that is allowed
            candidates = [a for a in boundary if a in allowed_here]
            if not candidates:
                continue
            # simple choice: first candidate
            a = candidates[0]

            # cross target supercell & entry anchor
            nsx, nsy = self._neighbor_super(self.sx, self.sy, h)
            if not super_in_bounds(self.world, nsx, nsy):
                continue
            entry_idx = CROSS_MAP[h].get(a, None)
            if entry_idx is None:
                continue
            ex, ey = micro_coords(nsx, nsy, entry_idx)

            # we only enter if the entry cell is *known* traversable
            if self.world.is_known_traversable(ex, ey):
                avail.append(h)
        return avail

    def _choose_next_dir(self) -> Optional[int]:
        """Prefer unvisited adjacent supercell in right/straight/left/back order.
        If none unvisited is available, pick any available (for backtracking)."""
        order = [RIGHT(self.heading), self.heading, LEFT(self.heading), BACK(self.heading)]
        avail = self._available_dirs()
        # prefer unvisited
        for h in order:
            if h in avail:
                ns = self._neighbor_super(self.sx, self.sy, h)
                if ns not in self.visited_super:
                    return h
        # otherwise any available (backtrack/loop)
        return avail[0] if avail else None

    # --- high-level step (populate motion if needed, otherwise move one cell) ---
    def step(self) -> bool:
        """One algorithmic step: either move one base cell if queue has items,
        or decide/plans the next chunk (enter neighbor/backtrack). Returns False when finished."""
        if self.done:
            return False

        # if we still have motion queued, consume one base-cell step
        if self._tick_move():
            # keep moving
            return True

        # otherwise, at a vertex (we are at some micro anchor inside current supercell)
        # decide next direction using boundary-following rule
        h = self._choose_next_dir()
        if h is None:
            # nowhere to go: finished
            self.done = True
            return False

        # plan re-position to boundary anchor on current supercell
        allowed_here = free_micro_set(self.world, self.sx, self.sy)
        boundary = [a for a in ANCHOR[h] if a in allowed_here]
        if not boundary:
            # cannot position to this side (blocked), re-evaluate next time
            return False

        # choose anchor that's closest from current micro
        # (tiny BFS on 2x2 micro graph)
        best_anchor = None
        best_path = None
        for a in boundary:
            p = micro_move_shortest(allowed_here, self.cur_micro, a)
            if best_path is None or len(p) < len(best_path):
                best_path = p
                best_anchor = a

        # enqueue micro reposition inside current supercell
        cells: List[Tuple[int,int]] = []
        for idx in best_path[1:]:
            cx, cy = micro_coords(self.sx, self.sy, idx)
            cells.append((cx, cy))

        # cross boundary into neighbor
        nsx, nsy = self._neighbor_super(self.sx, self.sy, h)
        entry_idx = CROSS_MAP[h][best_anchor]
        ex, ey = micro_coords(nsx, nsy, entry_idx)
        if not self.world.is_known_traversable(ex, ey):
            # cannot cross now (insufficient knowledge or blocked)
            # consume nothing; try again next tick (sensing might reveal later)
            return False

        # finalize planning: enqueue
        cells.append((ex, ey))
        self._enqueue_cells(cells)

        # update DFS / tree
        prev_super = (self.sx, self.sy)
        next_super = (nsx, nsy)
        if next_super not in self.visited_super:
            self.parent[next_super] = prev_super
            edge = tuple(sorted([prev_super, next_super]))
            self.tree_edges.add(edge)

        # update state to neighbor (we'll physically step in via queue)
        self.sx, self.sy = nsx, nsy
        self.cur_micro = entry_idx
        self.heading = h

        # upon entry, cover current supercell's 2x2 (online, with current knowledge)
        # (we do NOT enqueue here to avoid mixing with the boundary crossing just queued)
        # Instead, once the crossing step is consumed and we're "standing" inside, next call
        # will see empty queue and enqueue micro coverage.
        if next_super not in self.visited_super:
            # After the crossing step executes, self.cur_micro is still entry_idx.
            # We'll schedule micro coverage on the next turn **after** motion completes.
            pass

        return True

    def maybe_schedule_micro_after_motion(self):
        """If we just arrived in a new supercell and queue is empty, cover it now."""
        if not self.queue:
            sc = (self.sx, self.sy)
            if sc not in self.visited_super:
                # we are physically inside (because queue empty) — cover now
                self._ensure_micro_coverage_on_entry()

# =====================
# Runner (animation wrapper)
# =====================
class Runner:
    def __init__(self, world: GridWorld):
        self.world = world
        self.alg = OnlineSTCClassic(world, START)
        self.path_pts: List[Tuple[int,int]] = [self.alg.pos]
        self.done = False
        self.budget = STEP_BUDGET or 10**12
        self.used = 0

    def step(self):
        if self.done: 
            return
        moved = self.alg.step()
        if not moved:
            # if no motion queued and algorithm may want to schedule micro-coverage now
            self.alg.maybe_schedule_micro_after_motion()
        else:
            # record new point (already appended in alg.route)
            self.used = self.alg.steps_taken
            self.path_pts = self.alg.route[:]
            if self.used >= self.budget:
                self.alg.done = True
        self.done = self.alg.done

# =====================
# Visualization (Pygame)
# =====================
class Viewer:
    def __init__(self, truth: np.ndarray):
        pygame.init()
        self.truth = truth
        self.H, self.W = truth.shape

        self.world = GridWorld(truth, START)
        self.runner = Runner(self.world)

        self.w_px = self.W*CELL
        self.h_px = self.H*CELL + 40
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Original Online STC (Dual-Graph Boundary Following)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.paused = False
        self.accumulator = 0.0
        self.step_interval = 1.0 / MOVES_PER_SECOND
        self.show_tree = True

    def draw_world(self):
        # cells (unknown and free both white)
        for y in range(self.H):
            for x in range(self.W):
                v_known = self.world.known[y, x]
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                if v_known == 2:
                    pygame.draw.rect(self.screen, COLORS['ob'], rect)
                elif v_known == 1:
                    if self.world.cleaned[y, x]:
                        pygame.draw.rect(self.screen, COLORS['spill_cleaned'], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS['spill'], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['free'], rect)

        # optional: show discovered dual-tree edges between supercell centers
        if self.show_tree:
            for (a,b) in self.runner.alg.tree_edges:
                ax, ay = a; bx, by = b
                acx = (2*ax+1)*CELL
                acy = (2*ay+1)*CELL
                bcx = (2*bx+1)*CELL
                bcy = (2*by+1)*CELL
                pygame.draw.line(self.screen, COLORS['edge'], (acx,acy), (bcx,bcy), 2)

        # path (black polyline)
        pts = [(x*CELL + CELL//2, y*CELL + CELL//2) for (x,y) in self.runner.path_pts]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, COLORS['path'], False, pts, 3)

    def draw_status(self):
        bar = pygame.Rect(0, self.H*CELL, self.w_px, 40)
        pygame.draw.rect(self.screen, COLORS['bg'], bar)

        visited = int(self.world.visited.sum())
        total_trav = int(np.sum(self.world.known != 2))  # known traversable so far
        spills_total_known = int(np.sum(self.world.known == 1))
        spills_clean = int(np.sum(self.world.cleaned))

        status = "DONE" if self.runner.done else "RUNNING"
        budget_txt = "∞" if STEP_BUDGET is None else str(STEP_BUDGET)
        txt = (
            f"steps {self.runner.alg.steps_taken} (budget {budget_txt}) | "
            f"visited {visited} | spills cleaned {spills_clean}/{spills_total_known} | "
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
                    elif e.key == pygame.K_n: self.runner.step()
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS): self.step_interval = max(1/240, self.step_interval - 0.01)
                    elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE): self.step_interval = min(1/1, self.step_interval + 0.01)
                    elif e.key == pygame.K_t: self.show_tree = not self.show_tree
                    elif e.key == pygame.K_s:
                        fname = f"online_stc_{int(pygame.time.get_ticks()/1000)}.png"
                        pygame.image.save(self.screen, fname)
                        print("Saved:", fname)
                    elif e.key == pygame.K_r:
                        self.world = GridWorld(self.truth, START)
                        self.runner = Runner(self.world)
                        self.accumulator = 0.0
                        self.paused = False

            if not self.paused and not self.runner.done:
                self.accumulator += dt
                while self.accumulator >= self.step_interval:
                    self.runner.step()
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
