import heapq

_DIRS = (
    (0, -1, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (1, 0, 1.0),
    (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
)


class NavGrid:
    __slots__ = ("cell_size", "cols", "rows", "_blocked")

    def __init__(self, walls, frame_w, frame_h, cell_size):
        self.cell_size = cell_size
        self.cols = max(1, int(frame_w // cell_size))
        self.rows = max(1, int(frame_h // cell_size))
        blocked = set()
        for wall in walls:
            c0 = max(0, int(wall[0] // cell_size))
            r0 = max(0, int(wall[1] // cell_size))
            c1 = min(self.cols - 1, int(wall[2] // cell_size))
            r1 = min(self.rows - 1, int(wall[3] // cell_size))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    blocked.add((c, r))
        self._blocked = frozenset(blocked)

    def _is_free(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows and (c, r) not in self._blocked

    def _px_to_cell(self, px, py):
        return (
            max(0, min(self.cols - 1, int(px // self.cell_size))),
            max(0, min(self.rows - 1, int(py // self.cell_size))),
        )

    def _nearest_free(self, cell, max_r=5):
        c, r = cell
        for radius in range(1, max_r + 1):
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if abs(dc) == radius or abs(dr) == radius:
                        nc, nr = c + dc, r + dr
                        if self._is_free(nc, nr):
                            return (nc, nr)
        return None

    def find_next_step(self, start_px, goal_px, max_iters=200):
        start = self._px_to_cell(start_px[0], start_px[1])
        goal  = self._px_to_cell(goal_px[0],  goal_px[1])

        if not self._is_free(*start):
            start = self._nearest_free(start)
            if start is None:
                return None
        if not self._is_free(*goal):
            goal = self._nearest_free(goal)
            if goal is None:
                return None

        if start == goal:
            return goal_px

        open_heap = [(0.0, 0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        counter = 1

        while open_heap and len(closed) < max_iters:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            if cur == goal:
                node = cur
                while came_from.get(node) != start and node in came_from:
                    node = came_from[node]
                return (
                    (node[0] + 0.5) * self.cell_size,
                    (node[1] + 0.5) * self.cell_size,
                )

            cc, cr = cur
            for dc, dr, cost in _DIRS:
                nc, nr = cc + dc, cr + dr
                nb = (nc, nr)
                if nb in closed or not self._is_free(nc, nr):
                    continue
                if dc != 0 and dr != 0:
                    if not self._is_free(cc + dc, cr) or not self._is_free(cc, cr + dr):
                        continue
                ng = g_score[cur] + cost
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb] = ng
                    came_from[nb] = cur
                    h = max(abs(nc - goal[0]), abs(nr - goal[1]))
                    heapq.heappush(open_heap, (ng + h, counter, nb))
                    counter += 1

        return None


def pixel_dir_to_wasd(player_pos, target_pos, threshold=5.0):
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    keys = ""
    if dy < -threshold:
        keys += "W"
    elif dy > threshold:
        keys += "S"
    if dx < -threshold:
        keys += "A"
    elif dx > threshold:
        keys += "D"
    return keys
