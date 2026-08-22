"""
Grid-based global navigation.

Builds an occupancy grid from the detected tree trunks (inflated by the robot
radius) and runs A* to plan a collision-free path between base poses. This is
the standard mobile-robot "global planner": the robot only ever plans around
trees it has *perceived* with the lidar.
"""

from __future__ import annotations

import heapq

import numpy as np


class OccupancyGrid:
    def __init__(self, bounds, res, robot_radius):
        (x0, x1), (y0, y1) = bounds
        self.res = res
        self.x0, self.y0 = x0 - 1.0, y0 - 1.0
        self.nx = int(np.ceil((x1 - x0 + 2.0) / res))
        self.ny = int(np.ceil((y1 - y0 + 2.0) / res))
        self.robot_radius = robot_radius
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def to_cell(self, xy):
        return (int((xy[0] - self.x0) / self.res),
                int((xy[1] - self.y0) / self.res))

    def to_world(self, cell):
        return np.array([self.x0 + (cell[0] + 0.5) * self.res,
                         self.y0 + (cell[1] + 0.5) * self.res])

    def in_bounds(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def block_trees(self, trees, trunk_r=0.1, skip=None):
        """Mark cells within (trunk_r + robot_radius) of each tree as blocked.
        `skip` (an xy) leaves one tree open so we can approach/park at it."""
        self.grid[:] = False
        inflate = trunk_r + self.robot_radius
        for center in trees:
            if skip is not None and np.linalg.norm(center - skip) < 0.2:
                continue
            c = self.to_cell(center)
            rad = int(np.ceil(inflate / self.res))
            for i in range(max(0, c[0] - rad), min(self.nx, c[0] + rad + 1)):
                for j in range(max(0, c[1] - rad), min(self.ny, c[1] + rad + 1)):
                    if np.linalg.norm(self.to_world((i, j)) - center) < inflate:
                        self.grid[i, j] = True

    def free(self, c):
        return self.in_bounds(c) and not self.grid[c[0], c[1]]


def astar(grid: OccupancyGrid, start_xy, goal_xy):
    """8-connected A*; returns a list of world-xy waypoints or None."""
    start, goal = grid.to_cell(start_xy), grid.to_cell(goal_xy)
    if not grid.in_bounds(goal):
        return None
    # if start/goal landed in an inflated cell, nudge to nearest free cell
    start = _nearest_free(grid, start)
    goal = _nearest_free(grid, goal)
    if start is None or goal is None:
        return None

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    openh = [(0.0, start)]
    came, g = {}, {start: 0.0}
    while openh:
        _, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [grid.to_world(c) for c in path]
        for dx, dy in nbrs:
            nc = (cur[0] + dx, cur[1] + dy)
            if not grid.free(nc):
                continue
            step = np.hypot(dx, dy)
            ng = g[cur] + step
            if ng < g.get(nc, 1e18):
                g[nc] = ng
                f = ng + np.hypot(nc[0] - goal[0], nc[1] - goal[1])
                came[nc] = cur
                heapq.heappush(openh, (f, nc))
    return None


def _nearest_free(grid, cell, max_r=6):
    if grid.free(cell):
        return cell
    for r in range(1, max_r + 1):
        for i in range(cell[0] - r, cell[0] + r + 1):
            for j in range(cell[1] - r, cell[1] + r + 1):
                if grid.free((i, j)):
                    return (i, j)
    return None


def simplify(path, tol=0.05):
    """Drop collinear waypoints so the base drives smooth straight legs."""
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        a, b, c = out[-1], path[i], path[i + 1]
        # keep b only if it changes direction
        if abs(np.cross(b - a, c - a)) > tol:
            out.append(b)
    out.append(path[-1])
    return out
