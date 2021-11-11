import numpy as np
from collections import defaultdict
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def astar(X, Y, H, start, goal, hcost=0.1):
    """Astar function

    `X`, `Y`, `H` are 2d arrays indicating x, y, h scalars. `start`, `goal` are
    tuples; they are the index of a start point and an end point in those `X`, `Y`
    arrays.

    `hcost` adjusts how much the planner takes into account `h`
    """

    def yield_neighbors(ij):
        ij = np.array(ij)
        xmax = np.array(X.shape)
        zeros = np.array((0, 0))
        # neighbors
        for n in np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]):
            # edge of wall
            point = ij + n
            edge = np.any(point >= xmax) or np.any(point <= zeros)
            if not edge:
                yield tuple(point)

    def h(ij1, ij2):
        """distance heuristic on indices of X, Y."""
        i1, j1 = ij1
        i2, j2 = ij2
        x1, y1, h1 = X[i1, j1], Y[i1, j1], H[i1, j1]
        x2, y2, h2 = X[i2, j2], Y[i2, j2], H[i2, j2]
        pt1 = np.array([x1, y1, h1])
        pt2 = np.array([x2, y2, h2])
        v = np.array([1.0, 1.0, hcost])
        d = (pt2 + pt1 / 2) * v
        return np.linalg.norm(d)

    openset = {start}
    came_from = dict()
    gscore = defaultdict(lambda: 1e9)
    gscore[start] = 0

    fscore = defaultdict(lambda: 1e9)
    fscore[start] = h(start, goal)

    while len(openset) != 0:
        current = min(openset, key=lambda v: fscore[v])
        if current == goal:
            return reconstruct_path(came_from, current)
        openset.remove(current)
        for n in yield_neighbors(current):
            tentative_gscore = gscore[current] + h(current, n)
            if tentative_gscore < gscore[n]:
                came_from[n] = current
                gscore[n] = tentative_gscore
                fscore[n] = gscore[n] + h(n, goal)
                if n not in openset:
                    openset.add(n)


def reconstruct_path(predecessors, current):
    """from a map of `predecessors` and a `current` point, trace
    backwards a path from the current point to the origin point"""
    total = [current]
    while current in predecessors:
        current = predecessors[current]
        total.append(current)
    return total[::-1]


def get_line(X, Y, H, path, dims=2, c="k"):
    xyh = np.stack((X, Y, H), axis=-1)
    lc = []
    for i, _ in enumerate(path):
        i, j = path[i]
        if dims == 2:
            xy = xyh[i, j, :2]
            lc.append(xy)
        if dims == 3:
            xy = xyh[i, j, :]
            lc.append(xy)
    return np.array(lc)