import numpy as np
from collections import defaultdict


def astar(X, Y, H, start, goal, dist_heur="euclidean", heur_cost=0.1):
    """Astar function

    `X`, `Y`, `H` are 2d arrays indicating x, y, h scalars. `start`, `goal` are
    tuples; they are the index of a start point and an end point in those `X`, `Y`
    arrays.

    dist_heur is a string that can take values `"euclidean"`, `"least_diff"`, or `"lowest"`
    to generate the euclidean path in R3, the pairwise least diff path, or the lowest path,
    respectively.

    `heur_cost` is not required for `"euclidean"`, but is required for `"least_diff"` and
    `"lowest"` paths. It corresponds to the cost that should be added to the R2 euclidean
    distance when using those heuristics.
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
        # R3 distance
        if dist_heur == "euclidean":
            pt1 = np.array((X[tuple(ij1)], Y[tuple(ij1)], H[tuple(ij1)]))
            pt2 = np.array((X[tuple(ij2)], Y[tuple(ij2)], H[tuple(ij2)]))
            d = np.linalg.norm(pt2 - pt1)
        # R2 distance + height diff between i, j
        elif dist_heur == "least_diff":
            pt1 = np.array((X[tuple(ij1)], Y[tuple(ij1)]))
            pt2 = np.array((X[tuple(ij2)], Y[tuple(ij2)]))
            d = np.linalg.norm(pt2 - pt1) + heur_cost * np.abs(H[ij1] - H[ij2])
        # R2 distance + absolute height of destination
        elif dist_heur == "lowest":
            pt1 = np.array((X[tuple(ij1)], Y[tuple(ij1)]))
            pt2 = np.array((X[tuple(ij2)], Y[tuple(ij2)]))
            d = np.linalg.norm(pt2 - pt1) + heur_cost * np.abs(H[ij2])
        return d

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
    return total
