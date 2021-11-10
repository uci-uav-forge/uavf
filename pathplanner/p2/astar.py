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
    return total


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


if __name__ == "__main__":
    import optimal_path
    from matplotlib import pyplot as plt
    from matplotlib import cm

    xrange, yrange, step = (0, 50), (0, 50), 1
    X, Y = optimal_path.generate_xy_grid(xrange, yrange, step)
    obstacles = optimal_path.generate_obstacles(3, xrange, yrange, (5, 12), (2, 8))
    Hground = optimal_path.place_obstacles(X, Y, obstacles)
    dh, d2h, buffer, min_h = 0.4, 0.05, 1.0, 1.5
    Hsheet = optimal_path.get_optimal_grid(Hground, buffer, dh, d2h, min_h)
    # plot 2d
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    optimal_path.plot_mpl_2d(ax1, X, Y, Hsheet)

    # plot 3d
    fig2 = plt.figure(tight_layout=True)
    ax2 = fig2.add_subplot(projection="3d")
    optimal_path.plot_mpl3d(ax2, X, Y, Hground, Hsheet, wireframe=True)

    start = 1, 1
    goal = (X.shape[0] - 1, X.shape[1] - 1)

    heurs = ("euclideanR3", "least_diff", "lowest", "euclideanR2")
    heur_cost = [0, 1.0, 2.0, 4.0, 8.0, 10.0]

    for i, hc in enumerate(heur_cost):
        print(i)
        path = astar(X, Y, Hsheet, start, goal, hcost=hc)
        lc = get_line(X, Y, Hsheet, path)
        ax1.plot(lc[:, 0], lc[:, 1], label="H Cost={}".format(hc))
    ax1.legend()
    plt.show()
