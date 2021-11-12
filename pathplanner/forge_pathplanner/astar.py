import numpy as np
from collections import defaultdict
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class astar(object):
    def __init__(self, X, Y, H):
        self.X = X
        self.Y = Y
        self.H = H

    def make(self, start, goal, hcost):
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
            x1, y1, h1 = self.X[i1, j1], self.Y[i1, j1], self.H[i1, j1]
            x2, y2, h2 = self.X[i2, j2], self.Y[i2, j2], self.H[i2, j2]
            pt1 = np.array([x1, y1, h1])
            pt2 = np.array([x2, y2, h2])

            v = pt2[:2] - pt1[:2]
            h_avg = (pt2[2] + pt1[2]) / 2 * hcost

            return np.linalg.norm(v) + h_avg

        openset = {start}
        came_from = dict()
        gscore = defaultdict(lambda: 1e9)
        gscore[start] = 0

        fscore = defaultdict(lambda: 1e9)
        fscore[start] = h(start, goal)

        while len(openset) != 0:
            current = min(openset, key=lambda v: fscore[v])
            if current == goal:
                return self.reconstruct_path(came_from, current)
            openset.remove(current)
            for n in yield_neighbors(current):
                tentative_gscore = gscore[current] + h(current, n)
                if tentative_gscore < gscore[n]:
                    came_from[n] = current
                    gscore[n] = tentative_gscore
                    fscore[n] = gscore[n] + h(n, goal)
                    if n not in openset:
                        openset.add(n)

    def reconstruct_path(self, predecessors, current):
        """from a map of `predecessors` and a `current` point, trace
        backwards a path from the current point to the origin point"""
        total = [current]
        while current in predecessors:
            current = predecessors[current]
            total.append(current)
        return total[::-1]

    def get_line(self, path, dims=2):
        xyh = np.stack((self.X, self.Y, self.H), axis=-1)
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


def point2idx(point, X, Y):
    j = np.unravel_index(np.argmin(abs(X - point[0])), X.shape)[1]
    i = np.unravel_index(np.argmin(abs(Y - point[1])), Y.shape)[0]
    return (i, j)


if __name__ == "__main__":
    import surface
    from matplotlib import pyplot as plt
    from random import uniform

    xrange, yrange, step = (0, 80), (0, 40), 1.0
    X, Y = surface.generate_xy_grid(xrange, yrange, step)
    obstacles = surface.generate_obstacles(10, (15, 65), yrange, (4, 7), (1, 2.5))
    Hground = surface.place_obstacles(X, Y, obstacles)
    buffer, max_dh, max_d2h, min_h = 0.5, 0.2, 0.03, 0.5
    Hsheet = surface.get_optimal_grid(Hground, buffer, max_dh, max_d2h, min_h)

    # Plot 3d surface
    fig1 = plt.figure(figsize=(8, 10), tight_layout=True)
    ax1 = fig1.add_subplot((211), projection="3d")
    surface.plot_mpl3d(ax1, X, Y, Hground, Hsheet, zsquash=0.17, wireframe=True)

    # Get start, goal points
    get_rand_pt = lambda a, b, u, v: np.array([uniform(a, b), uniform(u, v)])
    # from one side...
    start = get_rand_pt(2, 5, 0, 20)
    # to another
    goal = get_rand_pt(74, 78, 20, 40)

    # get idx's of start, goal
    startij = point2idx(start, X, Y)
    goalij = point2idx(goal, X, Y)

    print("start={}, goal={}".format(startij, goalij))
    print("X={}, Y={}".format(X.shape, Y.shape))
    # solve Astar
    astarplanner = astar(X, Y, Hsheet)

    astar_paths = []
    hcosts = [0.0, 0.25, 2.0]
    linestyles = ["-", "--", ":"]
    for hcost in hcosts:
        path_tuples = astarplanner.make(startij, goalij, hcost=hcost)
        path = astarplanner.get_line(path_tuples)
        astar_paths.append(path)

    ax2 = fig1.add_subplot((212))
    surface.plot_mpl_2d(ax2, X, Y, Hsheet)
    ax2.scatter(start[0], start[1], marker="o", s=100, c="k", label="start")
    ax2.scatter(goal[0], goal[1], marker="*", s=100, c="r", label="goal")

    for c, path, ls in zip(hcosts, astar_paths, linestyles):
        ax2.plot(path[:, 0], path[:, 1], lw=3, ls=ls, label="h={}".format(c))

    ax2.legend()
    fig1.savefig("./astar.png", dpi=300)

    plt.show()
