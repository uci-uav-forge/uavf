import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib import cm
from rrtpp import RRTStar
import networkx as nx
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from math import ceil


class SheetRRT(RRTStar):
    def __init__(self, wshape, X, Y, H, g):
        world = np.zeros(wshape)
        super().__init__(world)
        self.interp = RegularGridInterpolator((X[0, :], Y[:, 0]), H.T)
        self.X, self.Y = X, Y
        # get max and min X, Y
        self.xmin, self.xmax = self.X[0, :].max(), self.X[0, :].min()
        self.ymin, self.ymax = self.Y[:, 0].max(), self.Y[:, 0].min()
        self.g = g

    def get_h(self, x, y, method="nearest"):
        return self.interp(np.array([x, y]), method=method)

    def get_pos(self):
        pos = {}
        for v in self.T:
            pos[v] = self.T.nodes[v]["point"]
        return pos

    def sample_all_free(self):
        x = np.random.uniform(self.xmin, self.xmax)
        y = np.random.uniform(self.ymin, self.ymax)
        return np.array((x, y))

    def get_end_node(self, xgoal, r_rewire):
        """get the 'end' node, given a tree and an end point. The end node is either the point itself,
        if a path to it is possible, or the closest node in the tree to the end point."""
        vnearest = self.near(xgoal, r_rewire)
        # get least cost near xgoal
        vn_best, vn_cost = min(
            [(vn, self.calc_cost(vn, xgoal)) for vn in vnearest], key=lambda t: t[1]
        )
        # assign new vertex
        v = max(self.T.nodes) + 1
        self.T.add_node(v, point=xgoal, cost=vn_cost)
        self.T.add_edge(vn_best, v, cost=vn_cost)
        return v

    def make(self, xstart, xgoal, N, r_rewire) -> tuple:
        self.reset_T()
        i = 1
        vstart = 0
        self.T.add_node(vstart, point=xstart, cost=0.0)
        if self.pbar:
            pbar = tqdm(total=N, desc="tree")
        while i < N:
            xnew = self.sample_all_free()
            vnearest = self.nearest_nodes(xnew)
            xnearest = self.T.nodes[vnearest[0]]["point"]
            vnew = i
            vmin = vnearest[0]
            xmin = xnearest
            cmin = self.calc_cost(vmin, xnew)
            # vnear contains at least vmin
            vnear = self.near(xnew, r_rewire)
            vnear.append(vmin)
            # search for a lesser cost vertex in connection radius
            for vn in vnear:
                xn = self.T.nodes[vn]["point"]
                cost = self.calc_cost(vn, xnew)
                if cost < cmin:
                    xmin = xn
                    cmin = cost
                    vmin = vn
            # add new vertex and edge connecting min-cost vertex with new point
            self.T.add_node(vnew, point=xnew, cost=cmin)
            self.T.add_edge(vmin, vnew, cost=cmin)
            # rewire the tree
            for vn in vnear:
                xn = self.T.nodes[vn]["point"]
                cn = self.calc_cost(vn)
                c = self.calc_cost(vn, xnew)
                if c < cn:
                    parent = self.get_parent(vn)
                    if parent is not None:
                        self.T.remove_edge(parent, vn)
                        self.T.add_edge(vnew, vn, cost=c)
            if self.pbar:
                pbar.update(1)
            i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()

        self.write_dist_edges()
        vend = self.get_end_node(xgoal, r_rewire)
        self.vstart = vstart
        self.vend = vend
        return vstart, vend

    def r2r3(self, p, method="linear"):
        pnew = np.empty((p.shape[0], 3), dtype=p.dtype)
        pnew[:, :2] = p
        try:
            h = self.interp(p, method=method)
        except ValueError:
            xminmax = X.min(), X.max()
            yminmax = Y.min(), Y.max()
            pminmax = np.ptp(p, axis=0)

            raise ValueError(
                "Point seems to be outside of bounds. xminmax={}, yminmax={}, point_minmax={}".format(
                    xminmax, yminmax, pminmax
                )
            )
        pnew[:, 2] = h
        return pnew

    def calc_cost(self, v, x=None):
        if x is not None:
            # points in R2
            p1 = self.T.nodes[v]["point"]
            p2 = x
            nsteps = ceil(np.linalg.norm(p2 - p1) * 1.1)
            points = np.linspace(p1, p2, nsteps)
            xyh = self.r2r3(points, method="nearest")
            xyh *= np.array([1.0, 1.0, self.g])
            # diff
            diff = np.diff(xyh, n=1, axis=0)
            c = np.linalg.norm(diff, axis=1).sum()
            # norm sum
            # c = np.linalg.norm(xyh, axis=1).sum()

            return self.T.nodes[v]["cost"] + c
        else:
            return self.T.nodes[v]["cost"]

    def get_path(self, vstart, vend):
        path = nx.shortest_path(self.T, source=vstart, target=vend, weight="cost")
        return np.array([self.T.nodes[p]["point"] for p in path])

    def get_tree_linecol(self, cmap="viridis"):
        lines, costs = [], []
        for e1, e2 in self.T.edges:
            p1, p2 = self.T.nodes[e1]["point"], self.T.nodes[e2]["point"]
            lines.append((p1, p2))
            cost = self.T[e1][e2]["cost"]
            costs.append(cost)
        norm = Normalize(min(costs), max(costs))
        return LineCollection(
            lines, colors=cm.get_cmap(cmap)(norm(costs)), linewidths=0.5
        )


if __name__ == "__main__":
    import surface
    from matplotlib import pyplot as plt
    from random import uniform

    xrange, yrange, step = (0, 80), (0, 40), 1.0
    X, Y = surface.generate_xy_grid(xrange, yrange, step)
    obstacles = surface.generate_obstacles(14, (10, 70), yrange, (4, 7), (1, 2.5))
    Hground = surface.place_obstacles(X, Y, obstacles)
    buffer, max_dh, max_d2h, min_h = 0.5, 0.2, 0.03, 0.5
    Hsheet = surface.get_optimal_grid(Hground, buffer, max_dh, max_d2h, min_h, step)

    # Plot 3d surface
    fig1 = plt.figure(figsize=(14, 8), tight_layout=True)
    ax1 = fig1.add_subplot((211), projection="3d")
    surface.plot_mpl3d(ax1, X, Y, Hground, Hsheet, zsquash=0.17, wireframe=True)

    # Get start, goal points
    get_rand_pt = lambda a, b, u, v: np.array([uniform(a, b), uniform(u, v)])
    # from one side...
    start = get_rand_pt(0, 5, 0, 20)
    # to another
    goal = get_rand_pt(75, 80, 20, 40)

    # Make RRT object and solve path
    srrt = SheetRRT(X.shape, X, Y, Hsheet, g=1000)
    vstart, vend = srrt.make(start, goal, 400, 18)

    ax2 = fig1.add_subplot((212))
    # 2D make and plot start, end
    ax2.scatter(start[0], start[1], marker="o", s=100, c="k", label="start")
    ax2.scatter(goal[0], goal[1], marker="*", s=100, c="r", label="goal")

    # 2D Make and plot lines
    surface.plot_mpl_2d(ax2, X, Y, Hsheet)
    path = srrt.get_path(vstart, vend)
    ax2.plot(path[:, 0], path[:, 1], lw=3, c="k")
    lc = srrt.get_tree_linecol()
    ax2.add_collection(lc)
    ax2.legend()

    fig1.savefig("./rrt.png", dpi=300)
    plt.show()
