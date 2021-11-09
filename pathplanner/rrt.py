from p2 import optimal_path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import cm
from rrtpp import RRTStar
from scipy.interpolate import RegularGridInterpolator
import networkx as nx
from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


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

    def get_end_node(self, xgoal):
        """get the 'end' node, given a tree and an end point. The end node is either the point itself,
        if a path to it is possible, or the closest node in the tree to the end point."""
        vnearest = self.nearest_nodes(xgoal)[0]
        v = max(self.T.nodes) + 1
        newcost = self.calc_cost(vnearest, xgoal)
        self.T.add_node(v, point=xgoal, cost=newcost)
        self.T.add_edge(vnearest, v, cost=newcost)
        return v

    def make(self, xstart, xgoal, N, r_rewire) -> tuple:
        """Make RRT star with `N` points from xstart to xgoal.
        Returns the tree, the start node, and the end node."""
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
        vend = self.get_end_node(xgoal)
        self.vstart = vstart
        self.vend = vend
        return vstart, vend

    def calc_cost(self, v, x=None):
        if x is not None:
            # points in R2
            p1 = self.T.nodes[v]["point"]
            p2 = x
            hfunc = lambda p: self.interp(p, method="nearest")
            # hvals = np.apply_along_axis(hfunc, 1, r2line)
            r2dist = np.linalg.norm(p2 - p1)
            hcost = hfunc(p2) - hfunc(p1)
            return self.T.nodes[v]["cost"] + self.g * hcost + r2dist
        else:
            return self.T.nodes[v]["cost"]

    def get_least_cost_path(self, vstart, vend, n_interp=20):
        path = nx.shortest_path(
            self.T,
            source=vstart,
            target=vend,
            weight="cost",
        )
        paths, costs = [], []
        for i, v in enumerate(path[:-1]):
            p1 = self.T.nodes[path[i]]["point"]
            p2 = self.T.nodes[path[i + 1]]["point"]
            path_interp = np.linspace(p1, p2, n_interp)
            for j, p in enumerate(path_interp[:-1]):
                paths.append((path_interp[j], path_interp[j + 1]))
            c1 = self.T.nodes[path[i]]["cost"]
            c2 = self.T.nodes[path[i + 1]]["cost"]
            cost = np.linspace(c1, c2, n_interp)
            for j, c in enumerate(cost[:-1]):
                costs.append(c)
        return paths, costs

    def get_linecoll(self):
        linecol = []
        costs = []
        for e1, e2 in self.T.edges():
            p1 = self.T.nodes[e1]["point"]
            p2 = self.T.nodes[e2]["point"]
            linecol.append((p1, p2))
            costs.append(self.T[e1][e2]["cost"])
        costs = np.array(costs)
        norm = Normalize(costs.min(), costs.max())
        costs = norm(costs)
        return linecol, costs

    def get_3d_linecoll(self):
        linecol = []
        costs = []
        for e1, e2 in self.T.edges:
            p1 = self.T.nodes[e1]["point"]
            p2 = self.T.nodes[e2]["point"]
            line2 = np.linspace([p1[0], p1[1]], [p2[0], p2[1]], 100)
            interpfn = lambda t: srrt.get_h(t[0], t[1], method="linear")
            lineh = np.apply_along_axis(interpfn, 1, line2)
            line3 = np.concatenate((line2, lineh), axis=1)
            linecol.append(line3)
            costs.append(self.T[e1][e2]["cost"])
        costs = np.array(costs)
        norm = Normalize(costs.min(), costs.max())
        costs = norm(costs)
        return linecol, costs


@mlab.show
def plot_mayavi(X, Y, H, Hnew, srrt):
    mlab.mesh(X, Y, Hnew, opacity=0.4, color=(0.6, 0.6, 0.6))
    mlab.mesh(X, Y, H, colormap="terrain")
    linecol, costs = srrt.get_3d_linecoll()
    colors = cm.get_cmap("inferno")(costs)
    for lc, c in zip(linecol, colors):
        c = np.squeeze(c)
        co = c[0], c[1], c[2]
        mlab.plot3d(lc[:, 0], lc[:, 1], lc[:, 2], color=co)


def plot_mpl3d(X, Y, H, Hnew, srrt: SheetRRT, start, end):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.45, 1]))
    ax.plot_wireframe(X, Y, Hnew, alpha=0.4, cmap=cm.get_cmap("bone"), zorder=2)
    ax.plot_surface(X, Y, H, cmap=cm.get_cmap("Blues"), zorder=1)
    linecol, costs = srrt.get_3d_linecoll()
    lc = Line3DCollection(linecol, colors=cm.get_cmap("viridis")(costs))
    ax.add_collection(lc)
    ax.scatter(start[0], start[1], s=100, marker=",", zorder=5, c="b")
    ax.scatter(end[0], end[1], s=100, marker="*", zorder=5, c="r")


def plot_mpl_2d(X, Y, H, srrt: SheetRRT, start, end):
    fig = plt.figure()
    ax = fig.add_subplot()
    linecol, costs = srrt.get_linecoll()
    lc = LineCollection(linecol, colors=cm.get_cmap("viridis")(costs))
    ax.add_collection(lc)
    points = np.array([srrt.T.nodes[v]["point"] for v in srrt.T])
    ax.scatter(points[:, 0], points[:, 1], s=6, c="k")
    ax.contour(X, Y, H, cmap=cm.get_cmap("coolwarm"), levels=14, linewidths=1)
    ax.scatter(start[0], start[1], s=100, marker=",", zorder=5, c="b")
    ax.scatter(end[0], end[1], s=100, marker="*", zorder=5, c="r")


if __name__ == "__main__":
    min_acc_climb = 0.1
    min_climb = 0.5
    # minimum distance
    buffer = 0.5
    min_altitude = 1.5
    # amount to penalize height when drawing a* path
    astar_heightcost = 0.001

    xrange, yrange = (0, 30), (0, 30)
    radrange = (2, 5)
    hrange = (1, 10)

    X, Y = optimal_path.generate_xy_grid(xrange, yrange, 0.5)
    # obstacles = optimal_path.generate_obstacles(8, xrange, yrange, radrange, hrange)

    obstacles = [
        [np.array([0, 15]), 8, 10],
        [np.array([30, 15]), 8, 10],
        [np.array([15, 0]), 20, 5],
        [np.array([15, 0]), 6, 7.5],
    ]

    H = optimal_path.place_obstacles(X, Y, obstacles)
    Hnew = optimal_path.get_optimal_grid(
        H, buffer, min_climb, min_acc_climb, min_altitude
    )
    # start = np.array(
    #     [np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max())]
    # )
    # end = np.array(
    #     [np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max())]
    # )

    start = np.array([1, 1])
    end = np.array([29, 29])

    fig, ax = plt.subplots()
    ax.contour(X, Y, H, cmap=cm.get_cmap("coolwarm"), levels=14, linewidths=1)
    ax.scatter(start[0], start[1], s=100, marker=",", zorder=5, c="b")
    ax.scatter(end[0], end[1], s=100, marker="*", zorder=5, c="r")
    n_ = 50
    for n in range(5):
        n_ += 50 * n
        srrt = SheetRRT(X.shape, X, Y, Hnew, g=8)
        vstart, vend = srrt.make(start, end, n_, 7)
        lcpath, lccosts = srrt.get_least_cost_path(vstart, vend)
        lc2 = LineCollection(lcpath, colors=cm.get_cmap("coolwarm")(lccosts))
        ax.add_collection(lc2)

    # plot_mpl3d(X, Y, H, Hnew, srrt, start, end)
    # plot_mpl_2d(X, Y, Hnew, srrt, start, end)
    # plot_mayavi(X, Y, H, Hnew, srrt)
    plt.show()
