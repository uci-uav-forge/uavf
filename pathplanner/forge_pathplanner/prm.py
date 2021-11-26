import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator
import networkx as nx
from pykdtree import kdtree
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm, LogNorm, SymLogNorm
from mayavi import mlab
from math import ceil


class PRM(object):
    def __init__(self, X, Y, H, n, hweight):
        # the surface given by X, Y, H
        self.H = H
        self.interp = RegularGridInterpolator((X[0, :], Y[:, 0]), H.T)
        self.xrange = X.min(), X.max()
        self.yrange = Y.min(), Y.max()
        self.xyh, self.prmGree, self.prmG = self.make(n, hweight)

    def sample(self, N):
        """Sample `N` random points from H.

        Parameters
        ----------
        N : int
            number of points to sample

        Returns
        -------
        np.ndarray
            Nx3 array of points in R3 that are on the sheet
        """
        x = np.random.uniform(*self.xrange, size=(N,))
        y = np.random.uniform(*self.yrange, size=(N,))
        return np.column_stack((x, y))

    def make(self, n, hweight):
        """Make PRM over the space using Delaunay Triangulation. The function
        does these things:

        1. make `n` points in R2 and interpolate to R3 using the height map.
        2. create a *2D* delaunay triangulation of the points to get connectivity.
        3. make a graph of the connectivity of the triangulation
        4. make a 3-dimensional KDtree of the points

        Parameters
        ----------
        n : int
            number of sample points for the PRM
        hweight : float, hweight >= 0
            amount to punish changes in height. see `edge_cost`.

        Returns
        -------
        tuple of (np.ndarray, kdtree.KDTree, nx.Graph)
            the points sampled as an Nx3 array `xyh`, the KDTree of those points,
            and the networkx graph of the probabilistic roadmap.
        """
        prmG = nx.Graph()
        xy = self.sample(n)
        # add points. vertexes to points are keys of `T`
        for i, point in enumerate(xy):
            prmG.add_node(i, point=point)
        # Delaunay in 2-D -> sheet connectivity
        dt = Delaunay(xy)
        for s in dt.simplices:
            # iterate through pairs
            for s1, s2 in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                # add edges for each triangle
                prmG.add_edge(s1, s2, cost=self.edge_cost(xy, s1, s2, hweight))
        # draw a KD tree for each xyh
        tree = kdtree.KDTree(xy)
        return xy, tree, prmG

    def r2r3(self, p, method="linear"):
        """Project a point from (x, y) space (R^2) -> (x, y, h) space (R^3) by
        interpolation to find `h`

        Parameters
        ----------
        xy : np.ndarray
            length-2 array of points in R^2
        method : str, optional
            method for interpolation. Can be "nearest" to just get the height of the nearest
            point or "linear" to linearly interpolate height., by default "linear"

        Returns
        -------
        np.ndarray
            length-3 array of points in R^3
        """
        pnew = np.empty((p.shape[0], 3), dtype=p.dtype)
        pnew[:, :2] = p
        try:
            h = self.interp(p, method=method)
        except ValueError:
            xminmax = X.min(), X.max()
            yminmax = Y.min(), Y.max()
            pxminmax = p[:, 0].min(), p[:, 0].max()
            pyminmax = p[:, 1].min(), p[:, 1].max()

            raise ValueError(
                "Point seems to be outside of bounds. xminmax={}, yminmax={}, px_minmax={}, py_minmax={}".format(
                    xminmax, yminmax, pxminmax, pyminmax
                )
            )
        pnew[:, 2] = h
        return pnew

    def compute_shortest_path(self, xstart, xend):
        """Determine the shortest (by path weight) path from `xstart` to `xend`.
        This computes a roadmap in the form of a list of vertices of the class's
        `prmG` object.

        Parameters
        ----------
        xstart : np.ndarray
            (2,) array as start point in (x, y)
        xend : np.ndarray
            (2,) array as end point in (x, y)

        Returns
        -------
        list of int
            list of vertices of `self.prmG` corresponding to shortest path.
        """
        # get nearest nodes on PRM
        _, vnear_start = self.prmGree.query(np.atleast_2d(xstart), k=1)
        _, vnear_end = self.prmGree.query(np.atleast_2d(xend), k=1)
        # get shortest path
        path = nx.shortest_path(
            self.prmG,
            source=vnear_start[0],
            target=vnear_end[0],
            weight="cost",
            method="bellman-ford",
        )
        return path

    def get_path_xy(self, path):
        return np.array([self.prmG.nodes[v]["point"] for v in path])

    def get_prm_lc(self, cmap="viridis", gamma=0.4):
        points, costs = [], []
        for e1, e2 in self.prmG.edges:
            p1 = self.prmG.nodes[e1]["point"][:2]
            p2 = self.prmG.nodes[e2]["point"][:2]
            points.append((p1, p2))
            cost = self.prmG[e1][e2]["cost"]
            costs.append(cost)
        norm = PowerNorm(gamma, vmin=min(costs), vmax=max(costs))
        colors = cm.get_cmap(cmap)(norm(costs))
        return LineCollection(points, colors=colors, linewidths=0.5)

    def edge_cost(self, xy, s1, s2, hweight=1.0):
        """Compute cost between vertices `s1` and `s2`. `s1` and s2` must be indices
        to rows of an Nx3 collection of points `xyh`. hweight is a measure of how
        significant we want `h` to be in the cost; a higher `hweight` means that positive
        changes in `h` are punished more severely.

        Parameters
        ----------
        xyh : Nx3 np.ndarray
            collection of points in R3
        s1 : int
            index to point xyh[s1]
        s2 : int
            index to point xyh[s2]
        hweight : float, optional
            how much to weight h. A value hweight=0 corresponds to just computing costs as
            the euclidean distance in R2 and not considering the height at all; a value
            hweight=1.0 corresponds to a cost over the euclidean distance in R3. In general,
            a higher value punishes `h` more. by default 1.0

        Returns
        -------
        float
            scalar cost
        """
        p1 = xy[s1]
        p2 = xy[s2]

        # points in R2
        nsteps = ceil(np.linalg.norm(p2 - p1))
        points = np.linspace(p1, p2, nsteps)
        xyh = self.r2r3(points, method="nearest")
        xyh *= np.array([1.0, 1.0, hweight])
        # diff
        diff = np.diff(xyh, n=1, axis=0)
        c = np.linalg.norm(diff, axis=1).sum()
        # norm sum
        # c = np.linalg.norm(xyh, axis=1).sum()
        return c


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

    # make PRM
    prm = PRM(X, Y, Hsheet, 1000, 50)
    vpath = prm.compute_shortest_path(start, goal)
    ppath = prm.get_path_xy(vpath)

    ax2 = fig1.add_subplot((212))
    surface.plot_mpl_2d(ax2, X, Y, Hsheet)
    ax2.plot(ppath[:, 0], ppath[:, 1], lw=3, c="k")
    lc = prm.get_prm_lc(gamma=0.25)
    ax2.add_collection(lc)

    ax2.legend()
    fig1.savefig("./prm.png", dpi=300)

    plt.show()
