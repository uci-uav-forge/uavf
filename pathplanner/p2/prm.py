import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator
import networkx as nx
from pykdtree import kdtree
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mayavi import mlab


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
        interp_f = lambda xy: self.interp(xy, method="linear")
        xy = np.stack((x, y), axis=-1)
        h = np.apply_along_axis(interp_f, axis=1, arr=xy)
        return np.concatenate((xy, h), axis=1)

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
        xyh = self.sample(n)
        # add points. vertexes to points are keys of `T`
        for i, point in enumerate(xyh):
            prmG.add_node(i, point=point)
        # Delaunay in 2-D -> sheet connectivity
        dt = Delaunay(xyh[:, :2])
        for s in dt.simplices:
            # iterate through pairs
            for s1, s2 in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                # add edges for each triangle
                prmG.add_edge(s1, s2, cost=self.edge_cost(xyh, s1, s2, hweight))
        # draw a KD tree for each xyh
        tree = kdtree.KDTree(xyh)
        return xyh, tree, prmG

    def r2r3(self, xy, method="linear"):
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

        xyh = np.empty((3,), dtype=xy.dtype)
        xyh[:2] = xy
        xyh[2] = self.interp(xy, method=method)
        return xyh

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
        # R2 -> R3
        xstart = np.atleast_2d(self.r2r3(xstart))
        xend = np.atleast_2d(self.r2r3(xend))
        # get nearest nodes on PRM
        _, vnear_start = self.prmGree.query(xstart, k=1)
        _, vnear_end = self.prmGree.query(xend, k=1)
        # get shortest path
        path = nx.shortest_path(
            self.prmG,
            source=vnear_start[0],
            target=vnear_end[0],
            weight="cost",
            method="bellman-ford",
        )
        return path

    def draw_path(self, path, dims=2, costs=False, color="r", lw=3, cmap="plasma"):
        """Get path from list of ordered nodes `path` and make a matplotlib LineCollection
        from it.

        >>> path = [4, 6, 20, 1, 3]
        >>> lc = prm_object.draw_path(path)
        >>> ax.add_collection(lc)

        Parameters
        ----------
        path : list of int
            nodes of `prmG` in the order that they are traversed. Subsequent nodes must have
            an edge in `prmG` or this will throw an error
        dims : int, optional
            dimensions to draw the line. For 2d plots, pass `2`, for 3d plots, pass `3`, by default 2
        costs : bool, optional
            whether to color the lines by cost or not, by default False
        color : str, optional
            if we aren't coloring by cost, what color the lines should be, by default "r"
        lw : int, optional
            line width, by default 3
        cmap : str, optional
            colormap to use for coloring lines by cost, by default "plasma"

        Returns
        -------
        (LineCollection or Line3DCollection, list)
            the line collection object and the list of lines.
        """
        linecol = []
        lcosts = []
        for i, _ in enumerate(path[:-1]):
            v1 = path[i]
            v2 = path[i + 1]
            if dims == 2:
                p1 = self.prmG.nodes[v1]["point"][:2]
                p2 = self.prmG.nodes[v2]["point"][:2]
            elif dims == 3:
                p1 = self.prmG.nodes[v1]["point"] + np.array([0, 0, 0.1])
                p2 = self.prmG.nodes[v2]["point"] + np.array([0, 0, 0.1])
            linecol.append((p1, p2))
            lcosts.append(self.prmG[v1][v2]["cost"])
        # normalize cost vals for colormap
        norm = Normalize(min(lcosts), max(lcosts))
        costnorm = norm(lcosts)

        # if R2, return LineCollection, if R3 return Line3DCollection
        if dims == 2 and costs == False:
            lc = LineCollection(linecol, linewidths=lw, colors=color)
        if dims == 2 and costs == True:
            lc = LineCollection(
                linecol, linewidths=lw, colors=cm.get_cmap(cmap)(costnorm)
            )
        if dims == 3 and costs == False:
            lc = Line3DCollection(linecol, linewidths=lw, colors=color)
        if dims == 3 and costs == True:
            lc = Line3DCollection(
                linecol, linewidths=lw, colors=cm.get_cmap(cmap)(costnorm)
            )
        return lc, linecol

    def edge_cost(self, xyh, s1, s2, hweight=1.0):
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
        p1 = xyh[s1]
        p2 = xyh[s2]
        # we can scale by h distance
        q = np.array([1, 1, hweight])
        d = p2 - p1
        cost = np.linalg.norm(d * q)
        return cost


def get_rand_pt(xrange, yrange):
    x = np.random.uniform(*xrange)
    y = np.random.uniform(*yrange)
    return np.array((x, y))


@mlab.show
def plot_mayavi(X, Y, Hground, Hsheet, lc, shrink_z=0.2):
    ground = mlab.mesh(X, Y, Hground, opacity=1.0, colormap="bone")
    ground.actor.actor.scale = (1.0, 1.0, shrink_z)
    sheet = mlab.mesh(X, Y, Hsheet, opacity=0.6, colormap="ocean")
    sheet.actor.actor.scale = (1.0, 1.0, shrink_z)

    for l in lc:
        l = np.array(l)
        path_line = mlab.plot3d(l[:, 0], l[:, 1], l[:, 2])
        path_line.actor.actor.scale = (1.0, 1.0, shrink_z)


if __name__ == "__main__":
    import optimal_path

    xrange, yrange, step = (0, 50), (0, 40), 1
    X, Y = optimal_path.generate_xy_grid(xrange, yrange, step)
    obstacles = optimal_path.generate_obstacles(4, xrange, yrange, (4, 10), (3, 8))
    Hground = optimal_path.place_obstacles(X, Y, obstacles)
    dh, d2h, buffer, min_h = 0.5, 0.05, 1.0, 4.0
    Hsheet = optimal_path.get_optimal_grid(Hground, buffer, dh, d2h, min_h)
    # plot 2d
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    optimal_path.plot_mpl_2d(ax1, X, Y, Hsheet)

    # plot 3d
    fig2 = plt.figure(tight_layout=True)
    ax2 = fig2.add_subplot(projection="3d")
    optimal_path.plot_mpl3d(ax2, X, Y, Hground, Hsheet)

    # make PRM
    prm = PRM(X, Y, Hsheet, 1000, 12.0)

    # plot a course between 4 random points
    p1 = get_rand_pt(xrange, yrange)
    p2 = get_rand_pt(xrange, yrange)
    p3 = get_rand_pt(xrange, yrange)
    p4 = get_rand_pt(xrange, yrange)
    for i, p in enumerate([p1, p2, p3, p4]):
        ax1.scatter(p[0], p[1], s=50, label="point {}".format(i))
    ax1.legend()
    p1_p2 = prm.compute_shortest_path(p1, p2)
    p2_p3 = prm.compute_shortest_path(p2, p3)
    p3_p4 = prm.compute_shortest_path(p3, p4)
    # concatenate lists of verts. each contains
    # the points so we don't want to have a point
    # appear twice on the big list.
    path = p1_p2[:-1] + p2_p3[:-1] + p3_p4

    # get line corresponding to path
    lc, line = prm.draw_path(path)
    ax1.add_collection(lc)

    plt.show()
