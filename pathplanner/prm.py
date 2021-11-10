import matplotlib.pyplot as plt
from p2 import optimal_path
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator
import networkx as nx
from pykdtree import kdtree
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from mayavi import mlab

def plot_mpl_2d(ax, X, Y, H, obstacles):
    ax.contour(X, Y, H, cmap=cm.get_cmap("coolwarm"), levels=30, linewidths=1)
    return ax


def plot_mpl3d(ax, X, Y, H, Hnew):
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.45, 1]))
    ax.plot_surface(X, Y, Hnew, alpha=0.5, cmap=cm.get_cmap("ocean"), zorder=0)
    ax.plot_surface(X, Y, H, cmap=cm.get_cmap("copper"), zorder=1)
    return ax


class PRM(object):
    def __init__(self, X, Y, H):
        # the surface given by X, Y, H
        self.H = H
        self.interp = RegularGridInterpolator((X[0, :], Y[:, 0]), H.T)
        self.xrange = X.min(), X.max()
        self.yrange = Y.min(), Y.max()
        self.T = nx.Graph()

    def sample(self, n):
        """get some random points in h"""
        x = np.random.uniform(*self.xrange, size=(n,))
        y = np.random.uniform(*self.yrange, size=(n,))
        interp_f = lambda xy: self.interp(xy, method="linear")
        xy = np.stack((x, y), axis=-1)
        h = np.apply_along_axis(interp_f, axis=1, arr=xy)
        return np.concatenate((xy, h), axis=1)

    def make(self, n, hweight):
        """make a PRM over the space using delaunay triangulation"""
        xyh = self.sample(n)
        # add points. vertexes to points are keys of `T`
        for i, point in enumerate(xyh):
            self.T.add_node(i, point=point)
        # Delaunay in 2-D -> sheet connectivity
        dt = Delaunay(xyh[:,:2])
        for s in dt.simplices:
            # iterate through pairs
            for s1, s2 in ((s[0], s[1]), (s[1], s[2]), (s[2],s[0])):
                self.T.add_edge(
                    s1, s2,
                    cost=self.edge_cost(xyh, s1, s2, hweight)
                )
        self.tree = kdtree.KDTree(xyh)

    def r2r3(self, x, method="linear"):
        # project a point from R2 to R3
        x_out = np.empty((3,),dtype=x.dtype)
        x_out[:2] = x
        x_out[2] = self.interp(x, method=method)
        return x_out


    def compute_shortest_path(self, xstart, xend):
        """path is given by an ordered list of vertices of the PRM graph."""
        # R2 -> R3
        xstart = np.atleast_2d(self.r2r3(xstart))
        xend = np.atleast_2d(self.r2r3(xend))
        # get nearest node
        _, vstart = self.tree.query(xstart, k=1)
        _, vend = self.tree.query(xend, k=1)
        # get shortest path
        path = nx.shortest_path(self.T, source=vstart[0], target=vend[0], weight="cost", method="bellman-ford")
        return path

    def draw_path(self, path, dims=2):
        """draw path given by a collection of nodes in `T`."""
        linecol = []
        costs = []
        for i, v in enumerate(path[:-1]):
            v1 = v
            v2 = path[i+1]

            if dims == 2:
                p1 = self.T.nodes[v1]["point"][:2]
                p2 = self.T.nodes[v2]["point"][:2]
            elif dims == 3:
                p1 = self.T.nodes[v1]["point"] + np.array([0,0,0.1])
                p2 = self.T.nodes[v2]["point"]+ np.array([0,0,0.1])
            linecol.append((p1, p2))
            costs.append(self.T[v1][v2]["cost"])
        norm = Normalize(min(costs), max(costs))
        return linecol, norm(costs)
        

    def edge_cost(self, xyh, s1, s2, hweight=1.0):
        """cost between vertices s1 and s2."""
        p1 = xyh[s1]
        p2 = xyh[s2]
        # we can scale by h distance
        q = np.array([1,1,hweight])
        d = p2 - p1
        cost = np.linalg.norm(d * q)
        return cost
    
    def get_mpl_edges(self, dims=2):
        """get edges in format for matplotlib LineCollection"""
        linecol = []
        costs = []
        for e1, e2 in self.T.edges:
            if dims==2:
                # clip to R2
                p1 = self.T.nodes[e1]["point"][:2]
                p2 = self.T.nodes[e2]["point"][:2]
            elif dims==3:
                p1 = self.T.nodes[e1]["point"]
                p2 = self.T.nodes[e2]["point"]
            linecol.append((p1, p2))
            costs.append(self.T[e1][e2]["cost"])
        norm = Normalize(min(costs), max(costs))
        return linecol, norm(costs)

def get_rand_pt(xrange, yrange):
    x= np.random.uniform(*xrange)
    y= np.random.uniform(*yrange)
    return np.array((x, y))


@mlab.show
def plot_mayavi(X, Y, H, Hnew, lc, shrink_z = 0.2):
    ground = mlab.mesh(X, Y, H, opacity=1.0, colormap="bone")
    ground.actor.actor.scale = (1.0, 1.0, shrink_z)
    sheet = mlab.mesh(X, Y, Hnew, opacity=0.6, colormap="ocean")
    sheet.actor.actor.scale = (1.0, 1.0, shrink_z)

    for l in lc:
        l = np.array(l)
        path_line = mlab.plot3d(l[:,0], l[:,1], l[:,2])
        path_line.actor.actor.scale = (1.0, 1.0, shrink_z)


if __name__ == "__main__":
    xrange, yrange = (0, 2), (0, 2)
    # get XY Grid
    X, Y = optimal_path.generate_xy_grid(xrange, yrange, 0.02)
    # set random obstacles
    obstacles = optimal_path.generate_obstacles(6, (0,2), (0,2), (0.12, 0.35), (0.95, 5.0))
    # make sheet
    hacc, hvel, hbuffer, hmin = 0.01, 0.3, 0.25, 1.0
    H = optimal_path.place_obstacles(X, Y, obstacles)
    Hnew = optimal_path.get_optimal_grid(H, hbuffer, hvel, hacc, hmin)
    # get random start point
    p1 = get_rand_pt(xrange, yrange)
    p2 = get_rand_pt(xrange, yrange)
    p3 = get_rand_pt(xrange, yrange)
    p4 = get_rand_pt(xrange, yrange)

    # make a PRM
    prm = PRM(X, Y, Hnew)
    prm.make(1200, 2.0)
    path1 = prm.compute_shortest_path(p1, p2)
    path2 = prm.compute_shortest_path(p2, p3)
    path3 = prm.compute_shortest_path(p3, p4)

    path = np.concatenate((path1[:-1], path2, path3[1:]))

    print(path)

    # make a couple plots
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot()
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection="3d")

    # 2d plot...
    # draw surface topo lines
    plot_mpl_2d(ax2d, X, Y, Hnew, obstacles)
    # draw start, end markers
    ax2d.scatter(p1[0], p1[1], marker="o", s=100, c="r")
    ax2d.scatter(p2[0], p2[1], marker="*", s=100, c="b")
    ax2d.scatter(p3[0], p3[1], marker="*", s=100, c="b")
    ax2d.scatter(p4[0], p4[1], marker="*", s=100, c="b")
    # draw path line
    prm_path2d, _ = prm.draw_path(path, dims=2)
    lc = LineCollection(prm_path2d, linewidths=3, colors="r")
    ax2d.add_collection(lc)

    # 3d plot...
    # draw surface and obstacles
    plot_mpl3d(ax3d, X, Y, H, Hnew)
    # draw path line
    prm_path3d, _ = prm.draw_path(path, dims=3)
    lc = Line3DCollection(prm_path3d, linewidths=4, colors="r")
    ax3d.add_collection(lc)

    plot_mayavi(X, Y, H, Hnew, prm_path3d, shrink_z=0.1)

    plt.show()