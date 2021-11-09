import matplotlib.pyplot as plt
from p2 import optimal_path
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import cvxpy as cp
from scipy.interpolate import RegularGridInterpolator
import networkx as nx
from pykdtree import kdtree


def plot_mpl_2d(X, Y, H):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contour(X, Y, H, cmap=cm.get_cmap("coolwarm"), levels=20, linewidths=1)
    return ax


def plot_mpl3d(X, Y, H, Hnew):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.45, 1]))
    ax.plot_wireframe(X, Y, Hnew, alpha=0.4, cmap=cm.get_cmap("inferno"), zorder=2)
    ax.plot_surface(X, Y, H, cmap=cm.get_cmap("Blues"), zorder=1)


def solve_path(X, Y, H, Hnew, start, goal):
    path_points = 10
    path = cp.Variable((path_points, 2))
    cost = 0
    # start at start and go to goal
    c1 = path[0, :] == start
    c2 = path[-1, :] == goal
    # we are inside the domain
    c4 = cp.max(path[:, 0]) <= X.max()
    c5 = cp.min(path[:, 0]) >= X.min()
    c6 = cp.max(path[:, 1]) <= Y.max()
    c7 = cp.min(path[:, 1]) >= Y.min()
    # h
    for p in path[1:-1]:
        cost += cp.sum_squares(p - goal)
        n = cp.floor(X.shape[0] * (p[0] - X.min()) / (X.max() - X.min()))
        m = cp.floor(X.shape[1] * (p[1] - Y.min()) / (Y.max() - Y.min()))
        cost += H[n, m]
    prob = cp.Problem(cp.Minimize(cost), [c1, c2, c4, c5, c6, c7])
    prob.solve(verbose=True, solver="ECOS")
    return path.value


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

    def make(self, n):
        """make a PRM over the space"""
        xyh = self.sample(n)
        tree = kdtree.KDTree(xyh)
        for i, pt in enumerate(xyh):
            dist, idx = tree.query(xyh, k=10)
            print(dist, idx)


if __name__ == "__main__":
    hacc, hvel, hbuffer, hmin = 0.01, 0.05, 0.25, 1.0
    xrange, yrange = (0, 1), (0, 1)
    X, Y = optimal_path.generate_xy_grid(xrange, yrange, 0.02)
    obstacles = [
        [np.array([0.01, 0.05]), 0.55, 1.1],
        [np.array([0.2, 0.4]), 0.15, 1.6],
        [np.array([0.9, 0.4]), 0.2, 2.4],
        [np.array([0.5, 0.90]), 0.20, 0.95],
    ]

    H = optimal_path.place_obstacles(X, Y, obstacles)
    Hnew = optimal_path.get_optimal_grid(H, hbuffer, hvel, hacc, hmin)
    ax1 = plot_mpl_2d(X, Y, Hnew)
    start = np.array([0.03, 0.21])
    goal = np.array([0.96, 0.94])
    ax1.scatter(start[0], start[1], marker="o", s=100)
    ax1.scatter(goal[0], goal[1], marker="*", s=100)

    prm = PRM(X, Y, H)
    prm.make(100)

    plt.show()
