import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import networkx as nx
from p2 import polygon, bdc
import cvxpy as cp


def set_equal(x, y, ax):
    """https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""
    # Create cubic bounding box to simulate equal aspect ratio
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)
    plot_radius = max(
        [
            abs(lim - mean)
            for lims, mean in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
            for lim in lims
        ]
    )
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


if __name__ == "__main__":
    xmin, xmax, ymin, ymax = 0, 20, 0, 20
    step = 1
    # create a grid
    X, Y = np.meshgrid(np.arange(xmin, xmax, step), np.arange(xmin, xmax, step))
    H = np.zeros(X.shape)
    print(X.shape, Y.shape, H.shape)

    # make new obstacles
    for _ in range(2):
        ocent = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
        orad = np.random.uniform(1, 5)
        oheight = np.random.uniform(0.1, 0.5)
        for (xij, x), (yij, y), (hij, h) in zip(
            np.ndenumerate(X), np.ndenumerate(Y), np.ndenumerate(H)
        ):
            pt = np.array([X[xij], Y[yij]])
            if np.dot(pt - ocent, pt - ocent) < orad * orad:
                H[hij] = oheight

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection="3d")

    newh = cp.Variable(shape=H.shape)

    alpha = 3.0

    relaxations = 12
    # as low as possible, with minimum variation
    smoothnessx = cp.diff(newh, 1, axis=0)
    smoothnessy = cp.diff(newh, 1, axis=1)
    smoothness = 0
    for _ in range(relaxations):
        smoothness += cp.sum_squares(smoothnessx) + cp.sum_squares(smoothnessy)

    cost_fn = cp.sum_squares(newh - H) + smoothness * alpha
    hconstraint = newh - H >= 0.05
    constraints = [hconstraint]
    problem = cp.Problem(cp.Minimize(cost_fn), constraints)
    problem.solve()

    ax1.plot_surface(X, Y, H, cmap=cm.get_cmap("cividis"))
    ax1.plot_wireframe(X, Y, newh.value, colors="red")

    plt.show()
