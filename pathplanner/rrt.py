from p2 import optimal_path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import cm
import rrtpp
from tqdm import tqdm
from scipy.spatial.distance import euclidean


class SheetRRT(rrtpp.RRTStar):
    def __init__(self, wshape, X, Y, H, alpha):
        # world is blank by default
        world = np.zeros(wshape)
        super().__init__(world)
        self.H = H
        self.alpha = alpha
        self.interp = self.make_interp(X, Y, H)

    def make_interp(self, X, Y, H):
        """make a linear interpolator"""
        x = X[0, :]
        y = Y[:, 0]
        return RegularGridInterpolator((x, y), H)

    def calc_cost(self, v, x=None):
        print(x)
        r2distcost = euclidean(self.T.nodes[v]["point"], x)
        if x is not None:
            cost = self.T.nodes[v]["cost"] + r2distcost
        else:
            cost = self.T.nodes[v]["cost"]
        return cost


def xform_2dline(p1, p2, X, Y, H):
    x = X[0, :]
    y = Y[:, 0]
    interp = RegularGridInterpolator((x, y), H)
    line2d = np.linspace(p1, p2, 1000)
    lineh = interp(line2d)
    return line2d, lineh


@mlab.show
def plot_mayavi(X, Y, H, Hnew):
    s1 = mlab.mesh(X, Y, Hnew, opacity=0.5, colormap="Greys")
    s2 = mlab.mesh(X, Y, H, colormap="terrain")


if __name__ == "__main__":
    min_acc_climb = 0.5
    min_climb = 2
    # minimum distance
    buffer = 1
    min_altitude = 10
    # amount to penalize height when drawing a* path
    astar_heightcost = 0.001

    xrange, yrange = (0, 30), (0, 30)
    radrange = (3, 6)
    hrange = (1, 25)

    X, Y = optimal_path.generate_xy_grid(xrange, yrange, 1)
    obstacles = optimal_path.generate_obstacles(4, xrange, yrange, radrange, hrange)
    H = optimal_path.place_obstacles(X, Y, obstacles)
    Hnew = optimal_path.get_optimal_grid(
        H, buffer, min_climb, min_acc_climb, min_altitude
    )

    rrt = SheetRRT(X.shape, X, Y, H, 1.0)
    end = np.array([X[2, 2], Y[2, 2]])
    start = np.array([X[-2, -2], Y[-2, -2]])

    rrt.make(start, end, 100, 10)

    """
    # plot
    fig = plt.figure(figsize=(8, 4.5), dpi=120, tight_layout=True)

    # get start, end in areas that are all over the map
    start = (2, 2)
    goal = (X.shape[0] - 2, X.shape[1] - 2)
    print("started at {} going to {}".format(start, goal))
    p1 = np.array([X[start], Y[start]])
    p2 = np.array([X[goal], Y[goal]])
    line2d, lineh = xform_2dline(p1, p2, X, Y, Hnew)

    ax = optimal_path.plot_grid(fig, X, Y, H, Hnew, ptype="sheet")
    ax.plot(line2d[:, 0], line2d[:, 1], lineh + 5)

    plt.show()
    """
