from p2 import optimal_path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mayavi import mlab
from tvtk.util import ctf
from matplotlib import cm


def xform_2dline(p1, p2, X, Y, H):
    x = X[0, :]
    y = Y[:, 0]
    interp = RegularGridInterpolator((x, y), H)
    line2d = np.linspace(p1, p2, 1000)
    lineh = interp(line2d)
    return line2d, lineh


def cmap_to_ctf(cmap_name):
    values = list(np.linspace(0, 1, 256))
    cmap = cm.get_cmap(cmap_name)(values)
    transfer_function = ctf.ColorTransferFunction()
    for i, v in enumerate(values):
        transfer_function.add_rgb_point(v, cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return transfer_function


@mlab.show
def plot_mayavi(X, Y, H, Hnew):

    s1 = mlab.surf(X, Y, Hnew, opacity=0.5, colormap="greys")
    s1 = mlab.surf(X, Y, H, colormap="gist_earth")
    # signal for update
    s1.update_ctf = True


if __name__ == "__main__":
    min_acc_climb = 0.25
    min_climb = 4
    # minimum distance
    buffer = 1
    min_altitude = 10
    # amount to penalize height when drawing a* path
    astar_heightcost = 0.001

    xrange, yrange = (0, 30), (0, 30)
    radrange = (2, 9)
    hrange = (1, 50)

    X, Y = optimal_path.generate_xy_grid(xrange, yrange, 1)
    obstacles = optimal_path.generate_obstacles(4, xrange, yrange, radrange, hrange)
    H = optimal_path.place_obstacles(X, Y, obstacles)
    Hnew = optimal_path.get_optimal_grid(
        H, buffer, min_climb, min_acc_climb, min_altitude
    )
    plot_mayavi(X, Y, H, Hnew)
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
