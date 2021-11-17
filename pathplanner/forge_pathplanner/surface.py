import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def rand_idx(X):
    """what the heck numpy >:("""
    xidx = list(np.ndindex(X.shape))
    idx_of_rand_xidx = np.random.choice(len(xidx))
    return xidx[idx_of_rand_xidx]


def generate_xy_grid(xrange, yrange, step):
    """[summary]

    Parameters
    ----------
    xrange : tuple of float
        x_lower, x_upper
    yrange : tuple of float
        x_lower, x_upper
    step : float
        step size of the grid.

    Returns
    -------
    tuple of np.ndarray
        outputs are given as a tuple of (X, Y) grids, each corresponding
        to the X and Y values. The index of the grid corresponds to the
        relationship between points in the grid.
    """
    xmin, xmax = xrange
    ymin, ymax = yrange
    return np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))


def generate_obstacles(n, xrange, yrange, radrange, height_range):
    """Generate list of `n` random obstacles within the x, y space.

    Parameters
    ----------
    n : int
        no of obstacles to generate
    xrange : tuple of float
        upper and lower x coordinates of the obstacles
    yrange : tuple of float
        upper and lower y coordinates of the obstacles
    radrange : tuple of float
        upper and lower range of obstacle radius
    height_range : tuple of float
        upper and lower range of obstacle height

    Returns
    -------
    list of tuple
        list of tuples with obstacle information:
        [((x, y), radius, height), ...]
    """
    obstacles = []
    xmin, xmax = xrange
    ymin, ymax = yrange
    rmin, rmax = radrange
    hmin, hmax = height_range
    for _ in range(n):
        # center
        ocent = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
        # radius
        orad = np.random.uniform(rmin, rmax)
        # height
        oheight = np.random.uniform(hmin, hmax)
        obstacles.append((ocent, orad, oheight))
    return obstacles


def place_obstacles(X, Y, obstacles):
    """Make a new grid, with the same dimensions as `X`, `Y`, containing
    the obstacles contained in `obstacles.`

    The obstacles will overlap each other by the order they are given in
    the list. So if an O1 overlaps O2, but has a height less than O2, the
    portion that overlaps will be overwritten with O1's height.

    Parameters
    ----------
    X : np.ndarray
        MxN array of x values.
    Y : np.ndarray
        MxN array of y values.
    obstacles : list
        list containing tuples, each with:
        ((obstacle x, obstacle y), obstacle radius, obstacle height)

    Returns
    -------
    np.ndarray
        MxN array of H values corresponding to the "ground" as if obstacles
        have been placed there.
    """
    H = np.zeros_like(X)
    # check each point
    for ij in np.ndindex(X.shape):
        pt = np.array([X[ij], Y[ij]])
        # against each obstacle
        for ocent, orad, oheight in obstacles:
            if np.dot(pt - ocent, pt - ocent) < orad * orad:
                H[ij] = oheight
    return H


def get_optimal_grid(
    H, buffer, max_dh, max_d2h, min_h, step, verbose=True, solver="ECOS"
):
    """Get an optimal height grid corresponding to a surface above the ground from an
    obstacle height grid `H`. The height grid has a minimum dh/dx, d2h/d2x, min h,
    and buffer between tall obstacles and the surface. The input grid `H` is structured
    like an MxN array of h values. The x, y positions are given by separate `X` and `Y`
    grids.

    Parameters
    ----------
    H : np.ndarray
        height grid
    buffer : float
        minimum distance to obstacle or ground
    max_dh : float
        maximum dh/dx that the vehicle can climb. Corresponds to the "slope" of the
        sheet, or the max climb angle of the vehicle
    max_d2h : float
        maximum d2h/dx2 that the vehicle can accelerate to climb. smaller
        values correspond to a "smoother" sheet.
    min_h : float
        minimum altitude of the sheet `h`, irrespective of obstacles
    step : step
        grid step
    verbose : bool, optional
        whether to print verbose solver information, by default True
    solver : str, optional
        the solver to use. The solver must be accessible to `cvxpy`, by default "ECOS"

    Returns
    -------
    np.ndarray
        an array with the same dimensions as `H`, corresponding to the sheet above `H`.
    """
    # new h is a free variable corresponding to H
    newh = cp.Variable(shape=H.shape)
    hc = newh - H >= buffer
    gc = newh >= min_h
    # 1st partial with respect to h -> climb rate
    dhx = cp.diff(newh, 1, axis=0)
    dhy = cp.diff(newh, 1, axis=1)
    dxc = cp.abs(dhx) / step <= max_dh
    dyc = cp.abs(dhy) / step <= max_dh
    # 2nd partial with respect to h -> change in climb rate
    d2hx = cp.diff(newh, 2, axis=0)
    d2hy = cp.diff(newh, 2, axis=1)
    d2hxc = cp.abs(d2hx) / (step ** 2) <= max_d2h
    d2hyc = cp.abs(d2hy) / (step ** 2) <= max_d2h

    # lowest possible
    cost_fn = cp.sum_squares(newh - H)
    constraints = [hc, gc, dxc, dyc, d2hxc, d2hyc]
    problem = cp.Problem(cp.Minimize(cost_fn), constraints)
    problem.solve(verbose=verbose, solver=solver)
    return newh.value


def plot_mpl_2d(ax, X, Y, Hsheet, cmap="coolwarm", levels=20):
    ax.contour(X, Y, Hsheet, cmap=cm.get_cmap(cmap), levels=levels, linewidths=1)
    ax.set_aspect("equal")
    return ax


def plot_mpl3d(
    ax: Axes3D,
    X,
    Y,
    Hground,
    Hsheet,
    zsquash=0.5,
    sheetcmap="gist_earth",
    groundcmap="bone",
    sheet_alpha=0.4,
    wireframe=False,
    wirecount=20,
):
    """Put surfaces from `Hground` (the "ground") and `Hsheet` (the "sheet") onto an `ax.` `ax` must
    be a 3d axis.

    Parameters
    ----------
    ax : Axes3D
        the axis on which to place the surfaces.
    X : np.ndarray
        MxN x coordinate information of the grid
    Y : np.ndarray
        MxN y coordinate information of the grid
    Hground : np.ndarray
        MxN "ground" height.
    Hsheet : np.ndarray
        MxN "sheet" height
    zsquash : float, optional
        by default, the 3d projection will render a cube. But this can make obstacles
        appear very tall. So we can squish the grid a bit to make it more like a 3d map and
        less like a cube., by default 0.5
    sheetcmap : str, optional
        colormap of the sheet, by default "ocean"
    groundcmap : str, optional
        colormap of the ground, by default "copper"
    sheet_alpha : float, optional
        alpha of the sheet. 1.0 is completely opaque, 0.0 is not visible, by default 0.4
    wireframe : bool, optional
        whether to draw wireframe or surface

    Returns
    -------
    Axes3D
        ax containing the new plots the `ax` object passed in is altered in place, so
        you don't necessarily need to do anything with this (just calling the function on `ax` is
        enough to alter the object)
    """
    ar = X.shape[0] / X.shape[1]
    ax.set_box_aspect((1, 1 * ar, zsquash))
    ax.set_proj_type("ortho")
    # draw sheet
    if wireframe:
        hmin, hmax = Hsheet.min(), Hsheet.max()
        norm = plt.Normalize((hmin - hmax) * 0.03, hmax)
        colors = cm.get_cmap(sheetcmap)(norm(Hsheet))
        s = ax.plot_surface(
            X,
            Y,
            Hsheet,
            zorder=2,
            linewidths=0.5,
            shade=False,
            facecolors=colors,
            rcount=X.shape[0],
            ccount=X.shape[1],
        )
        s.set_facecolor((0, 0, 0, 0))
    else:
        ax.plot_surface(
            X, Y, Hsheet, alpha=sheet_alpha, cmap=cm.get_cmap(sheetcmap), zorder=2
        )
    # draw ground
    ax.plot_surface(X, Y, Hground, cmap=cm.get_cmap(groundcmap), zorder=1)

    ax.set_xlim3d(X.min(), X.max())
    ax.set_ylim3d(Y.min(), Y.max())
    ax.set_zlim3d(Hground.min(), Hsheet.max())
    return ax


if __name__ == "__main__":
    dh, d2h, buffer, min_h = 4.0, 7.5, 0.4, 0.25
    obstacles = [((0.6, -0.25), 0.7, 0.3), ((0, 0), 0.3, 0.5)]
    fig = plt.figure(figsize=(9, 9))
    ax1 = (fig.add_subplot(3, 2, 1), fig.add_subplot(3, 2, 3), fig.add_subplot(3, 2, 5))
    ax2 = (
        fig.add_subplot(3, 2, 2, projection="3d"),
        fig.add_subplot(3, 2, 4, projection="3d"),
        fig.add_subplot(3, 2, 6, projection="3d"),
    )
    xrange, yrange = (-1, 1), (-1, 1)
    # get 3 surfaces at different resolutions (step sizes)
    for i, step in enumerate([0.025, 0.05, 0.1]):
        X, Y = generate_xy_grid(xrange, yrange, step)
        Hground = place_obstacles(X, Y, obstacles)
        Hsheet = get_optimal_grid(Hground, buffer, dh, d2h, min_h, step)
        # plot 2d
        plot_mpl_2d(ax1[i], X, Y, Hsheet)

        # plot 3d
        plot_mpl3d(ax2[i], X, Y, Hground, Hsheet, wireframe=True)

    plt.show()
