import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_surface_3d(
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
    return ax


def plot_surface_2d(ax, X, Y, Hsheet, cmap="coolwarm", levels=20):
    ax.contour(X, Y, Hsheet, cmap=cm.get_cmap(cmap), levels=levels, linewidths=1)
    ax.set_aspect("equal")
    return ax
