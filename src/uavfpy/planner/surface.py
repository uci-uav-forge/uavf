import numpy as np
import cvxpy as cp
from typing import Tuple
import logging


def generate_xy_grid(xrange, yrange, step):
    """Generate an x y grid over `xrange` `yrange` with grid-size `step`.

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


def generate_random_obstacles(
    n,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    radrange: Tuple[float, float],
    height_range: Tuple[float, float],
) -> dict:
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
    list of dict
        dict with keys "x", "y", "r", "h"
    """
    obstacles = []
    xmin, xmax = xrange
    ymin, ymax = yrange
    rmin, rmax = radrange
    hmin, hmax = height_range
    for _ in range(n):
        obstacles.append(
            {
                "x": np.random.uniform(xmin, xmax),
                "y": np.random.uniform(ymin, ymax),
                "r": np.random.uniform(rmin, rmax),
                "h": np.random.uniform(hmin, hmax),
            }
        )
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
    obstacles : list of dict
        eahc obstacle is dict with fields "x", "y", "r", "h"

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
        for o in obstacles:
            ocent = np.array([o["x"], o["y"]])
            if np.dot(pt - ocent, pt - ocent) < o["r"] * o["r"]:
                H[ij] = o["h"]
    return H


def _diff(x: cp.Variable, h: float, k: int = 1):
    xbak = x[:, 2:]
    xfor = x[:, :-2]
    xmid = x[:, 1:-1]
    if k == 1:
        """first derivative"""
        dx = (xfor - xbak) / (2 * h)
        return dx
    elif k == 2:
        """second derivative"""
        d2x = (xfor - 2 * xmid + xbak) / (h**2)
        return d2x
    else:
        raise ValueError(f"k must be 1 or 2. k={k}")


def get_optimal_grid(
    X,
    Y,
    H,
    buffer,
    max_dh,
    max_d2h,
    min_h,
    step: Tuple[float, float],
    waypoints=None,
    waypointcost=1e5,
    verbose=True,
    solver="ECOS",
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

    solvers_avail_str = ", ".join(cp.installed_solvers())
    logging.info("Solvers Available: {}".format(solvers_avail_str))
    logging.info("Solving with {}".format(solver))
    logging.info("Buffer: {}".format(buffer))
    logging.info("Max dh/dx: {}".format(max_dh))
    logging.info("Max d2h/dx2: {}".format(max_d2h))
    logging.info("Min h: {}".format(min_h))
    logging.info("Grid Step Size: {}".format(step))

    S = cp.Variable(shape=H.shape)

    # Minimum Altitude Constraint
    min_alt_constraint = S - H >= buffer
    # Safe Altitude Constraint
    safe_alt_constraint = S >= min_h

    constraints, cost = [], 0.0

    constraints.append(min_alt_constraint)
    constraints.append(safe_alt_constraint)

    # Magnitude of First Partials dh/dx
    dx = _diff(S, step[0], k=1)
    dy = _diff(S, step[1], k=1)
    c_dx, c_dy = cp.abs(dx) <= max_dh, cp.abs(dy) <= max_dh
    constraints += [c_dx, c_dy]
    # First Derivative Constraints
    # dhx_c, dhy_c = dhx <= max_dh * step[0], dhy <= max_dh * step[1]
    # constraints.append(dhx_c)
    # constraints.append(dhy_c)

    # Magnitude of Second Partials d2h/dx2
    d2x = _diff(S, step[0], k=2)
    d2y = _diff(S, step[1], k=2)
    c_d2x, c_d2y = cp.abs(d2x) <= max_d2h, cp.abs(d2y) <= max_d2h
    constraints += [c_d2x, c_d2y]
    # Second Derivative Constraints

    # derivative1cost = cp.sum_squares(d2hx) / d2hx.size
    # derivative1cost += cp.sum_squares(d2hy) / d2hy.size
    # derivative2cost = cp.sum_squares(dhx) / dhx.size
    # derivative2cost += cp.sum_squares(dhy) / dhy.size

    # cost += derivative1cost * 1.0 + derivative2cost * 1.0

    # waypoints
    if waypoints is not None:
        for wp in waypoints:
            # find closest 4 grid points
            wpy_i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            wpx_i = np.argsort(np.abs(Y - wp[1]), axis=0)[0][:4]
            # average of quadrilateral
            cost += cp.sum_squares(S[wpx_i, wpy_i] - wp[2]) * waypointcost

    # lowest possible
    cost += cp.sum(S)
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # solve problem
    problem.solve(verbose=verbose, solver=solver, warm_start=True)

    # log waypoint vertical errors to console
    waypoint_vertical_errors = []
    for wp in waypoints:
        j = np.argsort(np.abs(X - wp[0]), axis=1)[0][0]
        i = np.argsort(np.abs(Y - wp[1]), axis=0)[0][0]
        waypoint_vertical_errors.append(S.value[i, j] - wp[2])
    logging.info(
        "Max Waypoint Vertical Error: {}".format(max(waypoint_vertical_errors))
    )

    # return solved value
    return S.value
