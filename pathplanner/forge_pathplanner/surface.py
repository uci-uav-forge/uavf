import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


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


def generate_random_obstacles(n, xrange, yrange, radrange, height_range):
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
        oheight = np.random.unibform(hmin, hmax)
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
    X,
    Y,
    H,
    buffer,
    max_dh,
    max_d2h,
    min_h,
    step,
    waypoints=None,
    waypointcost=1e4,
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
    S = cp.Variable(shape=H.shape)
    n_points = S.shape[0] * S.shape[1]
    constraints = []
    cost = 0

    # Minimum Altitude Constraint
    min_alt_constraint = S - H >= buffer
    # Safe Altitude Constraint
    safe_alt_constraint = S >= min_h

    constraints.append(min_alt_constraint)
    constraints.append(safe_alt_constraint)

    # Magnitude of First Partials dh/dx
    dhx, dhy = cp.abs(cp.diff(S, 1, axis=0)), cp.abs(cp.diff(S, 1, axis=1))

    # First Derivative Constraints
    dhx_c, dhy_c = dhx <= max_dh * step, dhy <= max_dh * step
    # constraints.append(dhx_c)
    # constraints.append(dhy_c)

    # First Derivative Costs (Normed by Array Size)
    cost += cp.sum(dhx) / n_points * 1e2
    cost += cp.sum(dhy) / n_points * 1e2

    # Magnitude of Second Partials d2h/dx2
    d2hx, d2hy = cp.abs(cp.diff(S, 2, axis=0)), cp.abs(cp.diff(S, 2, axis=1))
    d2hx_c, d2hy_c = d2hx <= max_d2h * step * 2, d2hy <= max_d2h * step * 2
    constraints.append(d2hx_c)
    constraints.append(d2hy_c)

    cost += cp.sum(d2hx) / n_points * 1e2
    cost += cp.sum(d2hy) / n_points * 1e2

    # waypoints
    if waypoints is not None:
        for wp in waypoints:
            # find closest 4 grid points
            wpy_i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            wpx_i = np.argsort(np.abs(Y - wp[1]), axis=0)[0][:4]
            # average of quadrilateral
            average = cp.sum(S[wpx_i, wpy_i]) / 4
            # add to cost
            cost += cp.abs(average - wp[2]) * waypointcost

    # lowest possible
    cost += cp.sum(S) / n_points * 1e2
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # solve problem
    if solver is not None:
        problem.solve(verbose=verbose, solver=solver)
    else:
        problem.solve(verbose=verbose)

    # return solved value
    return S.value
