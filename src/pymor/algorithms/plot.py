# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np


def adaptive(f, a, b, initial_num=10, max_num=2000, tol=2, min_rel_dist=1e-3,
             aspect_ratio=4/3):
    """Adaptive plot of an array-valued function.

    Parameters
    ----------
    f
        The function to plot.
    a
        The left bound.
    b
        The right bound.
    initial_num
        Initial number of linearly spaced sampling points.
    max_num
        Maximum number of sampling points.
    tol
        Tolerance for the maximum pointwise angle in degrees away from 180°.
    min_rel_dist
        Minimum distance between two neighbouring points relative to the
        width of the plot.
    aspect_ratio
        Ratio between width and height of the plot, used in calculating
        angles and distances.

    Returns
    -------
    points
        A 1D |NumPy array| of sampled points.
    fvals
        An |NumPy array| of function values.
    """
    tol *= np.pi / 180
    points = list(np.linspace(a, b, initial_num))
    fvals = [f(p) for p in points]
    while len(points) < max_num:
        angles, dists = _angles_and_dists(points, fvals, aspect_ratio)
        dists_pair_max = np.max(np.vstack((dists[:-1], dists[1:])), axis=0)
        angles[dists_pair_max <= min_rel_dist] = np.pi
        idx = np.argmin(angles)
        if np.pi - angles[idx] <= tol:
            break
        if dists[idx + 1] > min_rel_dist:
            p2 = (points[idx + 1] + points[idx + 2]) / 2
            points.insert(idx + 2, p2)
            fvals.insert(idx + 2, f(p2))
        if dists[idx] > min_rel_dist and len(points) < max_num:
            p1 = (points[idx] + points[idx + 1]) / 2
            points.insert(idx + 1, p1)
            fvals.insert(idx + 1, f(p1))
    return np.array(points), np.array(fvals)


def _angles_and_dists(x, y, aspect_ratio):
    x_range = x[-1] - x[0]
    y_range = max(y) - min(y)
    x = np.array(x) / x_range * aspect_ratio
    if y_range > 0:
        y = np.array(y) / y_range
    dx = x[:-1] - x[1:]
    dy = y[:-1] - y[1:]
    vectors = np.vstack((dx, dy))
    dists = np.linalg.norm(vectors, axis=0)
    inner_products = -(vectors[:, :-1] * vectors[:, 1:]).sum(axis=0)
    angles = np.arccos(inner_products / (dists[:-1] * dists[1:]))
    return angles, dists
