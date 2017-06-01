# -*- coding: utf-8 -*-

# Copyright 2017 PaGMO development team
#
# This file is part of the PaGMO library.
#
# The PaGMO library is free software; you can redistribute it and/or modify
# it under the terms of either:
#
#   * the GNU Lesser General Public License as published by the Free
#     Software Foundation; either version 3 of the License, or (at your
#     option) any later version.
#
# or
#
#   * the GNU General Public License as published by the Free Software
#     Foundation; either version 3 of the License, or (at your option) any
#     later version.
#
# or both in parallel, as here.
#
# The PaGMO library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received copies of the GNU General Public License and the
# GNU Lesser General Public License along with the PaGMO library.  If not,
# see https://www.gnu.org/licenses/.

# for python 2.0 compatibility
from __future__ import absolute_import as _ai

# Plotting functions


def plot_non_dominated_fronts(points, marker='o', comp=[0, 1]):
    """
    Plots the nondominated fronts of a set of points. Makes use of :class:`~pygmo.fast_non_dominated_sorting` to
    compute the non dominated fronts.

    Args:
        points (2d array-like): points to plot
        marker (``str``): matplotlib marker used to plot the *points*
        comp (``list``): Components to be considered in the two dimensional plot (useful in many-objectives cases)

    Returns:
        ``matplotlib.axes.Axes``: the current ``matplotlib.axes.Axes`` instance on the current figure

    Examples:
        >>> from pygmo import *
        >>> prob = problem(zdt())
        >>> pop = population(prob, 40)
        >>> ax = plot_non_dominated_fronts(pop.get_f()) # doctest: +SKIP
    """
    from matplotlib import pyplot as plt
    from ..core import fast_non_dominated_sorting, population
    from numpy import linspace

    # We plot
    fronts, _, _, _ = fast_non_dominated_sorting(points)

    # We define the colors of the fronts (grayscale from black to white)
    cl = list(zip(linspace(0.1, 0.9, len(fronts)),
                  linspace(0.1, 0.9, len(fronts)),
                  linspace(0.1, 0.9, len(fronts))))

    for ndr, front in enumerate(fronts):
        # We plot the points
        for idx in front:
            plt.plot(points[idx][comp[0]], points[idx][
                     comp[1]], marker=marker, color=cl[ndr])
        # We plot the fronts
        # Frist compute the points coordinates
        x = [points[idx][0] for idx in front]
        y = [points[idx][1] for idx in front]
        # Then sort them by the first objective
        tmp = [(a, b) for a, b in zip(x, y)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot using step
        plt.step([c[0] for c in tmp], [c[1]
                                       for c in tmp], color=cl[ndr], where='post')

    plt.show()
    return plt.gca()


def _dtlz_plot(self, pop, az=40, comp=[0, 1, 2]):
    """
    Plots solutions to the DTLZ problems in three dimensions. The Pareto Front is also
    visualized if the problem id is 2,3 or 4.

    Args:
        pop (:class:`~pygmo.population`): population of solutions to a dtlz problem
        az (``float``): angle of view on which the 3d-plot is created
        comp (``list``): indexes the fitness dimension for x,y and z axis in that order

    Returns:
        ``matplotlib.axes.Axes``: the current ``matplotlib.axes.Axes`` instance on the current figure

    Raises:
        ValueError: if *pop* does not contain a DTLZ problem (veryfied by its name only) or if *comp* is not of length 3

    Examples:
        >>> import pygmo as pg
        >>> udp = pg.dtlz(prob_id = 1, fdim =3, dim = 5)
        >>> pop = pg.population(udp, 40)
        >>> udp.plot(pop) # doctest: +SKIP
    """
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np

    if (pop.problem.get_name()[:-1] != "DTLZ"):
        raise(ValueError, "The problem seems not to be from the DTLZ suite")

    if (len(comp) != 3):
        raise(ValueError, "The kwarg *comp* needs to contain exactly 3 elements (ids for the x,y and z axis)")

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the points
    fit = np.transpose(pop.get_f())
    try:
        ax.plot(fit[comp[0]], fit[comp[1]], fit[comp[2]], 'ro')
    except IndexError:
        print('Error. Please choose correct fitness dimensions for printing!')

    # Plot pareto front for dtlz 1
    if (pop.problem.get_name()[-1] in ["1"]):

        X, Y = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
        Z = - X - Y + 0.5
        # remove points not in the simplex
        for i in range(100):
            for j in range(100):
                if X[i, j] < 0 or Y[i, j] < 0 or Z[i, j] < 0:
                    Z[i, j] = float('nan')

        ax.set_xlim(0, 1.)
        ax.set_ylim(0, 1.)
        ax.set_zlim(0, 1.)

        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        plt.plot([0, 0.5], [0.5, 0], [0, 0])

    # Plot pareto fronts for dtlz 2,3,4
    if (pop.problem.get_name()[-1] in ["2", "3", "4"]):
        # plot the wireframe of the known optimal pareto front
        thetas = np.linspace(0, (np.pi / 2.0), 30)
        # gammas = np.linspace(-np.pi / 4, np.pi / 4, 30)
        gammas = np.linspace(0, (np.pi / 2.0), 30)

        x_frame = np.outer(np.cos(thetas), np.cos(gammas))
        y_frame = np.outer(np.cos(thetas), np.sin(gammas))
        z_frame = np.outer(np.sin(thetas), np.ones(np.size(gammas)))

        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.set_autoscalez_on(False)

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.8)
        ax.set_zlim(0, 1.8)

        ax.plot_wireframe(x_frame, y_frame, z_frame)

    ax.view_init(azim=az)
    plt.show()
    return plt.gca()

from ..core import dtlz
dtlz.plot = _dtlz_plot
