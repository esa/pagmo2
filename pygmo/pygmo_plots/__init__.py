def plot_non_dominated_fronts(points, marker = 'o'):
    """
    Plots the population pareto front in a 2-D graph
    USAGE: pop.plot_pareto_front(comp = [0,1], rgb=(0,1,0))
    * comp: components of the fitness function to plot in the 2-D window
    * rgb: specify the color of the 1st front (use strong colors here)
    * symbol: marker for the individual
    * size: size of the markersymbol
    * fronts: list of fronts to be plotted (use [0] to only show the first)
    """
    from matplotlib import pyplot as plt
    from pygmo.core import fast_non_dominated_sorting, population
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
            plt.plot(points[idx][0], points[idx][1], marker = marker, color = cl[ndr])
        # We plot the fronts
        # Frist compute the points coordinates
        x = [points[idx][0] for idx in front]
        y = [points[idx][1] for idx in front]
        # Then sort them by the first objective
        tmp = [(a, b) for a, b in zip(x, y)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot
        plt.step([c[0] for c in tmp], [c[1] for c in tmp], color=cl[ndr], where='post')


    plt.show()
    return plt.gca()
