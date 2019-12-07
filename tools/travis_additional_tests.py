# NOTE: a couple of serialization tests from:
# https://github.com/esa/pagmo2/issues/106


class toy_problem:

    def fitness(self, x):
        import numpy as np
        return [np.sum(np.sin((x - .2)**2))]

    def get_bounds(self):
        import numpy as np
        return (np.array([-1] * 3), np.array([1] * 3))


class toy_problem_2:

    def fitness(self, x):
        import numpy as np
        return [np.sin(x[0] + x[1] - x[2])]

    def gradient(self, x):
        import pygmo as pg
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def get_bounds(self):
        import numpy as np
        return (np.array([-1] * 3), np.array([1] * 3))


def test1():
    import pygmo as pg
    algo = pg.algorithm(pg.de(gen=1000, seed=126))
    prob = pg.problem(toy_problem())
    pop = pg.population(prob=prob, size=10)
    print(pop.champion_f)
    pop = algo.evolve(pop)
    print(pop.champion_f)
    # fine up to this point

    archi = pg.archipelago(n=6, algo=algo, prob=prob, pop_size=70)
    archi.evolve()
    archi.wait_check()
    print("end of test1")


def test2():
    import pygmo as pg
    mma = pg.algorithm(pg.nlopt("mma"))
    p_toy = pg.problem(toy_problem_2())
    archi = pg.archipelago(n=6, algo=mma, prob=p_toy, pop_size=1)
    archi.evolve()
    archi.wait_check()
    print("end of test2")


if __name__ == '__main__':
    test1()
    test2()
    pg.mp_island.shutdown_pool()
