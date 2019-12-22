import uda_basic
import pygmo
import pickle

ub = pygmo.algorithm(uda_basic.uda_basic())
assert pickle.dumps(pickle.loads(pickle.dumps(ub))) == pickle.dumps(ub)

isl = pygmo.island(algo=ub, prob=pygmo.rosenbrock(), size=20)
risl = repr(isl)
assert "Thread" in risl
isl.evolve()
isl.wait_check()
assert risl == repr(isl)


class py_udp(object):

    def get_bounds(self):
        return ([0, 0], [1, 1])

    def fitness(self, a):
        return [42]


if __name__ == '__main__':
    isl = pygmo.island(algo=ub, prob=py_udp(), size=20)
    isl.evolve()
    isl.wait_check()
    pygmo.mp_island.shutdown_pool()
    pygmo.mp_bfe.shutdown_pool()
    print("All good!")
