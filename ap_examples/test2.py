import uda_basic
import udp_basic
import pygmo
import pickle

ub = pygmo.problem(udp_basic.udp_basic())
assert pickle.dumps(pickle.loads(pickle.dumps(ub))) == pickle.dumps(ub)

isl = pygmo.island(algo=uda_basic.uda_basic(), prob=ub, size=20)
risl = repr(isl)
assert "Thread" in risl
isl.evolve()
isl.wait_check()
assert risl == repr(isl)


class py_uda(object):

    def evolve(self, pop):
        return pop


if __name__ == '__main__':
    isl = pygmo.island(algo=py_uda(), prob=ub, size=20)
    isl.evolve()
    isl.wait_check()
    pygmo.mp_island.shutdown_pool()
    pygmo.mp_bfe.shutdown_pool()
    print("All good!")
