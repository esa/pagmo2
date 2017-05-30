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

# isl = pygmo.island(algo=ub, prob=pygmo.rosenbrock(),
#                    size=20, udi=pygmo.mp_island())
# isl.evolve()
# isl.wait_check()
