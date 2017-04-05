import pygmo as pg

# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
prob = pg.problem(pg.schwefel(30))

# 2 - Instantiate a pagmo algorithm
algo = pg.algorithm(pg.sade(gen=100))

# 3 - Instantiate an archipelago with 16 islands having each 20 individuals
archi = pg.archipelago(16, algo=algo, prob=prob, size=20)

# 4 - Run the evolution in parallel on the 16 separate islands 10 times.
archi.evolve(10)

# 5 - Wait for the evolutions to be finished
archi.wait()

# 6 - Print the fitness of the best solution in each island
res = [isl.get_population().champion_f for isl in archi]
print(res)
