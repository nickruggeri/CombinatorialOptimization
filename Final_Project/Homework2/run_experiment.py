import math
import _pickle as pkl

from Homework2.TSPSolver import TSPSolver


# 20 nodes experiment, repeat 5 times
print('Starting 20 nodes experiment...')
with open('dist_mat_20.pkl', 'rb') as file:
    dist_mat = pkl.load(file)

for i in range(1, 6):
    print('fitting solver number ', i)
    solver = TSPSolver(
        init_size=100, init_type='best2opt', init_best2opt_frac=0.2, fitness='total_cost',
        selection='montecarlo', mating='tuples', mating_n=100, crossover='OX', mutation_prob=0.2,
        gen_replacement='keep_best', gen_replacement_par=100,
        stopping={'time': 1000, 'not_improving_gen': 1000}
    )
    solver.solve(dist_mat)
    print('Best solution found wit fitness', solver.best_fitness)
    with open('20_nodes_experiment_results/solver_{}.pkl'.format(i), 'wb') as file:
        pkl.dump(solver, file)


# 100 nodes experiment, repeat 5 times
print('\nStarting 100 nodes experiment...')
with open('dist_mat_100.pkl', 'rb') as file:
    dist_mat = pkl.load(file)

for i in range(1, 6):
    print('fitting solver number ', i)
    solver = TSPSolver(
        init_size=1000, init_type='best2opt', init_best2opt_frac=0.2, fitness='total_cost',
        selection='montecarlo', mating='tuples', mating_n=1000, crossover='OX', mutation_prob=0.2,
        gen_replacement='keep_best', gen_replacement_par=1000,
        stopping={'time': 1000, 'not_improving_gen': math.inf}
    )
    solver.solve(dist_mat)
    print('Best solution found wit fitness', solver.best_fitness)
    with open('100_nodes_experiment_results/solver_{}.pkl'.format(i), 'wb') as file:
        pkl.dump(solver, file)


# 200 nodes experiment, execute only once
print('\nStarting 200 nodes experiment...')
with open('dist_mat_200.pkl', 'rb') as file:
    dist_mat = pkl.load(file)
print('fitting solver...')
solver = TSPSolver(
    init_size=5000, init_type='best2opt', init_best2opt_frac=0.2, fitness='total_cost',
    selection='montecarlo', mating='tuples', mating_n=5000, crossover='OX', mutation_prob=0.2,
    gen_replacement='keep_best', gen_replacement_par=5000,
    stopping={'time': 10000, 'not_improving_gen': math.inf}
)
solver.solve(dist_mat)
print('Best solution found wit fitness', solver.best_fitness)
with open('200_nodes_experiment_results/solver_1.pkl', 'wb') as file:
    pkl.dump(solver, file)





