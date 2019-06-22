import math
import _pickle as pkl
import random

import numpy as np

from Final_Project.Homework2.TSPSolver import TSPSolver

'''
# set seed for reproducibility
random.seed(48)
np.random.seed(48)


# ------------------------------------------------------------------------------------------------
# 20 nodes experiment
# ------------------------------------------------------------------------------------------------
print('Starting 20 nodes experiment...')
with open('dist_mat_20.pkl', 'rb') as file:
    dist_mat = pkl.load(file)

# Parameters selection on a grid
best_score = math.inf
print('starting parameters selection loop...')
for init_size in [50, 100, 150]:
    mating_n = init_size
    for mutation_prob in [0.2, 0.4, 0.6]:
        for gen_replacement_par in [50, 100, 150]:
            parameters = {
                'init_size': init_size,
                'mating_n': mating_n,
                'mutation_prob': mutation_prob,
                'gen_replacement_par': gen_replacement_par,
                }
            print(parameters)
            solver = TSPSolver(stopping={'time': 100, 'not_improving_gen': 100}, **parameters)
            solver.solve(dist_mat)
            if solver.best_fitness < best_score:
                best_parameters = parameters
print('optimal parameters found: ', best_parameters)

# repeat experiment 5 times
print('starting to solve problem with optimal parameters...')
for i in range(1, 6):
    print('fitting solver number ', i)
    solver = TSPSolver(stopping={'time': 1000, 'not_improving_gen': 1000}, **best_parameters)
    solver.solve(dist_mat)
    print('Best solution found with fitness', solver.best_fitness)
    with open('20_nodes_experiment_results/solver_{}.pkl'.format(i), 'wb') as file:
        pkl.dump(solver, file)


# ------------------------------------------------------------------------------------------------
# 100 nodes experiment
# ------------------------------------------------------------------------------------------------
print('\n\nStarting 100 nodes experiment...')
with open('dist_mat_100.pkl', 'rb') as file:
    dist_mat = pkl.load(file)

# Parameters selection on a grid
best_score = math.inf
print('starting parameters selection loop...')
for init_size in [500, 1000, 1500]:
    mating_n = init_size
    for mutation_prob in [0.2, 0.4, 0.6]:
        for gen_replacement_par in [500, 1000, 1500]:
            parameters = {
                'init_size': init_size,
                'mating_n': mating_n,
                'mutation_prob': mutation_prob,
                'gen_replacement_par': gen_replacement_par,
                }
            print(parameters)
            solver = TSPSolver(stopping={'time': 100, 'not_improving_gen': 1000}, **parameters)
            solver.solve(dist_mat)
            if solver.best_fitness < best_score:
                best_parameters = parameters
print('optimal parameters found: ', best_parameters)

for i in range(1, 6):
    print('fitting solver number ', i)
    solver = TSPSolver(stopping={'time': 1000}, **best_parameters)
    solver.solve(dist_mat)
    print('Best solution found with fitness', solver.best_fitness)
    with open('100_nodes_experiment_results/solver_{}.pkl'.format(i), 'wb') as file:
        pkl.dump(solver, file)

'''
# ------------------------------------------------------------------------------------------------
# 200 nodes experiment
# ------------------------------------------------------------------------------------------------
# set seed for reproducibility
random.seed(48)
np.random.seed(48)

print('\n\nStarting 200 nodes experiment...')
with open('dist_mat_200.pkl', 'rb') as file:
    dist_mat = pkl.load(file)

# Parameters selection on a grid
best_score = math.inf
print('starting parameters selection loop...')
for init_size in [2000, 5000, 8000]:
    mating_n = init_size
    for mutation_prob in [0.2, 0.4, 0.6]:
        for gen_replacement_par in [2000, 5000, 8000]:
            parameters = {
                'init_size': init_size,
                'mating_n': mating_n,
                'mutation_prob': mutation_prob,
                'gen_replacement_par': gen_replacement_par,
                }
            print(parameters)
            solver = TSPSolver(stopping={'time': 1000, 'not_improving_gen': 10000}, **parameters)
            solver.solve(dist_mat)
            if solver.best_fitness < best_score:
                best_parameters = parameters
print('optimal parameters found: ', best_parameters)

print('fitting solver...')
solver = TSPSolver(stopping={'time': 10000}, **best_parameters)
solver.solve(dist_mat)
print('Best solution found wit fitness', solver.best_fitness)
with open('200_nodes_experiment_results/solver_1.pkl', 'wb') as file:
    pkl.dump(solver, file)






