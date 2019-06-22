import _pickle as pkl
from Final_Project.Homework2.TSPSolver import TSPSolver


# load small data matrix for running the solver
data_path = 'dist_mat_20.pkl'
with open(data_path, 'rb') as file:
    dist_mat = pkl.load(file)


# try all combinations of parameters
for init_type in ['random', 'best2opt']:
    for selection in ['montecarlo', 'linear_ranking', 'n_tournament']:
        for mating in ['tuples', 'mating_pool']:
            for crossover in ['PMX', 'CX', 'OX', 'LOX', 'SCX']:
                for gen_replacement in ['keep_best', 'remove_worst']:
                    parameters = {
                        'init_type': init_type,
                        'selection': selection,
                        'mating': mating,
                        'crossover': crossover,
                        'gen_replacement': gen_replacement,
                        'stopping': {'time': 10}
                    }
                    print('\n', parameters)
                    solver = TSPSolver(**parameters)
                    solver.solve(dist_mat)

                    print('best solution found with cost', solver.best_fitness, 'in', solver.solution_time, 'seconds')
