import _pickle as pkl
from Homework2.TSPSolver import TSPSolver


data_path = 'dist_mat_try.pkl'
with open(data_path, 'rb') as file:
    dist_mat = pkl.load(file)

solver = TSPSolver()
solver.solve(dist_mat)

print('best solution found', solver.best_individual)
print('with cost', solver.best_fitness)
