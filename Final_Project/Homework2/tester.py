import _pickle as pkl
from Final_Project.Homework2.TSPSolver import TSPSolver


data_path = 'dist_mat_try.pkl'
with open(data_path, 'rb') as file:
    dist_mat = pkl.load(file)

solver = TSPSolver()
solver.solve(dist_mat)

print('best solution found with cost', solver.best_fitness, 'in', solver.solution_time, 'seconds')
print('logger keys', solver.logger.keys())
print('solution at time 1, 5, 15, 30', solver.solution_at_time(1)[1], solver.solution_at_time(5)[1], solver.solution_at_time(15)[1], solver.solution_at_time(30)[1])
