import _pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


def solution_at_time(t, solutions):
    """
    Given a time t, return score that was found at that time. If no solution was found,
    return -1
    """
    # recall that (from Python 3) dictionary keys are assured to be returned in the insertion order, so that
    # self.logger.keys are sorted in increasing order by time
    if t < list(log_time for log_time in solutions.keys())[0]:
        return None
    old_log_time = -1
    for log_time in solutions.keys():
        if log_time > t:
            return solutions[old_log_time]
        old_log_time = log_time
    return solutions[old_log_time]


def plot_results(opl_results, evalg_results, times, optimal_sol=None, title=None, save_path=None, show_fig=False,
                 log_scale=False, plot_confidence_interval=True):
    # compute means and standard deviations
    n_results = len(opl_results)
    opl_means = np.array([
        np.mean([solution_at_time(t, opl_results[i]) for i in range(1, n_results + 1) if solution_at_time(t, opl_results[i]) is not None]) for t in times
    ])
    opl_means[np.isnan(opl_means)] = None

    opl_std = np.array([
        np.std([solution_at_time(t, opl_results[i]) for i in range(1, n_results + 1) if solution_at_time(t, opl_results[i]) is not None]) for t in times
    ])
    opl_std[np.isnan(opl_std)] = None
    print('OPL MEANS', opl_means)
    print('OPL STD', opl_std)

    evalg_means = np.array([
        np.mean([evalg_results[i].solution_at_time(t)[1] for i in range(1, n_results + 1) if evalg_results[i].solution_at_time(t) is not None]) for t in times
    ])
    evalg_means[np.isnan(evalg_means)] = None

    evalg_std = np.array([
        np.std([evalg_results[i].solution_at_time(t)[1] for i in range(1, n_results + 1) if evalg_results[i].solution_at_time(t) is not None]) for t in times
    ])
    evalg_std[np.isnan(evalg_std)] = None
    print('EVALG MEANS', evalg_means)
    print('EVALG STD', evalg_std)

    # confidence basrs for genetic algorithm
    upper_conf = evalg_means + evalg_std
    lower_conf = evalg_means - evalg_std
    # plot results
    if log_scale:
        opl_means = np.log(opl_means)
        evalg_means = np.log(evalg_means)
        upper_conf = np.log(upper_conf)
        lower_conf = np.log(lower_conf)
        optimal_sol = np.log(optimal_sol) if optimal_sol is not None else None
    plt.figure()
    plt.plot(times, opl_means, 'y-')
    plt.plot(times, evalg_means, 'k')
    if plot_confidence_interval:
        plt.fill_between(times, lower_conf, upper_conf)
    if optimal_sol is not None:
        plt.plot(times, [optimal_sol]*len(times), '--g')
    if title is not None:
        plt.title(title)
    plt.xticks(times)
    plt.xlabel('time (seconds)')
    plt.ylabel('{}score'.format('log-' if log_scale else ''))
    plt.legend(['OPL solver mean', 'Genetic algorithm mean', 'Optimal solution'])

    if save_path is not None:
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    else:
        plt.close()





# open 20 nodes experiment results
nodes = 20
print('\n{} nodes experiment'.format(nodes))
opl_results = dict()
evalg_results = dict()
for i in range(1, 6):
    # open OPL results
    with open('Homework1/{}_nodes_experiment_results/round_{}.pkl'.format(nodes, i), 'rb') as file:
        opl_results[i] = pkl.load(file)
        if i == 1:
            print('times available for this OPL solution are in the range ',
                  list(opl_results[i].keys())[0], list(opl_results[i].keys())[-1])
    # open gentic algorithm results
    with open('Homework2/{}_nodes_experiment_results/solver_{}.pkl'.format(nodes, i), 'rb') as file:
        evalg_results[i] = pkl.load(file)
        if i==1: print('times available for this genalg solution are in the range ',
                       list(evalg_results[i].logger.keys())[0], list(evalg_results[i].logger.keys())[-1])

optimal_solution = 4.213
times = np.linspace(0, 0.9, 10)
saving_path = 'experimental_data/{}_nodes_results.png'.format(nodes)
plot_results(opl_results, evalg_results, times, optimal_sol=optimal_solution,
             title='Plot of results for {} nodes experiment'.format(nodes), show_fig=True, save_path=saving_path)


# open 100 nodes experiment results
nodes = 100
print('\n{} nodes experiment'.format(nodes))
opl_results = dict()
evalg_results = dict()
for i in range(1, 6):
    # open OPL results
    with open('Homework1/{}_nodes_experiment_results/round_{}.pkl'.format(nodes, i), 'rb') as file:
        opl_results[i] = pkl.load(file)
        if i == 1:
            print('times available for this OPL solution are in the range ',
                  list(opl_results[i].keys())[0], list(opl_results[i].keys())[-1])
    # open gentic algorithm results
    with open('Homework2/{}_nodes_experiment_results/solver_{}.pkl'.format(nodes, i), 'rb') as file:
        evalg_results[i] = pkl.load(file)
        if i == 1:
            print('times available for this genalg solution are in the range ',
                  list(evalg_results[i].logger.keys())[0], list(evalg_results[i].logger.keys())[-1])

optimal_solution = 6.02
times = np.linspace(16, 1000, 10)
log_scale = True
saving_path = 'experimental_data/{}_nodes_results.png'.format(nodes)
plot_results(opl_results, evalg_results, times, optimal_sol=optimal_solution,
             title='Plot of results for {} nodes experiment {}'.format(nodes, 'in log scale' if log_scale else ''),
             show_fig=True, log_scale=log_scale, save_path=saving_path)


# open 200 nodes experiment results
nodes = 200
print('\n{} nodes experiment'.format(nodes))
opl_results = dict()
evalg_results = dict()
for i in range(1, 2):
    # open OPL results
    with open('Homework1/{}_nodes_experiment_results/round_{}.pkl'.format(nodes, i), 'rb') as file:
        opl_results[i] = pkl.load(file)
        if i == 1:
            print('times available for this OPL solution are in the range ',
                  list(opl_results[i].keys())[0], list(opl_results[i].keys())[-1])
    # open gentic algorithm results
    with open('Homework2/{}_nodes_experiment_results/solver_{}.pkl'.format(nodes, i), 'rb') as file:
        evalg_results[i] = pkl.load(file)
        if i == 1:
            print('times available for this genalg solution are in the range ',
                  list(evalg_results[i].logger.keys())[0], list(evalg_results[i].logger.keys())[-1])

optimal_solution = None
times = np.linspace(1000, 10000, 10)
log_scale = False
saving_path = 'experimental_data/{}_nodes_results.png'.format(nodes)
plot_results(opl_results, evalg_results, times, optimal_sol=optimal_solution,
             title='Plot of results for {} nodes experiment {}'.format(nodes, 'in log scale' if log_scale else ''),
             show_fig=True, log_scale=log_scale, plot_confidence_interval=False, save_path=saving_path)




















