import _pickle as pkl

results = {
    7551: 9.90,
    7768: 9.42,
    8060: 9.08,
    8385: 9.07,
    8961: 8.95,
    9069: 8.95,
    9450: 8.85,
    9937: 8.84,
    10000: 8.84
}

round_n = 1
nodes = 200

with open('{}_nodes_experiment_results/round_{}.pkl'.format(nodes, round_n), 'wb') as file:
    pkl.dump(results, file)


with open('{}_nodes_experiment_results/round_{}.pkl'.format(nodes, round_n), 'rb') as file:
    print(pkl.load(file))