import _pickle as pkl

results = {

}

round_n = 1
nodes = 200

with open('{}_nodes_experiment_results/round_{}.pkl'.format(nodes, round_n), 'wb') as file:
    pkl.dump(results, file)


with open('{}_nodes_experiment_results/round_{}.pkl'.format(nodes, round_n), 'rb') as file:
    print(pkl.load(file))