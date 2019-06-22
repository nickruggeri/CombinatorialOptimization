import math
import time
import random

import numpy as np


class TSPSolver:
    """ TSP solver using genetic algorithms """

    def __init__(self, init_size=100, init_type='random', init_best2opt_frac=0.2, fitness='total_cost',
                 selection='montecarlo', n_tournament_frac=None,  mating='tuples', mating_n=10, crossover='OX',
                 mutation_prob=0.3, gen_replacement='keep_best', gen_replacement_par=10,
                 stopping={'time': math.inf, 'not_improving_gen': 1000}, use_logger=True):
        """
        INPUTS:
            - init_size: size of first population
            - init_type: way to initialize first population. One between 'random' (i.e. just random population) and
              'best2opt' (i.e. select the best 2 opt neighbour after random initialization). In case 'best2opt' is
              selected, only a fraction of the population can be mutated, indicated in init_best2opt_frac
            - init_best2opt_frac: fraction of initial population that is substituted with the best 2-opt neighbour
              available. Used only if self.init_type = 'best2opt'
            - fitness: fitness function. Accepted values are 'total_cost', in which case the cost of the individual is
              computed as fitness function. The project is to expand the list of available fitness functions
            - selection: accepted values are
                    'montecarlo'
                    'linear_ranking'
                    'n_tournament'
            - n_tournament_frac: None or float in (0,1]. Used only if selection = 'n_tournament'. It is the fraction of
              the current generation to select at random for selecting the best individual. Therefore for the new
              generation repeat the following for the required number of steps: select a fraction n_tournament_frac of
              the current generation at random, and pick the best individual.
              If None, defaults to 0.2
            -mating: the way to choose the couples to execute crossover. Accepted values are:
                    'tuples' select a number of tuples, specified by mating_n
                    'mating_pool' select a mating pool of mating_n and crossover among all possible couples
            - mating_n: number of couples or mating pool size depending on the values for parameter mating
            - crossover: accepted values are
                    'PMX' (partially mapped crossover)
                    'CX' (cycle crossover)
                    'OX' (order crossover),
                    'LOX' (linear order crossover)
                    'SCX' (sequential constructive operator). This is the only crossover yielding one offspring instead of two
              (see "Study of Crossover operators in Genetic Algorithm for Travelling Salesman Problem" for reference and
              notation)
            - mutation_prob: the probability of mutation for every individual after crossover. If a mutation happens, it
              is of 2-opt type, i.e. two cut points are selected at random and substring reversal is executed
            - gen_replacement: accepted values
                    -'keep_best' keep only the best gen_replacement_par individuals among the union between old and new
                      generation
                    - 'remove_worst' remove the worst gen_replacement_par individuals among the union between old and
                      new generation
            - gen_replacement_par: integer, see gen_replacement for explanation
            - stopping: a dictionary that includes all stopping criteria to check. Accepted values in the dictionary are
                    {'time': n_seconds} after which to stop
                    {'not_improving_gen':n} number of consecutive non improving generations
                    {'target': x} target objective value, stop if best individual is below x
                    {'max_generations':n} number of maximum generations
            - use_logger: if to save intermediate results to logger for future exploration of intermediate results.
              Since intermediate results could be memory consuming, this option can be set to False

        SOME NOTES:
            - the neighbourhood considered is the standard 2-opt, no option is made available to modify this. For
              preserving diversification use other techniques, like a custom gen_replacement, selection or mating
              type
            - the encoding is done using path representation, i.e. an individual is a numpy array of length
              self.problem_size containing all elements in range(0, self.problem_size) i.e. visit vertices in the
              order specified in the array. Consequently a generation is a list of such numpy arrays
        """

        self.init_size = init_size
        self.init_type = init_type
        self.init_best2opt_frac = init_best2opt_frac
        self.fitness = fitness
        self.selection = selection
        self.n_tournament_frac = n_tournament_frac if n_tournament_frac is not None else 0.2
        self.mating = mating
        self.mating_n = mating_n
        self.crossover = crossover
        self.mutation_prob = mutation_prob
        self.gen_replacement = gen_replacement
        self.gen_replacement_par = gen_replacement_par
        self.stopping = stopping

        # logger for saving the intermediate solutions. It is a dictionary with keys the times at which new solutions
        # are found, and values the tuples (solution, score)
        self._use_logger = use_logger
        if use_logger:
            self.logger = dict()

        # problem related parameters, instanciated when an actual cost matrix is given for solving a problem instance
        self.problem_size = None
        self.cost_matrix = None
        self.best_fitness = None
        self.best_individual = None
        self._current_generation = []
        self._current_fitnesses = []
        self.generations_count = 0
        self._starting_time = None
        self._not_improving_gen_count = 0

        # total solution time
        self.solution_time = None

    def solve(self, cost_matrix):
        """
        Given the numpy array of the costs, with shape [n,n], solve the problem.
        Returns the best solution found and its fitness value
        """
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError('The input matrix must be a 2 dimensional squared numpy array')

        self.problem_size = cost_matrix.shape[0]
        self.cost_matrix = cost_matrix
        self.best_fitness = math.inf

        # if time is in stopping criteria, save starting time
        if 'time' in self.stopping:
            self._starting_time = time.time()

        # notice that self._initialize_generation modifies many attributes of the class in place
        self._initialize_generation()

        while not self._stop_check_satisfied():
            # run all evolution steps, and evaluate fitness recursively.
            # self.current generation, self._current_fitnesses, self.best_fitness, self.best_individual and
            # self._not_improving_gen_count are modified in place
            self._evolve_generation()

        self.solution_time = time.time() - self._starting_time
        return self.best_individual, self.best_fitness

    def _initialize_generation(self):
        """
        Initialize first generation. Every individual is a permutation of range(0, self.problem_size), in case optimized
        via 2-opt heuristic.
        In case we use the 2-opt heuristic, we compute the actual maximum in the neighbourhood. We could also just look
        for the first improvement found. This can be done since the neighbourhood function is a generator, but could be
        a biased approach due to the ordered discovery of the neighbourhood. To comply with how generations are treated
        in the other functions, the generation is sorted by fitness value
        """
        self._current_generation = [np.random.permutation(self.problem_size) for _ in range(self.init_size)]
        if self.init_type == 'best2opt':
            # optimize individual with 2-opt heuristic with probability self.init_best2opt_frac
            for i, individual in enumerate(self._current_generation):
                if random.random() <= self.init_best2opt_frac:
                    neighs = list(self.neighbourhood(individual))
                    best_neigh_idx = max(range(len(neighs)), key=lambda idx: self.fitness_evaluation(neighs[idx]))
                    self._current_generation[i] = neighs[best_neigh_idx]
        # compute fitnesses, sort, update best fitness and best individual
        self._current_fitnesses = [self.fitness_evaluation(individual) for individual in self._current_generation]

        ordered_idx = sorted(
            range(len(self._current_generation)),
            key=lambda ind: self._current_fitnesses[ind]
        )
        self._current_generation = [self._current_generation[i] for i in ordered_idx]
        self._current_fitnesses = [self._current_fitnesses[i] for i in ordered_idx]

        self.best_individual = self._current_generation[0]
        self.best_fitness = self._current_fitnesses[0]

    def _stop_check_satisfied(self):
        """ Check if any stopping criterion instanciated is satisfied """
        if 'time' in self.stopping:
            if time.time() - self._starting_time > self.stopping['time']:
                return True
        if 'not_improving_gen' in self.stopping:
            if self._not_improving_gen_count > self.stopping['not_improving_gen']:
                return True
        if 'target' in self.stopping:
            if self.best_fitness <= self.stopping['target']:
                return True
        if 'max_generations' in self.stopping:
            if self.generations_count > self.stopping['max_generations']:
                return True
        return False

    def neighbourhood(self, individual):
        """ Generator of all the 2-opt neighbours """
        for i in range(1, self.problem_size-1):
            for j in range(i+1, self.problem_size):
                yield np.hstack([individual[:i], individual[j-1:i-1:-1], individual[j:]])

    def fitness_evaluation(self, individual, split_indices=None, prev_score=None):
        """
        Evaluate the fitness of an individual. To allow for iterative evaluation, also allow the option to
        input the score of a 2-opt neighbour of the individual,and the two splits that differentiate them.
        If either of the two is missing, then the whole fitness function is computed.
        INPUT:
            - individual: numpy array, permutation of range(0, self.problem_size)
            - split_indices: tuple (c1, c2) with c1,c 2 indices of the cuts between the 2-opt neighbours.
              Cuts are the index of the element of their right in the array. For example a cut
              in an array like [elem_0, elem_1 ||| elem_2, elem_3, ..., elem_j ||| elem_j+1, ...]
              is (c1, c2) = (2, j+1). Consequently, since node 0 has to stay the first element, c1 and c2 have
              to be between 1 and self.problem_size (extremes included)
            - prev_score: the fitness score of the previous individual, to be modified iteratively
        """
        if self.fitness == 'total_cost':
            if split_indices is not None and prev_score is not None:
                # incremental cost: prev_cost - c_[i-1, j] - c [j, j+1] + c[i-1, j] + c[j-1, j]
                c1, c2 = split_indices
                return(
                    prev_score +
                    self.cost_matrix[individual(c1-1), individual(c1)] +
                    self.cost_matrix[individual(c2-1), individual(c2 % self.problem_size)] -
                    self.cost_matrix[individual(c1-1), individual(c2-1)] -
                    self.cost_matrix[individual(c1+1), individual(c2 % self.problem_size)]
                )
            else:
                return sum(
                    self.cost_matrix[individual[i], individual[(i+1) % self.problem_size]]
                    for i in range(self.problem_size)
                )

    def _evolve_generation(self):
        """ Run the whole evolution process, as specified by the parameters initialized to the TSPSolver """

        sel = self._select_individuals()
        next_gen = self._mating_and_mutation(sel)
        next_fit = list(map(lambda x: self.fitness_evaluation(x), next_gen))

        # select individuals according to generational replacement policy
        old_new_gen = self._current_generation + next_gen
        old_new_fit = self._current_fitnesses + next_fit

        sorted_idx = sorted(range(len(old_new_fit)), key=lambda i: old_new_fit[i])
        if self.gen_replacement == 'keep_best':
            self._current_generation = [old_new_gen[i] for i in sorted_idx[:self.gen_replacement_par]]
            self._current_fitnesses = [old_new_fit[i] for i in sorted_idx[:self.gen_replacement_par]]
        elif self.gen_replacement == 'remove_worst':
            self._current_generation = [old_new_gen[i] for i in sorted_idx[:-self.gen_replacement_par]]
            self._current_fitnesses = [old_new_fit[i] for i in sorted_idx[:-self.gen_replacement_par]]

        # if better solution is found, update the best individual and save to logger
        if self._current_fitnesses[0] < self.best_fitness:
            self.best_fitness = self._current_fitnesses[0]
            self.best_individual = self._current_generation[0]
            self._not_improving_gen_count = 0
            if self._use_logger:
                self.logger[time.time() - self._starting_time] = (self.best_individual, self.best_fitness)
        else:
            self._not_improving_gen_count += 1

        # update generation count
        self.generations_count += 1

    def _select_individuals(self):
        """
        Select individuals according to probability and mating type.
        Depending on the type of mating, return a list of couples for mating or a list
        of individuals. In the returned object, individuals are represented by their
        index in self._current_generation
        """
        if self.mating == 'tuples':
            return [self._random_selection(2) for _ in range(self.mating_n)]
        elif self.mating == 'mating_pool':
            return self._random_selection(self.mating_n)

    def _random_selection(self, n):
        """ Select n individuals from the current generation, according to criterion from self.selection """
        if self.selection == 'montecarlo':
            # select individuals with probability proportional to their fitness
            return list(np.random.choice(
                    len(self._current_generation), n,
                    p=self._current_fitnesses / sum(self._current_fitnesses)
                    ))
        elif self.selection == 'linear_ranking':
            # recall that generations are sorted by fitness, so that the ranks are already (k, k-1, ... ,2,1)
            m = len(self._current_generation)
            binomial_m_2 = m * (m + 1) / 2
            p = np.arange(m, 0, -1) / binomial_m_2
            return list(np.random.choice(
                    len(self._current_generation), n,
                    p=p
                    ))
        elif self.selection == 'n_tournament':
            # recall that generations are sorted by fitness, so that if we select some indices at random, the minimum
            # one is already the index of the fittest individual
            return [
                min(
                    np.random.choice(len(self._current_generation),
                                     size=int(self.n_tournament_frac * len(self._current_generation)))
                )
                for _ in range(n)
            ]

    def _mating_and_mutation(self, sel):
        """
        Given the selection of individuals to execute mating on, execute mating.
        The shape of sel depends on the type of mating (see self._select_individuals),
        the returned object is the list of individuals deriving from the crossover of
        individuals in sel and consequent mutation.
        Return the new generation
        """
        if self.mating == 'tuples':
            next_gen = [ind for x1, x2 in sel for ind in self._crossover(x1, x2)]
        elif self.mating == 'mating_pool':
            next_gen = [
                ind
                for i in range(len(sel)-1)
                for j in range(i+1, len(sel))
                for ind in self._crossover(sel[i], sel[j])
            ]
        assert all(x in ind for x in range(self.problem_size) for ind in next_gen)
        return [self._mutate_individual(ind) for ind in next_gen]

    def _crossover(self, ind1, ind2):
        """
        Execute crossover between individuals ind1 and ind2. The two individuals are represented by their index in
        self._current_generation. Return two individuals as offspring
        """
        # individuals are given in input as indexes. Retrieve the actual individuals from self._current_generation
        ind1, ind2 = self._current_generation[ind1], self._current_generation[ind2]
        # initialize offspring
        off1 = np.zeros(self.problem_size, dtype=np.int)
        off2 = np.zeros(self.problem_size, dtype=np.int)    # not used if self.crossover == 'SCX'

        if self.crossover == 'PMX':
            # random cut points
            c1 = random.randint(0, self.problem_size-1)
            c2 = random.randint(c1+1, self.problem_size)

            off1[c1:c2] = ind2[c1:c2]
            elements1 = set(ind2[c1:c2])
            off2[c1:c2] = ind1[c1:c2]
            elements2 = set(ind1[c1:c2])

            # build offspring1
            sub_rule_1 = {ind2[i]: ind1[i] for i in range(c1, c2)}
            for i, x in enumerate(ind1[:c1]):
                curr = x
                while curr in elements1:
                    curr = sub_rule_1[curr]
                off1[i] = curr
            for i, x in enumerate(ind1[c2:]):
                curr = x
                while curr in elements1:
                    curr = sub_rule_1[curr]
                off1[i+c2] = curr
            # build offspring2
            sub_rule_2 = {ind1[i]: ind2[i] for i in range(c1, c2)}
            for i, x in enumerate(ind2[:c1]):
                curr = x
                while curr in elements2:
                    curr = sub_rule_2[curr]
                off2[i] = curr
            for i, x in enumerate(ind2[c2:]):
                curr = x
                while curr in elements2:
                    curr = sub_rule_2[curr]
                off2[i+c2] = curr

        elif self.crossover == 'CX':
            # find the cycle
            x = np.random.randint(0, self.problem_size)
            list_ind1 = list(ind1)
            visited = {x}
            visited_values = {ind1[x]}
            while ind2[x] not in visited_values:
                x = list_ind1.index(ind2[x])
                visited.add(x)
                visited_values.add(ind1[x])
            del list_ind1
            visited = list(visited)
            mask = np.zeros(self.problem_size, dtype=bool)
            mask[visited] = True
            off1[mask] = ind1[mask]
            off1[~mask] = ind2[~mask]
            off2[mask] = ind2[mask]
            off2[~mask] = ind1[~mask]
            assert all(x in off1 for x in range(self.problem_size))
            assert all(x in off2 for x in range(self.problem_size))

        elif self.crossover == 'OX':
            # random cut points
            c1 = random.randint(0, self.problem_size-1)
            c2 = random.randint(c1+1, self.problem_size)
            off1[c1:c2] = ind1[c1:c2]
            elements1 = set(ind1[c1:c2])
            off2[c1:c2] = ind2[c1:c2]
            elements2 = set(ind2[c1:c2])
            off1[np.r_[c2:self.problem_size, 0:c1]] = [x for x in ind2[np.r_[c2:self.problem_size, 0:c2]]
                                                       if x not in elements1]
            off2[np.r_[c2:self.problem_size, 0:c1]] = [x for x in ind1[np.r_[c2:self.problem_size, 0:c2]]
                                                       if x not in elements2]
        elif self.crossover == 'LOX':
            # random cut points
            c1 = random.randint(0, self.problem_size-1)
            c2 = random.randint(c1+1, self.problem_size)

            off1[c1:c2] = ind2[c1:c2]
            elements1 = set(ind2[c1:c2])
            off2[c1:c2] = ind1[c1:c2]
            elements2 = set(ind1[c1:c2])

            off1[np.r_[0:c1, c2:self.problem_size]] = [x for x in ind1 if x not in elements1]
            off2[np.r_[0:c1, c2:self.problem_size]] = [x for x in ind2 if x not in elements2]

        elif self.crossover == 'SCX':
            # off1[0] is already zero, as off1 = np.zeros(self.problem_size), but include just for clarity
            off1[0] = 0
            # Keep a set of nodes inserted in the offspring, so as to allow fast checkup through hashing
            elements = {0}
            list_ind1 = list(ind1)   # to avoid converting list(ind1) at every loop
            list_ind2 = list(ind2)
            for i in range(self.problem_size - 1):
                idx1 = list_ind1.index(off1[i])
                idx2 = list_ind2.index(off1[i])
                # first node visited after node off1[i] in ind1
                # if no node available, explore sequentially nodes 2, 3, ..., n
                while idx1 < self.problem_size and ind1[idx1] in elements:
                    idx1 += 1
                if idx1 < self.problem_size:
                    next_node_1 = ind1[idx1]
                else:
                    next_node_1 = 0
                    while next_node_1 in elements:
                        next_node_1 += 1
                cost1 = self.cost_matrix[off1[i], next_node_1]
                # same thing for parent 2
                while idx2 < self.problem_size and ind2[idx2] in elements:
                    idx2 += 1
                if idx2 < self.problem_size:
                    next_node_2 = ind2[idx2]
                else:
                    next_node_2 = 0
                    while next_node_2 in elements:
                        next_node_2 += 1
                cost2 = self.cost_matrix[off1[i], next_node_2]
                # compare costs of connection with newly chosen node
                if cost1 < cost2:
                    new_node = next_node_1
                else:
                    new_node = next_node_2
                off1[i+1] = new_node
                elements.add(new_node)
            return [off1]        # to comply with returned object type of other crossover types, return a list
        assert off1.shape[0] == off2.shape[0] == self.problem_size
        assert all(x in off1 for x in range(self.problem_size)) and all(x in off2 for x in range(self.problem_size))
        return off1, off2

    def _mutate_individual(self, ind):
        """ Mutate individual with a random 2-opt neighbour with probability self.mutation_prob """
        assert all(x in ind for x in range(self.problem_size))
        if random.random() <= self.mutation_prob:
            # random cut points
            c1 = random.randint(0, self.problem_size-1)
            c2 = random.randint(c1+1, self.problem_size)
            mutated_ind = ind[np.r_[0:c1, c2-1:c1-1:-1, c2:self.problem_size]]
            assert all(x in mutated_ind for x in range(self.problem_size))
            return mutated_ind
        return ind

    def solution_at_time(self, t):
        """
        Given a time t, return the couple (solution, score) that was found at that time. If no solution was found,
        return the tuple (np.array(), -1)
        REMARK: this search could be made faster if the logger was a binary tree
        """
        # recall that (from Python 3) dictionary keys are assured to be returned in the insertion order, so that
        # self.logger.keys are sorted in increasing order by time
        if t < list(log_time for log_time in self.logger.keys())[0]:
            return None
        old_log_time = -1
        for log_time in self.logger.keys():
            if log_time > t:
                return self.logger[old_log_time]
            old_log_time = log_time
        return self.logger[old_log_time]




























