## A repository containing all the code and data for solving the TSP problem
### Structure
The travelling salesman problem here is represented by some points on the [0, 1] x [0, 1] square. The main repository 
contains:

- **data_generator.py**: all the utility functions to reproduce thea actual data used in the experiments. Seed is
 provided for reproducibility. This script saves the datasets provided in three locations: in experimental_data it saves
 the .pkl files of the raw arrays, in Homework1 it saves the distance matrix in .dat format ready for the OPL models and 
 in Homework2 it saves another .pkl file with the same distance matrix read for the genetic algorithm
- **read_and_plot_results.py**: script for retrieving the results saved after the experiments, plot them and save the 
plots to the folder experimental_data

The **experimental_data** folder contains: the datasets generated for the experiments, their plots and the plots of the
experiments' results

The **Homework1** folder contains the OPL model, the .dat files and the results of the experiments run with the OPL 
solver

The **Homework2** folder contains the genetic algorithm in *TSPSolver*, the script for running the entire experiment (
complete of automatic parameters selection, repeated experiments and saving of experiments) and the experiments' 
results.