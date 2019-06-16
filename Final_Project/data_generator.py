import _pickle as pkl
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numpy.linalg import norm


def gaussian_blob(n, sigma_sq, centers=None, seed=None):
    """
    Generates n points according to a gaussian distribution divided into n_blobs clusters,
    all with variance sigma_sq.
    If centers is int, the centers are initialized randomly in [0,1]x[0,1], otherwise can
    be input in an array of shape (n_blobs, 2), If centers is None, 3 random centers are generated
    Returns an array of 2D points in the square [0,1]x[0,1]
    """
    np.random.seed(seed)

    # make blob and clip values outside [0,1]x[0,1] box
    X, _ = make_blobs(n, centers=centers, cluster_std=sigma_sq, center_box=(0., 1.))
    X[:, 0] = np.apply_along_axis(lambda x: np.clip(x, 0, 1), 0, X[:, 0])
    X[:, 1] = np.apply_along_axis(lambda x: np.clip(x, 0, 1), 0, X[:, 1])

    return X


def grid_blob(n, axis_density, seed=None):
    """
    Randomly generates n points in a grid of contained in [0,1]x[0,1],
    where each axis is equally divided into axis_density points
    """
    np.random.seed(seed)

    # To generate 2D points, simply generate points ar random and independently for every dim
    x = np.linspace(0, 1, axis_density)[np.random.randint(0, axis_density, n)]
    y = np.linspace(0, 1, axis_density)[np.random.randint(0, axis_density, n)]

    return np.array(list(zip(x, y)))


def plot_blob(blob, save=False, path=None, show_plot=True):
    """ Plots a blob, saving to path if save=True and path specified """
    plt.figure()
    plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], '-', c='k')
    plt.plot(blob[:, 0], blob[:, 1], '.', c='r')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    if save and path is not None:
        plt.savefig(path)

    if show_plot:
        plt.show()


def distance_matrix(blob):
    """
    Given the set of points, return the distance matrix
    D_ij = distance(blob[i], blob[j])
    The distance is simply computed as the euclidean distance
    """
    n = blob.shape[0]
    d = np.array([[norm(blob[i, :]-blob[j, :]) for j in range(n)] for i in range(n)])
    return d


def create_dat_file(dist_mat, path):
    """
    Given the distance matrix create the .dat file to feed the OPL model with.
    The file is saved at the specified path.
    """
    N = dist_mat.shape[0]
    with open(path, 'w') as file:
        file.write('N = ' + str(N) + '; \n ')
        file.write('C= [\n')
        for i in range(N):
            file.write('[' +
                       ' '.join(str(x) for x in dist_mat[i, :]) +
                       '],\n'
                       )
        file.write('\n];')


'''
# for saving to Homework1_original OPL folder
def main_OPL():
    blob1 = gaussian_blob(80, 0.02, 3)
    blob2 = grid_blob(100, 1000)
    blob3 = np.vstack([blob1, blob2])
    #plot_blob(blob3)
    
    dist_mat = distance_matrix(blob1)
    saving_path = 'Homework1_original/Homework_1_OPL/HHomework_1.dat'
    print('saving distance matrix of shape', dist_mat.shape, 'in .dat format at path', saving_path)
    create_dat_file(dist_mat, saving_path)

main_OPL()

# for saving to homework 2 folder
def main_genetic():
    blob1 = gaussian_blob(100, 0.02, 3)
    blob2 = grid_blob(100, 1000)
    blob3 = np.vstack([blob1, blob2])
    # plot_blob(blob3)

    dist_mat = distance_matrix(blob3)
    saving_path = 'Homework2/dist_mat_try.pkl'
    print('saving distance matrix of shape', dist_mat.shape, 'at path', saving_path)
    with open(saving_path, 'wb') as file:
        pkl.dump(dist_mat, file)

main_genetic()
'''

# used to generate the data fro the actual experiments
def main():
    # set seed for reproducibility
    np.random.seed(48)

    # data are saved (for convenience) in three different locations: a folder where we save the actual points in pickle
    # format (and relative plots), a .dat file in Homework1 folder for the OPL models and a distance matrix in Homework2
    # for TSPSolver
    experimetal_data_dir = 'experimental_data'
    if not os.path.isdir(experimetal_data_dir):
        os.mkdir(experimetal_data_dir)

    # build data at various sizes. We use half gaussian points and half grid points
    SAMPLE_SIZES = [100, 1000, 3000]
    for sample_size in SAMPLE_SIZES:
        print('creating and saving data for sample size', sample_size)
        gaussian_points = gaussian_blob(sample_size, sigma_sq=0.02, centers=3)
        grid_points = grid_blob(sample_size, axis_density=1000)
        all_points = np.vstack([gaussian_points, grid_points])
        path_plot = os.path.join(experimetal_data_dir, 'experimental_data_{}.png'.format(sample_size))
        plot_blob(all_points, save=True, path=path_plot, show_plot=False)
        # save points to pickle
        pkl_path = os.path.join(experimetal_data_dir, 'points_{}.pkl'.format(sample_size))
        with open(pkl_path, 'wb') as file:
            pkl.dump(all_points, file)
        # save distance matrix for Homework2
        dist_mat = distance_matrix(all_points)
        homework2_path = 'Homework2/dist_mat_{}.pkl'.format(sample_size)
        with open(homework2_path, 'wb') as file:
            pkl.dump(dist_mat, file)
        # save dat file
        dat_path = 'Homework1/dist_mat_{}.dat'.format(sample_size)
        create_dat_file(dist_mat, dat_path)

main()



























