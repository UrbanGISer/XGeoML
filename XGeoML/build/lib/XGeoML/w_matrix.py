import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

#01 Define spatial weight matrix
class w_matrix:
    @staticmethod
    def _compute_distance_matrix(df, xcoord, ycoord, metric='euclidean', **kwargs):
        points = df[[xcoord, ycoord]].values
        dismatrix = cdist(points, points, metric=metric, **kwargs)
        return dismatrix

    @staticmethod
    def _compute_weights_row(i, dismatrix, bandwidths, kernel_type):
        if kernel_type == 'Binary':
            return (dismatrix[i] <= bandwidths[i]).astype(float)
        elif kernel_type == 'Gaussian':
            return np.exp(-dismatrix[i]**2 / (2 * bandwidths[i]**2))
        elif kernel_type == 'GaussianBinary':
            gaussian_weights = np.exp(-dismatrix[i]**2 / (2 * bandwidths[i]**2))
            binary_weights = (dismatrix[i] <= bandwidths[i]).astype(float)
            return gaussian_weights * binary_weights
    @staticmethod
    def spatial_weight(df, xcoord, ycoord, fix=False, bandwidth=10, kernel_type='Binary', n_jobs=-1):
        dismatrix = w_matrix._compute_distance_matrix(df, xcoord, ycoord)
        bandwidths = np.zeros(dismatrix.shape[0])
        if fix:
            bandwidths.fill(bandwidth)
        else:
            for i in range(dismatrix.shape[0]):
                sorted_distances = np.sort(dismatrix[i])
                bandwidths[i] = sorted_distances[min(bandwidth, len(sorted_distances)-1)]
        weights = Parallel(n_jobs=n_jobs)(
            delayed(w_matrix._compute_weights_row)(i, dismatrix, bandwidths, kernel_type) for i in range(dismatrix.shape[0])
        )
        return np.array(weights)**0.5
    def from_libpysal(w):
        W_matrix, ids = w.full()
        W_symmetric = np.maximum(W_matrix, W_matrix.T)
        np.fill_diagonal(W_symmetric, 1)
        return W_symmetric**0.5
