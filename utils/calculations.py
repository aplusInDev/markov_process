import numpy as np
from scipy import linalg


def compute_stationary_distribution(transition_matrix):
    eigenvals, eigenvecs = linalg.eig(transition_matrix.T)
    stationary = eigenvecs[:, np.argmax(eigenvals.real)].real
    return stationary / stationary.sum()

def calculate_mixing_time(distributions, pi, threshold=0.01):
    distances = np.max(np.abs(distributions - pi), axis=1)
    return np.where(distances < threshold)[0][0]
