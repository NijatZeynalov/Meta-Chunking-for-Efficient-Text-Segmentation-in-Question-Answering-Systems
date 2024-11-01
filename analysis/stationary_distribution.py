import numpy as np

class StationaryDistribution:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def calculate(self, tol=1e-6):
        n = self.transition_matrix.shape[0]
        pi = np.ones(n) / n
        diff = tol + 1
        while diff > tol:
            new_pi = pi.dot(self.transition_matrix)
            diff = np.linalg.norm(new_pi - pi)
            pi = new_pi
        return pi
