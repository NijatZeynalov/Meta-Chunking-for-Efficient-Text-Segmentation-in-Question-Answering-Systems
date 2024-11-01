import numpy as np

class TemperatureEffects:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    def adjust_temperature(self, temperature):
        adjusted_matrix = np.power(self.transition_matrix, 1 / temperature)
        row_sums = adjusted_matrix.sum(axis=1)
        return np.divide(adjusted_matrix, row_sums[:, None], where=row_sums[:, None] != 0)
