import torch
 as nn
import numpy as np

class MarkovChain:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.transition_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    def update_transitions(self, sequences):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_token = sequence[i]
                next_token = sequence[i + 1]
                self.transition_matrix[current_token, next_token] += 1

    def normalize(self):
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = np.divide(self.transition_matrix, row_sums[:, None], where=row_sums[:, None] != 0)

    def get_transition_prob(self, current_token, next_token):
        return self.transition_matrix[current_token, next_token]
