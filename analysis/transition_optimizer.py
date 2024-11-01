class TransitionOptimizer:
    def __init__(self, markov_chain):
        self.markov_chain = markov_chain

    def optimize_transitions(self, target_distribution, lr=0.01, steps=1000):
        for _ in range(steps):
            current_distribution = self.markov_chain.transition_matrix.mean(axis=0)
            error = target_distribution - current_distribution
            self.markov_chain.transition_matrix += lr * error.reshape(-1, 1)
            self.markov_chain.normalize()
        return self.markov_chain.transition_matrix
