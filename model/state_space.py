import numpy as np

class StateSpace:
    def __init__(self, vocab):
        self.vocab = vocab
        self.state_space = {token: idx for idx, token in enumerate(vocab)}

    def get_state(self, token):
        return self.state_space.get(token, None)

    def get_token(self, state):
        return self.vocab[state] if state < len(self.vocab) else None

    def state_space_size(self):
        return len(self.vocab)
