class Config:
    def __init__(self):
        self.batch_size = 8
        self.epochs = 5
        self.learning_rate = 0.001
        self.max_length = 512
        self.temperature = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
