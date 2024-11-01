import torch

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        return encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()

def get_data_loader(data_path, tokenizer, batch_size=8, max_length=512):
    dataset = TextDataset(data_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
