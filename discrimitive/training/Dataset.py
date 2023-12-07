import json
from torch.utils.data import Dataset

def open_json(filename):
    with open(f'../data/{filename}', 'r') as file:
        return json.load(file)

class QA_loader(Dataset):
    def __init__(self, filename):
        data = open_json(filename)
        self.questions = [item['question'] for item in data]
        self.answers = [item['answer'] for item in data]
        self.rating = [item['rating'] for item in data]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx], self.rating[idx]