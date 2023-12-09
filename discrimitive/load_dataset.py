from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchtext.vocab import vocab
from collections import Counter
import torch
import json

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def open_json(filename):
    with open(f'../data/{filename}', 'r') as file:
        return json.load(file)

def tokenize(text):
    return tokenizer.tokenize(text)

def build_vocab(text_iterable):
    counter = Counter()
    for text in text_iterable:
        counter.update(tokenize(text))
    return vocab(counter, specials=['<UNK>'])

class QA_loader(Dataset):
    def __init__(self, data, vocab, max_length=100):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_answer = self.data[idx]
        question = self.preprocess(question_answer['Question'])
        answer = self.preprocess(question_answer['Answer'])
        rating = torch.tensor(question_answer['Rating'], dtype=torch.int)
        return question, answer, rating

    def preprocess(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_length]
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices, dtype=torch.long)


def load_datasets(train_filename, val_filename, test_filename):
    train_data, val_data, test_data = open_json(train_filename), open_json(val_filename), open_json(test_filename)
    vocab = build_vocab([item['Question'] for item in train_data] + [item['Answer'] for item in train_data])
    qa_train_dataset = QA_loader(train_data, vocab)
    qa_val_dataset = QA_loader(val_data, vocab)
    return qa_val_dataset, qa_train_dataset, vocab






