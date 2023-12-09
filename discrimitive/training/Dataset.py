from torch.utils.data import Dataset
from collections import Counter
from ..utils import open_json
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def build_vocab(texts):

    tokens = [token for text in texts for token in tokenizer.tokenize(text)]
    token_counts = Counter(tokens)

    # Create vocabulary mapping each token to a unique index
    vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items(), start=1)}

    # Add a special token for unknown words
    vocab['<UNK>'] = 0
    return vocab


class QA_loader(Dataset):
    def __init__(self, data):
        self.questions = questions
        self.answers = answers
        self.ratings = ratings
        self.vocab = vocab

        self.question
        data = open_json(filename)
        self.questions = [item['Question'] for item in data]
        self.answers = [item['Answer'] for item in data]
        self.rating = [item['Rating'] for item in data]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_answer = self.data
        return self.questions[idx], self.answers[idx], self.rating[idx]
    def convert_to_tensor(self, text):
        # Tokenize and convert to indices
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokenize(text)]
        return torch.tensor(indices, dtype=torch.long)