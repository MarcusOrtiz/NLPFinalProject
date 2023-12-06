import torch as nn

class Embedding(nn.Module):

    def __init__(self):
        super(Embedding, self).__init__()
        self.vocab_size = 10000
        self.embedding_dim = 100
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, x):
        return self.embedding(x)