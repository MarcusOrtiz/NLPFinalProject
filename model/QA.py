import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim):
        super(LSTM, self).__init__()
        # Input Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Hidden Layer
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x, question, answer):
        question_emb = self.embedding(question)
        answer_emb = self.embedding(answer)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        _, (question_hidden, _) = self.lstm(question_emb)
        _, (answer_hidden, _) = self.lstm(answer_emb)

        # Concatenate the final hidden states of answer and question
        out = torch.cat((question_hidden[-1, :, :], answer_hidden[-1, :, :]), dim=1)

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

