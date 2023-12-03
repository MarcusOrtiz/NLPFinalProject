import torch
import torch.nn as nn
from Embedding import Embedding

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        super(LSTM, self).__init__()
        # Input Layer
        self.embedding = Embedding()

        # Hidden Layer
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

