import torch
import torch.nn as nn

EMBEDDING_DIM = 15
HIDDEN_DIM = 80
HIDDEN_LAYERS = 1
OUTPUT_DIM = 1
INPUT_DIM = EMBEDDING_DIM

PADDING_INDEX = 1

class QA(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim):
        super(QA, self).__init__()
        # Input Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_INDEX)

        # Hidden Layer Dimensions
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.hidden_layers, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, question, answer):
        question_emb = self.embedding(question)
        answer_emb = self.embedding(answer)

        _, (question_hidden, _) = self.lstm(question_emb)
        _, (answer_hidden, _) = self.lstm(answer_emb)

        # Concatenate the final hidden states of answer and question
        out = torch.cat((question_hidden[-1, :, :], answer_hidden[-1, :, :]), dim=1)

        out = self.fc(out)
        return out

def create_model(vocab_size):
    return QA(INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS, OUTPUT_DIM, vocab_size, EMBEDDING_DIM)

