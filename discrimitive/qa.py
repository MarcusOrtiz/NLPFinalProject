import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 200
HIDDEN_DIM = 1200
HIDDEN_LAYERS = 2
OUTPUT_DIM = 1
INPUT_DIM = EMBEDDING_DIM
DROPOUT_OUT = 0.2
DROPOUT_LSTM = 0.8
PADDING_INDEX = 1

class QA(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim):
        super(QA, self).__init__()
        # Input Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_INDEX)

        # Hidden Layer Dimensions
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # Dropout Layer
        self.dropout = nn.Dropout(DROPOUT_OUT)

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.hidden_layers, dropout=DROPOUT_LSTM, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, question, answer):
        question_emb = self.embedding(question)
        answer_emb = self.embedding(answer)

        _, (question_hidden, _) = self.lstm(question_emb)
        _, (answer_hidden, _) = self.lstm(answer_emb)

        # Concatenate the final hidden states of answer and question
        concatenated = torch.cat((question_hidden[-1, :, :], answer_hidden[-1, :, :]), dim=1)

        dropout_out = self.dropout(concatenated)

        activated = F.relu(dropout_out)

        out = self.fc(activated)
        return out

def create_model(vocab_size):
    return QA(INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS, OUTPUT_DIM, vocab_size, EMBEDDING_DIM)

