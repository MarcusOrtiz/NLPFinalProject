import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 130
HIDDEN_DIM = 300
HIDDEN_LAYERS = 2
OUTPUT_DIM = 4
INPUT_DIM = EMBEDDING_DIM
DROPOUT_OUT = 0.35
DROPOUT_LSTM = 0.35
PADDING_INDEX = 1

class QA(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim):
        super(QA, self).__init__()
        # Input Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_INDEX)

        # Hidden Layer Dimensions
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # LSTM Layer
        self.lstm_question = nn.LSTM(input_dim, self.hidden_dim, self.hidden_layers, dropout=DROPOUT_LSTM, batch_first=True)
        self.lstm_answer = nn.LSTM(input_dim, self.hidden_dim, self.hidden_layers, dropout=DROPOUT_LSTM, batch_first=True)
        # Dropout Layer
        self.dropout = nn.Dropout(DROPOUT_OUT)

        # Output Layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, question, answer):
        question_emb = self.embedding(question)
        answer_emb = self.embedding(answer)

        _, (question_hidden, _) = self.lstm_question(question_emb)
        _, (answer_hidden, _) = self.lstm_answer(answer_emb)

        # Concatenate the final hidden states of answer and question
        concatenated = torch.cat((question_hidden[-1, :, :], answer_hidden[-1, :, :]), dim=1)

        dropout_out = self.dropout(concatenated)

        out = self.fc(dropout_out)
        return out

def create_model(vocab_size):
    return QA(INPUT_DIM, HIDDEN_DIM, HIDDEN_LAYERS, OUTPUT_DIM, vocab_size, EMBEDDING_DIM)

