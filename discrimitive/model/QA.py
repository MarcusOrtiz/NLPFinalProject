import torch
import torch.nn as nn

class QA(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim):
        super(QA, self).__init__()
        # Input Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hidden Layer Dimensions
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.hidden_layers, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, question, answer):
        print("check 1")
        print("Question:", question[0:10])
        print("Answer:", answer[0:10])
        print("Question type:", question.dtype)
        print("Answer type:", answer.dtype)
        question
        question_emb = self.embedding(question)
        answer_emb = self.embedding(answer)
        print("check 2")

        _, (question_hidden, _) = self.lstm(question_emb)
        _, (answer_hidden, _) = self.lstm(answer_emb)

        # Concatenate the final hidden states of answer and question
        out = torch.cat((question_hidden[-1, :, :], answer_hidden[-1, :, :]), dim=1)

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

