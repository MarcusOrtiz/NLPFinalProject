from QA import LSTM

# Define dimensions for LSTM
input_dim = 10
hidden_dim = 100
hidden_layers = 1
output_dim = 1
vocab_size = 10000
embedding_dim = 100

# Create the LSTM model
model = LSTM(input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim)