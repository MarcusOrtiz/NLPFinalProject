from LSTM import LSTM

# Define dimensions for LSTM
input_dim = 10
hidden_dim = 100
hidden_layers = 1
output_dim = 1

# Create the LSTM model
model = LSTM(input_dim, hidden_dim, hidden_layers, output_dim)