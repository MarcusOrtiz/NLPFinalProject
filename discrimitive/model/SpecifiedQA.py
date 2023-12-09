from .QA import QA


def create_model(vocab_size):
    input_dim = 10
    hidden_dim = 100
    hidden_layers = 1
    output_dim = 1
    embedding_dim = 100

    vocab_size = vocab_size
    # Return the LSTM model
    return QA(input_dim, hidden_dim, hidden_layers, output_dim, vocab_size, embedding_dim)
