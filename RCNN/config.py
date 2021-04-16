# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 64
    output_size = 4
    max_epochs = 100
    hidden_size_linear = 64
    lr = 0.1
    batch_size = 128
    max_sen_len = None # Sequence length for RNN
    dropout_keep = 0.4
    momentum = 0.9
    seed = 5
    rho = 0.01

