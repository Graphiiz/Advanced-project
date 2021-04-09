# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 128
    bidirectional = True
    output_size = 4
    max_epochs = 100
    lr = 0.25
    batch_size = 128
    max_sen_len = 50 # Sequence length for RNN
    dropout_keep = 0.5
    momentum = 0.9
    seed = 1
    factor = 0.1
    patience = 5