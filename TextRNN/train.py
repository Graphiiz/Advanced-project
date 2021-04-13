# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import random
import json

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

if __name__=='__main__':
    config = Config()
    train_file = '/ext3/Text-Classification-Models-Pytorch/data/ag_news.train' #to be modified
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = '/ext3/Text-Classification-Models-Pytorch/data/ag_news.test' #to be modified
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    w2v_file = '/ext3/Text-Classification-Models-Pytorch/data/glove.840B.300d.txt' #to be modified

    initialize(seed=config.seed)
    
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = TextRNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=config.max_epochs)
    ##############################################################
    
    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    test_accuracies = []
    test_losses = []

    tracked_val = 0
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = model.run_epoch(dataset.train_iterator, dataset.val_iterator, dataset.test_iterator, i)

        train_losses.append(float(train_loss)) #cast float tinto json saveable
        train_accuracies.append(float(train_acc))

        val_losses.append(float(val_loss))
        val_accuracies.append(float(val_acc))
       
        test_losses.append(float(test_loss))
        test_accuracies.append(float(test_acc))

        #tracked_val = val_acc

        #scheduler.step()

    log_dict = {'train_loss': train_losses, 'test_loss': test_losses, 'val_loss': val_losses, 'val_acc': val_accuracies,
                            'train_acc': train_accuracies, 'test_acc': test_accuracies, 'best_test_acc': max(test_accuracies),
                            'best_val_acc': max(val_accuracies), 'hidden_layers': config.hidden_layers, 'hidden_size': config.hidden_size,
                            'epoch': config.max_epochs, 'lr': config.lr, 'batch_size': config.batch_size, 'max_sen_len': config.max_sen_len,
                            'dropout': config.dropout_keep, 'momentum': config.momentum, 'seed': config.seed }
    with open(f'train_rnn.json', 'w') as outfile:
        json.dump(log_dict, outfile)

    # train_acc = evaluate_model(model, dataset.train_iterator)
    # val_acc = evaluate_model(model, dataset.val_iterator)
    # test_acc = evaluate_model(model, dataset.test_iterator)

    # print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    # print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    # print ('Final Test Accuracy: {:.4f}'.format(test_acc))