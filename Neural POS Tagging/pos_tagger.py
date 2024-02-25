import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
from conllu import parse

with open("UD_English-Atis/en_atis-ud-train.conllu", "r", encoding="utf-8") as f:
    data = f.read()
    train_sentences = parse(data)
    
tokens = []
for sentence in train_sentences:
    for token in sentence:
        tokens.append(token)
        
df = pd.DataFrame(tokens)
pos_class = {token: idx for idx, token in enumerate(df['upos'].unique())}
pos_class_rev = {idx: token for idx, token in enumerate(df['upos'].unique())}
vocab = df['form'].unique()
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
vocab = np.append(vocab, [START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
vocab = {token: idx for idx, token in enumerate(vocab)}
vocab_rev = {idx: token for idx, token in enumerate(vocab)}

class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_layers, hidden_dims, output_dim, activation = 'relu', p = 1, s = 1):
        super(FeedForwardNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([])
        input_dim = (p+s+1)*embedding_dim
        
        for layer in range(hidden_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dims[layer]))
            input_dim = hidden_dims[layer]
            
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x).reshape(len(x),-1)
        for layer in self.layers:
            embedded = self.activation(layer(embedded))
        output = self.final_layer(embedded)
        return output
    
    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(self.softmax(out), axis = 1)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, output_dim,activation = 'relu', classification_dim = 64, rnn_type='rnn', bidirectionality = False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx= vocab[PAD_TOKEN])
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional = bidirectionality)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional = bidirectionality)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional = bidirectionality)
        self.fc1 = nn.Linear((bidirectionality+1)*hidden_dim, classification_dim)
        self.fc2 = nn.Linear(classification_dim, output_dim)
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc2(self.act(self.fc1(output)))
        return output
    
    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(self.softmax(out), axis = 2)

def tokenize_ffn(input_sentence, p = 1, s = 1):
    tokens = word_tokenize(input_sentence)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    test_tokens = [vocab[START_TOKEN]]*p + [vocab[token] if token in vocab else vocab[UNKNOWN_TOKEN] for token in tokens]+ [vocab[END_TOKEN]]*s
    ret = []
    for token in range(len(test_tokens)-s-p):
        ret.append(test_tokens[token:token+p+s+1])
    return torch.tensor(ret), tokens

def tokenize_rnn(input_sentence):
    tokens = word_tokenize(input_sentence)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    test_tokens = [vocab[START_TOKEN]] + [vocab[token] if token in vocab else vocab[UNKNOWN_TOKEN] for token in tokens]+ [vocab[END_TOKEN]]
    return torch.tensor(test_tokens), tokens

args = sys.argv

opt = args[1]

if opt == '-f':
    model_type = 'FFN'
elif opt == '-r':
    model_type = 'RNN' 
elif opt == '-g':
    model_type = 'GRU' 
elif opt == '-l':
    model_type = 'LSTM'
else:
    raise ValueError('Command line arguement not supported')

input_sentence = input('Input sentence: ')

embedding_dims = 100

if model_type != 'FFN':
    pos_class['<SPECIAL>'] = len(pos_class)

if model_type == 'FFN':
    layers = 2
    activation = 'relu'
    context = 1
    hidden_dims = [64,32]
    model = FeedForwardNN(len(vocab), embedding_dims,layers, hidden_dims, len(pos_class), activation = activation, p = context, s = context)
    model.load_state_dict(torch.load(f'./ffn/model_{context}_{1}_{0}.pth'))
elif model_type == 'RNN':
    bidirectional = True
    model_type = 'rnn'
    layers = 1
    hidden_dim = 64
    activation = 'relu'
    class_dim = 128
    model = RNNModel(len(vocab), embedding_dims, layers, hidden_dim, len(pos_class)-1, activation = activation, classification_dim = class_dim, rnn_type = 'rnn', bidirectionality = bidirectional)
    model.load_state_dict(torch.load(f'./RNN/model_{bidirectional}_{layers}_{activation}_{hidden_dim}_{class_dim}.pth'))
elif model_type == 'GRU':
    bidirectional = True
    model_type = 'gru'
    layers = 1
    hidden_dim = 128
    activation = 'tanh'
    class_dim = 128
    model = RNNModel(len(vocab), embedding_dims, layers, hidden_dim, len(pos_class)-1, activation = activation, classification_dim = class_dim, rnn_type = 'gru', bidirectionality = bidirectional)
    model.load_state_dict(torch.load(f'./GRU/model_{bidirectional}_{layers}_{activation}_{hidden_dim}_{class_dim}.pth'))
elif model_type == 'LSTM':
    bidirectional = True
    model_type = 'lstm'
    layers = 1
    hidden_dim = 64
    activation = 'tanh'
    class_dim = 64
    model = RNNModel(len(vocab), embedding_dims, layers, hidden_dim, len(pos_class)-1, activation = activation, classification_dim = class_dim, rnn_type = 'lstm', bidirectionality = bidirectional)
    model.load_state_dict(torch.load(f'./LSTM/model_{bidirectional}_{layers}_{activation}_{hidden_dim}_{class_dim}.pth'))


if model_type == 'FFN':
    test_tokens, input_tokens = tokenize_ffn(input_sentence, context, context)
    predict = model.predict(test_tokens)
    ret = zip(input_tokens, [pos_class_rev[pred.item()] for pred in predict])
    for inp, pred in ret:
        print(inp, pred)
else:
    test_tokens, input_tokens = tokenize_rnn(input_sentence)
    predict = model.predict(test_tokens.reshape(1,-1))[0]
    ret = zip(input_tokens, [pos_class_rev[pred.item()] for pred in predict][1:-1])
    for inp, pred in ret:
        print(inp, pred)
    
  

