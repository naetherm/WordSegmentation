# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abstract_model import AbstractModel

class LSTMModel(AbstractModel):

  def __init__(
    self, 
    embedding_dim, 
    hidden_dim, 
    vocab_size, 
    tags_size,
    num_layers=1, 
    bias=True, 
    dropout=0.1, 
    bidirectional=False):
    super(LSTMModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim,
      num_layers = num_layers,
      bias = bias,
      dropout = dropout,
      bidirectional=bidirectional)
    self.hidden2tag = nn.Linear(hidden_dim * (1 + self.bidirectional), tags_size)
    self.hidden = self.init_hidden() # hidden, cell

  def forward(self, char_idxs):
    embeds = self.embeddings(char_idxs)
    lstm_out, self.hidden = self.lstm(
      embeds.view(len(char_idxs), 1, -1), self.hidden)
    tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
    tag_scores = F.log_softmax(tag_space, dim=1)

    return tag_scores

  def init_hidden(self):
    # (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1 + self.bidirectional, 1, self.hidden_dim),
            torch.zeros(1 + self.bidirectional, 1, self.hidden_dim))


class GRUModel(AbstractModel):

  def __init__(
    self, 
    embedding_dim, 
    hidden_dim, 
    vocab_size, 
    tags_size,
    num_layers=1, 
    bias=True, 
    dropout=0.1, 
    bidirectional=False):
    super(LSTMModel, self).__init__()

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.GRU(embedding_dim, hidden_dim,
      num_layers = num_layers,
      bias = bias,
      dropout = dropout,
      bidirectional=bidirectional)
    self.hidden2tag = nn.Linear(hidden_dim * (1 + self.bidirectional), tags_size)
    self.hidden = self.init_hidden() # hidden, cell

  def forward(self, char_idxs):
    embeds = self.embeddings(char_idxs)
    lstm_out, self.hidden = self.lstm(
      embeds.view(len(char_idxs), 1, -1), self.hidden)
    tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
    tag_scores = F.log_softmax(tag_space, dim=1)

    return tag_scores

  def init_hidden(self):
    # (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1 + self.bidirectional, 1, self.hidden_dim),
            torch.zeros(1 + self.bidirectional, 1, self.hidden_dim))
