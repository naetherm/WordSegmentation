# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
  """Encodes a sequence of word embeddings"""
  def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
    super(Encoder, self).__init__()
    self.num_layers = num_layers
    self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                      batch_first=True, bidirectional=True, dropout=dropout)
      
  def forward(self, x, mask, lengths):
    """
    Applies a bidirectional GRU to sequence of embeddings x.
    The input mini-batch x needs to be sorted by length.
    x should have dimensions [batch, time, dim].
    """
    packed = pack_padded_sequence(x, lengths, batch_first=True)
    output, final = self.rnn(packed)
    output, _ = pad_packed_sequence(output, batch_first=True)

    # we need to manually concatenate the final states for both directions
    fwd_final = final[0:final.size(0):2]
    bwd_final = final[1:final.size(0):2]
    final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

    return output, final


class Decoder(nn.Module):
  """A conditional RNN decoder with attention."""
  
  def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                bridge=True):
    super(Decoder, self).__init__()
    
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.attention = attention
    self.dropout = dropout
              
    self.rnn = nn.GRU(
      emb_size + 2*hidden_size, 
      hidden_size, 
      num_layers,
      batch_first=True, 
      dropout=dropout)
              
    # to initialize from the final encoder state
    self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

    self.dropout_layer = nn.Dropout(p=dropout)
    self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                      hidden_size, bias=False)
      
  def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
    """Perform a single decoder step (1 word)"""

    # compute context vector using attention mechanism
    query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
    context, attn_probs = self.attention(
        query=query, proj_key=proj_key,
        value=encoder_hidden, mask=src_mask)

    # update rnn hidden state
    rnn_input = torch.cat([prev_embed, context], dim=2)
    output, hidden = self.rnn(rnn_input, hidden)
    
    pre_output = torch.cat([prev_embed, output, context], dim=2)
    pre_output = self.dropout_layer(pre_output)
    pre_output = self.pre_output_layer(pre_output)

    return output, hidden, pre_output
  
  def forward(self, trg_embed, encoder_hidden, encoder_final, 
              src_mask, trg_mask, hidden=None, max_len=None):
    """Unroll the decoder one step at a time."""
                                      
    # the maximum number of steps to unroll the RNN
    if max_len is None:
        max_len = trg_mask.size(-1)

    # initialize decoder hidden state
    if hidden is None:
        hidden = self.init_hidden(encoder_final)
    
    # pre-compute projected encoder hidden states
    # (the "keys" for the attention mechanism)
    # this is only done for efficiency
    proj_key = self.attention.key_layer(encoder_hidden)
    
    # here we store all intermediate hidden states and pre-output vectors
    decoder_states = []
    pre_output_vectors = []
    
    # unroll the decoder RNN for max_len steps
    for i in range(max_len):
        prev_embed = trg_embed[:, i].unsqueeze(1)
        output, hidden, pre_output = self.forward_step(
          prev_embed, encoder_hidden, src_mask, proj_key, hidden)
        decoder_states.append(output)
        pre_output_vectors.append(pre_output)

    decoder_states = torch.cat(decoder_states, dim=1)
    pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
    return decoder_states, hidden, pre_output_vectors  # [B, N, D]

  def init_hidden(self, encoder_final):
    """Returns the initial decoder state,
    conditioned on the final encoder state."""

    if encoder_final is None:
        return None  # start with zeros

    return torch.tanh(self.bridge(encoder_final)) 