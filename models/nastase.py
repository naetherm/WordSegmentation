# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import torch
import torch.nn as nn
import torch.nn.functional as F

from word_segmentation.models.abstract_model import AbstractModel
from word_segmentation.layers.torch_nastase_layers import Encoder, Decoder


class Generator(nn.Module):
  """Define standard linear + softmax generation step."""
  def __init__(self, hidden_size, vocab_size):
    super(Generator, self).__init__()
    self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

  def forward(self, x):
    return F.log_softmax(self.proj(x), dim=-1)


class NastaseModel(AbstractModel):

  def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
    super(NastaseModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.trg_embed = trg_embed
    self.generator = generator

  def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
    """Take in and process masked src and target sequences."""
    encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
    return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
  
  def encode(self, src, src_mask, src_lengths):
    return self.encoder(self.src_embed(src), src_mask, src_lengths)
  
  def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
              decoder_hidden=None):
    return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                        src_mask, trg_mask, hidden=decoder_hidden)