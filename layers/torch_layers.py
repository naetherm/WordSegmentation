# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import math

from torch_utils import make_positions, strip_pad
import torch
import torch.nn as nn
from torch.nn.modules.utils import _single
import torch.nn.functional as F


class FSeqConvTBC(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, padding=0):

    super(FSeqConvTBC, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _single(kernel_size)
    self.padding = _single(padding)

    self.weight = torch.nn.Parameter(
      torch.Tensor(self.kernel_size[0], in_channels, out_channels)
    )
    self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

  def forward(self, input):
    return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

  def __repr__(self):
    s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    s += ')'

    return s.format(name=self.__class__.__name__, **self.__dict__)


class LearnedPositionalEmbedding(nn.Embedding):

  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int
  ):
    super(LearnedPositionalEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)

    if self.padding_idx is not None:
      self.max_positions = self.num_embeddings - self.padding_idx - 1
    else:
      self.max_positions = self.num_embeddings

  def forward(self, input, incremental_state=None, positions=None):

    assert(
      (positions is None) or (self.padding_idx is None),

    ), "If positions is pre-computed then padding_idx should not be set!"

    if positions is None:
      if incremental_state is not None:
        positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
      else:
        positions = make_positions(input, self.padding_idx)

    return super().forward(positions)




class SinusoidalPositionalEmbedding(nn.Module):
  """This module produces sinusoidal positional embeddings of any length.
  Padding symbols are ignored.
  """

  def __init__(self, embedding_dim, padding_idx, init_size=2048):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.weights = SinusoidalPositionalEmbedding.get_embedding(
      init_size, embedding_dim, padding_idx
    )
    self.register_buffer("_float_tensor", torch.FloatTensor(1))
    self.max_positions = int(1e5)

  @staticmethod
  def get_embedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int = None
  ):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
      1
    ) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
      num_embeddings, -1
    )
    if embedding_dim % 2 == 1:
      # zero pad
      emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
      emb[padding_idx, :] = 0
    return emb

  def forward(
    self,
    input,
    incremental_state = None,
    timestep = None,
    positions = None,
  ):
    """Input is expected to be of size [bsz x seqlen]."""
    #bspair = torch.onnx.operators.shape_as_tensor(input)
    bsz, seq_len, _ = list(input.size())
    max_pos = self.padding_idx + 1 + seq_len
    if self.weights is None or max_pos > self.weights.size(0):
      # recompute/expand embeddings if needed
      self.weights = SinusoidalPositionalEmbedding.get_embedding(
        max_pos, self.embedding_dim, self.padding_idx
      )
    self.weights = self.weights.to(self._float_tensor)

    if incremental_state is not None:
      # positions is the same for every token when decoding a single step
      pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len

      return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

    positions = utils.make_positions(
      input, self.padding_idx, onnx_trace=self.onnx_trace
    )

    return (
      self.weights.index_select(0, positions.view(-1))
      .view(bsz, seq_len, -1)
      .detach()
    )


def ConvEmbedding(num_embeddings, embedding_dim, padding_idx):
  m = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
  nn.init.normal_(m.weight, 0, 0.1)
  nn.init.constant_(m.weight[padding_idx], 0)
  return m

def PositionEmbedding(num_embeddings, embedding_dim, padding_idx, learn_embedding=True):
  if learn_embedding:
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
  else:
    m = SinusoidalPositionalEmbedding(embedding_dim=embedding_dim, padding_idx=padding_idx)
    return m

def Linear(in_features, out_features, dropout=0):
  """Weight-normalized Linear layer (input: N x T x C)"""
  m = nn.Linear(in_features, out_features)
  nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
  nn.init.constant_(m.bias, 0)
  return nn.utils.weight_norm(m)

def extend_conv_spec(convolutions):
  """
  Extends convolutional spec that is a list of tuples of 2 or 3 parameters
  (kernel size, dim size and optionally how many layers behind to look for residual)
  to default the residual propagation param if it is not specified
  """
  extended = []
  for spec in convolutions:
    if len(spec) == 3:
      extended.append(spec)
    elif len(spec) == 2:
      extended.append(spec + (1,))
    else:
      raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
  return tuple(extended)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
  m = FSeqConvTBC(in_channels, out_channels, kernel_size, **kwargs)
  std = math.sqrt((4*(1.0-dropout))/ (m.kernel_size[0] * in_channels))
  nn.init.normal_(m.weight, mean=0, std=std)
  nn.init.constant_(m.bias, 0)

  return nn.utils.weight_norm(m, dim=2)