# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import os
import sys
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _single
import torch.nn.functional as F

def make_positions(tensor, padding_idx):

  mask = tensor.ne(padding_idx).int()

  return (torch.cumsum(mask, dim=1).type_as(mask)*mask).long() + padding_idx

def strip_pad(tensor, pad):
  return tensor[tensor.ne(pad)]

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_threads():
  """ Returns the number of available threads on a posix/win based system """
  if sys.platform == 'win32':
    return (int)(os.environ['NUMBER_OF_PROCESSORS'])
  else:
    return (int)(os.popen('grep -c cores /proc/cpuinfo').read())