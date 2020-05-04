# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import torch
import torch.nn as nn

class AbstractModel(nn.Module):
  """ Abstract wrapper arround the nn.Module of torch """
  
  def __init__(self):
    """ Constructor """
    super(AbstractModel, self).__init__()

