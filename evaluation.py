# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import os
import sys
import argparse

from utils import lazy_scan_vocabulary, sent_to_xy, correct, correct_cnn

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time


def main(argv=None):

  if argv is None:
    argv = sys.argv[1:]

  parser = argparse.ArgumentParser()

  args = parser.parse_args()

  seq_acc = 0.0
  seg_acc = 0.0
  seg_tot = 0.0
  fn_count = 0.0
  tn_count = 0.0
  tp_count = 0.0
  fp_count = 0.0
  precision = 0.0
  recall = 0.0
  f_score = 0.0


if __name__ == '__main_':
  main()