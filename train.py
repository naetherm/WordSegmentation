# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import os
import sys
import argparse
import time
import logging

import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import lazy_scan_vocabulary, sent_to_xy
from torch_utils import get_threads

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



def batched_train(model, args):
  pass


def train(model, args, char2idx, idx2char, use_gpu=False):
  print("Un train")
  batch_size = args.batch_size

  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  is_cuda = use_gpu and torch.cuda.is_available()
  if is_cuda:
    LOGGER.info("CUDA is available")
    model = model.cuda()
  else:
    LOGGER.info("CPU mode")

  # Create training and testing data
  X = []
  Y = []
  with open(args.training_data, encoding='utf-8') as fin:
    for line in fin:
      # Generate X and Y
      x, y = sent_to_xy(line, char2idx)
      # Char2idx
      #x = [char2idx[x] if x in char2idx.keys() else char2idx['@@@PADDING@@@'] for x in line]

      # Append
      X.append(x)
      Y.append(y)

  #print("X: {}".format(X[:10]))
  #print("Y: {}".format(Y[:10]))
  #X = np.asarray(X)
  #Y = np.asarray(Y)

  train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
  

  for e in range(args.num_epochs):
    start_time = time.time()
    LOGGER.info("Epoch %d/%d" % (e, args.num_epochs))

    # Training
    train_acc, train_loss, test_acc, test_loss = 0.0, 0.0, 0.0, 0.0
  
    pbar = tqdm.tqdm(
      range(0, len(train_X), batch_size), desc="Training"
    )
    for i in pbar:
      batch_x = train_X[i]#:min(i + batch_size, len(train_X))]
      batch_y = train_Y[i]#:min(i + batch_size, len(train_X))]
      
      # TODO(naetherm): Train!
      if is_cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
      
      #print("batch_x: {}".format(batch_x))
      model.zero_grad()

      # Perform model
      tag_scores = model(batch_x)
      tag_scores = tag_scores.squeeze()
      tags = batch_y.squeeze()

      try:
        loss = loss_function(tag_scores, tags)
      except:
        print("error!")

      loss.backward()
      optimizer.step()

      acc = (np.argmax(tag_scores.detach().numpy(), -1) == np.asarray(tags)).sum().item() / len(tags)

    
      # Sanity check
      assert not np.isnan(loss.detach().numpy())
      train_loss += loss.detach().numpy()
      train_acc += acc
      pbar.set_postfix(loss=loss, accuracy=acc)

    # Validation
    pbar = tqdm.tqdm(
      range(0, len(test_X), batch_size), desc="Validation"
    )
    for i in pbar:
      batch_x = test_X[i]#:min(i + batch_size, len(train_X))]
      batch_y = test_Y[i]#:min(i + batch_size, len(train_X))]
      
      # TODO(naetherm): Train!
      if is_cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
      
      #print("batch_x: {}".format(batch_x))
      model.zero_grad()

      # Perform model
      tag_scores = model(batch_x)
      tag_scores = tag_scores.squeeze()
      tags = batch_y.squeeze()

      try:
        loss = loss_function(tag_scores, tags)
      except:
        print("error!")

      #loss.backward()
      #optimizer.step()

      acc = (np.argmax(tag_scores.detach().numpy(), -1) == np.asarray(tags)).sum().item() / len(tags)

    
      # Sanity check
      assert not np.isnan(loss.detach().numpy())
      train_loss += loss.detach().numpy()
      train_acc += acc
      pbar.set_postfix(loss=loss, accuracy=acc)

    diff_time = time.time() - start_time

    LOGGER.info("Epoch time: {}".format(diff_time))

  # Testing
  pbar = tqdm.tqdm(
    range(0, len(test_X), batch_size), desc="Validation"
  )
  for i in pbar:
    batch_x = test_X[i]#:min(i + batch_size, len(train_X))]
    batch_y = test_Y[i]#:min(i + batch_size, len(train_X))]
    
    # TODO(naetherm): Train!
    if is_cuda:
      batch_x = batch_x.cuda()
      batch_y = batch_y.cuda()
    
    #print("batch_x: {}".format(batch_x))
    model.zero_grad()

    # Perform model
    tag_scores = model(batch_x)
    tag_scores = tag_scores.squeeze()
    tags = batch_y.squeeze()

    try:
      loss = loss_function(tag_scores, tags)
    except:
      print("error!")

    #loss.backward()
    #optimizer.step()

    acc = (np.argmax(tag_scores.detach().numpy(), -1) == np.asarray(tags)).sum().item() / len(tags)

  
    # Sanity check
    assert not np.isnan(loss.detach().numpy())
    train_loss += loss.detach().numpy()
    train_acc += acc
    pbar.set_postfix(loss=loss, accuracy=acc)

def main(argv=None):

  if argv is None:
    argv = sys.argv[1:]

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--training-data', dest='training_data', type=str, required=True, 
    help='The path to the training data'
  )
  parser.add_argument(
    '--model', dest='model', type=str, default="",
    choices=['cnn', 'lstm', 'gru', 'nastase'],
    help='The model to train'
  )
  parser.add_argument(
    '--num-epochs', dest='num_epochs', type=int, default=50,
    help='The number of epochs to train for'
  )
  parser.add_argument(
    '--batch-size', dest='batch_size', type=int, default=2,
    help='The batch size to train with'
  )
  parser.add_argument(
    '--lr', dest='lr', type=float, default=0.0025,
    help='The learning rate.'
  )
  parser.add_argument(
    '--char2idx', dest='char2idx', type=str, default=None,
    help='char2idx'
  )
  parser.add_argument(
    '--idx2char', dest='idx2char', type=str, default=None,
    help='idx2char'
  )
  parser.add_argument(
    '--seq-length', dest='seq_length', type=int, default=2048,
    help='The maximum sequence length in characters.'
  )
  parser.add_argument(
    '--output-path', dest='output_path', type=str,
    help='The path everything to save.'
  )

  args = parser.parse_args()

  # Sanity check for the output-path
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  # Read and process the data
  # If args.char2idx or args.idx2char are None
  if args.char2idx is None or args.idx2char is None:
    # Generate those files
    idx2char, char2idx = lazy_scan_vocabulary(args.training_data)

    np.save(os.path.join(args.output_path, "char2idx.npy"), char2idx)
    np.save(os.path.join(args.output_path, "idx2char.npy"), idx2char)
  else:
    pass

  vocab_size = len(idx2char) + 1
  print("Vocab_Size: {}".format(vocab_size))
  num_threads = get_threads()
  TOTAL_LABELS = 0
  LABEL_COUNTS = [0., 0.]

  torch.set_num_threads(num_threads)

  if args.model == 'cnn':
    from models.cnn_model import CNNModel
    
    model = CNNModel(
      32,
      vocab_size,
      2,
      0
    )
    
  elif args.model == 'lstm':
    from models.rnn_model import LSTMModel
    """
    model = LSTMModel(
      32,
      256,
      -1,
      2
    )
    """
  elif args.model == 'gru':
    from models.rnn_model import GRUModel
    
    model = GRUModel(
      32,
      256,
      -1,
      2
    )
  elif args.model == 'nastase':
    from models.nastase import NastaseModel
    #from word_segmentation.layers.torch_nastase_layers import *
    
    #model = NastaseModel()
  else:
    LOGGER.error("Unknown model '" + args.model + "'")
    # Break
    exit(1)

  # Start training
  train(model, args, char2idx, idx2char)

  # Save trained model
  #model.save()


if __name__ == '__main__':
  main()