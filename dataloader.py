# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>

import os
import pickle
import time
from collections import Counter

import torch
import numpy as np

def load_textfile(filename):
  with open(filename, encoding='utf-8') as fin:
    text = [l.rstrip() for l in fin]
  return text

def space_tag(sent, nonspace=0, space=1):
  """
  :param sent: str
      Input sentence
  :param nonspace: Object
      Non-space tag. Default is 0, int type
  :param space: Object
      Space tag. Default is 1, int type
  It returns
  ----------
  chars : list of character
  tags : list of tag
  (example)
      sent  = 'Test sentence for demonstration purposes.'
      chars = list('Testsentencefordemonstrationpurposes.')
      tags  = [{0,1},...]
  """

  sent = sent.strip()
  chars = list(sent.replace(' ',''))
  tags = [nonspace]*(len(chars) - 1) + [space]
  idx = 0

  for c in sent:
    if c == ' ':
      tags[idx-1] = space
    else:
      idx += 1

  return chars, tags

def lazy_scan_vocabulary(text_file, padding_idx=0, min_count=1):
  counter = Counter()

  with open(text_file, encoding='utf-8') as fin:
    text = fin.readline().rstrip()

    c2 = Counter(vocab for vocab in text)
    
    counter.update(c2)
  
  idx2char = []
  char2idx = {}
  idx2char.append("@@@PADDING@@@")
  char2idx["@@@PADDING@@@"] = padding_idx
  idx2char.extend([vocab for vocab in sorted(counter, key=lambda x:-counter[x])])
  char2idx.update({vocab:idx+1 for idx, vocab in enumerate(idx2char)})

  return idx2char, char2idx

def scan_vocabulary(texts, padding_idx=0,min_count=1):
  """
  :param texts: list of str
      Input sentences
  :param min_count: int
      Minimum frequency of vocabulary occurrence. Default is 1
  It returns
  ----------
  idx_to_char: list of str
  char_to_idx: dict, str to int
  """

  counter = Counter(vocab for text in texts for vocab in text)
  counter = {vocab:count for vocab, count in counter.items() if count >=min_count}
  idx2char = []
  char2idx = {}
  idx2char.append("@@@PADDING@@@")
  char2idx["@@@PADDING@@@"] = padding_idx
  idx_to_char = [vocab for vocab in sorted(counter, key=lambda x:-counter[x])]
  char_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_char)}
  return idx_to_char, char_to_idx

def to_idx(item, mapper, unknown=None):
  """
  :param item: Object
      Object to be encoded
  :param mapper: dict
      Dictionary from item to idx
  :param unknown: int
      Index of unknown item. If None, use len(mapper)
  It returns
  ----------
  idx : int
      Index of item
  """

  if unknown is None:
    unknown = len(mapper)
  return mapper.get(item, unknown)

def to_item(idx, idx_to_char, unknown='Unk'):
  """
  :param idx: int
      Index of item
  :param idx_to_char: list of Object
      Mapper from index to item object
  :param unknown: Object
      Return value when the idx is outbound of idx_to_char
      Default is 'Unk', str type
  It returns
  ----------
  object : object
      Item that corresponding idx
  """

  if 0 <= idx < len(idx_to_char):
    return idx_to_char[idx]
  return unknown

def sent_to_xy(sent, char_to_idx):
  """
  :param sent: str
      Input sentence
  :param char_to_idx: dict
      Dictionary from character to index
  It returns
  ----------
  idxs : torch.LongTensor
      Encoded character sequence
  tags : torch.LongTensor
      Space tag sequence
  """

  chars, tags = space_tag(sent)
  idxs = torch.LongTensor(
    [[to_idx(c, char_to_idx) for c in chars]])
  #  [to_idx(c, char_to_idx) for c in chars])
  tags = torch.LongTensor([tags])
  return idxs, tags

def sent_to_xy_ensemble(sent, char_to_idx):
  """
  :param sent: str
      Input sentence
  :param char_to_idx: dict
      Dictionary from character to index
  It returns
  ----------
  idxs : torch.LongTensor
      Encoded character sequence
  tags : torch.LongTensor
      Space tag sequence
  """

  chars, tags = space_tag(sent)
  rnn_idxs = torch.LongTensor(
    [to_idx(c, char_to_idx) for c in chars])
  rnn_tags = torch.LongTensor(tags)
  cnn_idxs = torch.LongTensor(
    [[to_idx(c, char_to_idx) for c in chars]])
  cnn_tags = torch.LongTensor([tags])
  return rnn_idxs, rnn_tags, cnn_idxs, cnn_tags


def correct(sent, char_to_idx, model):
  """
  :param sent: str
      Input sentence
  :param char_to_idx: dict
      Mapper from character to index
  :param model: torch.nn.Module
      Trained space correction model
  It returns
  ----------
  sent_ : str
      Space corrected sentence
  """

  x, y = sent_to_xy(sent, char_to_idx)
  tags = torch.argmax(model(x), dim=1).numpy()
  chars = sent.replace(' ','')
  sent_ = ''.join([c if t == 0 else (c + ' ') for c, t in zip(chars, tags)]).strip()
  return sent_

def correct_cnn(sent, char_to_idx, model):
  """
  :param sent: str
      Input sentence
  :param char_to_idx: dict
      Mapper from character to index
  :param model: torch.nn.Module
      Trained space correction model
  It returns
  ----------
  sent_ : str
      Space corrected sentence
  """

  _, _, x, y = sent_to_xy_ensemble(sent, char_to_idx)
  tags = torch.argmax(model(x).squeeze(), dim=-1).numpy()
  chars = sent.replace(' ','')
  sent_ = ''.join([c if t == 0 else (c + ' ') for c, t in zip(chars, tags)]).strip()
  return sent_