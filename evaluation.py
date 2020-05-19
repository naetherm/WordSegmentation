# -*- coding: utf-8 -*-
'''
Copyright 2019-2020, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import os
import sys
import math
import codecs
import argparse

def line_split(line):
  return [char for char in line]

def evaluate(args):

  grt_filename = args.grt_file
  prd_filename = args.prd_file
  beta = args.beta

  with codecs.open(grt_filename, 'r', encoding='utf-8') as fin:
    grt_content = [line.rstrip() for line in fin.readlines()]
  with codecs.open(prd_filename, 'r', encoding='utf-8') as fin:
    prd_content = [line.rstrip() for line in fin.readlines()]

  num_sentences = len(grt_content)

  prediction = 0.0
  recall = 0.0
  f_score = 0.0

  seq_total = 0.0
  seq_correct = 0.0
  seg_total = 0.0
  seg_correct = 0.0

  tp_count = 0.0
  tn_count = 0.0
  fn_count = 0.0
  fp_count = 0.0

  precision = 0.0
  recall = 0.0
  f_score = 0.0

  for g, p in zip(grt_content, prd_content):
    # TODO(naetherm): Calculate prediction and recall
    seq_total += 1.0
    prd_tokens = line_split(p)
    grt_tokens = line_split(g)
    is_correct = True
    for pt, gt in zip(prd_tokens, grt_tokens):
      seg_total += 1.0
      pt = int(pt)
      gt = int(gt)
      if pt != gt:
        is_correct = False
      else:
        seg_correct += 1.0
      # TODO(naetherm): fn, tn, tp, fp

      if gt == 1 and pt == 1:
        tp_count += 1.0
      if gt == 1 and pt == 0:
        fn_count += 1.0
      if gt == 0 and pt == 0:
        tn_count += 1.0
      if gt == 0 and pt == 1:
        fp_count += 1.0


    if is_correct:
      seq_correct += 1.0

  seq_acc = seq_correct / seq_total
  seg_acc = seg_correct / seg_total

  precision = tp_count / (tp_count+fp_count)
  recall = tp_count / (tp_count+fn_count)

  # Calculate f-score
  f_score = (1.0 + beta*beta)*((precision*recall)/(beta*beta*precision+recall))

  print("Precision\t\tRecall\t\t\tF-Score\n")
  print(f"{precision}\t{recall}\t{f_score}\n")


def main(argv=None):

  if argv is None:
    argv = sys.argv[1:]

  parser = argparse.ArgumentParser()
  parser.add_argument('--grt-file', dest='grt_file', type=str)
  parser.add_argument('--prd-file', dest='prd_file', type=str)
  parser.add_argument('--beta', dest='beta', type=float, default=0.5)

  args = parser.parse_args(argv)

  evaluate(args)


if __name__ == '__main__':
  main()
