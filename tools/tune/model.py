from __future__ import division, print_function
import os, sys, subprocess
from time import sleep
# Get the git root directory
root=repo_dir = subprocess.Popen(['git'
                                      ,'rev-parse'
                                      , '--show-toplevel']
                                      , stdout=subprocess.PIPE
                                    ).communicate()[0].rstrip()
sys.path.append(root)

import numpy as np
from random import uniform
from tools.oracle import rforest
from tools.misc import *
from tools.stats import ABCD

class rf:
  """
  Random Forest
  """
  def __init__(i, data, obj=2,n_dec=5):
    i.n_dec = 5
    i.train = csv2DF(data[:-1], toBin=True)
    i.test = csv2DF([data[-1]], toBin=True)
    i.n_obj = obj # 2=precision
    i.dec_lim = [(10, 100)  # n_estimators
                , (1, 100)  # max_features
                , (1, 10)   # min_samples_leaf
                , (2, 10)   # min_samples_split
                , (2, 50)]  # max_leaf_nodes

  def generate(i,n):
    return [[uniform(i.dec_lim[indx][0]
                     , i.dec_lim[indx][1]) for indx in xrange(i.n_dec)
             ] for _ in xrange(n)]

  def solve(i,dec):
    actual, predicted = rforest(i.train, i.test, tunings=dec, smoteit=False)
    abcd = ABCD(before=actual, after=predicted)
    qual = np.array([k.stats()[i.n_obj] for k in abcd()])
    # set_trace()
    return qual[1]

if __name__=='__main__':
  problem = DTLZ2(30,3)
  row = problem.generate(1)
  print(problem.solve(row[0]))
