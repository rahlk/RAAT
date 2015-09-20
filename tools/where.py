from __future__ import division, print_function
from pdb import set_trace
import pandas as pd
import numpy as np
from os import walk
from random import randint as randi, seed as rseed
__author__ = 'rkrsn'

def getDataFrame(dir='Data/Jureczko/ant/'):
  files=[]
  for (dirpath, _, filename) in walk(dir):
    for f in filename:
      df=pd.read_csv(dirpath+f)
      headers = [h for h in df.columns if '?' not in h]
      files.append(df[headers])

  "For N files in a project, use 1 to N-1 as train."
  train = pd.concat(files[:-1])
  #
  # "For N files in a project, N as test."
  # test = files[-1]
  return train.as_matrix()


def where():
  """
  Recursive FASTMAP clustering.
  """
  rseed(0)
  dataset = getDataFrame()
  N = np.shape(dataset)[0]
  clusters = []

  def aDist(one, two):
    return np.sqrt(np.sum((np.array(one[:-1])-np.array(two[:-1]))**2))

  def farthest(one,rest):
    return sorted(rest, key=lambda F: aDist(F,one))[-1]

  def recurse(dataset):
    R, C = np.shape(dataset) # No. of Rows and Col
    # Find the two most distance points.
    one=dataset[randi(0,R-1)]
    mid=farthest(one, dataset)
    two=farthest(mid, dataset)

    # Project each case on
    def proj(test):
      a = aDist(one, test)
      b = aDist(two, test)
      c = aDist(one, two)
      return (a**2-b**2+c**2)/(2*c)

    if R<np.sqrt(N):
      clusters.append(dataset)
    else:
      _ = recurse(sorted(dataset,key=lambda F:proj(F))[:int(R/2)])
      _ = recurse(sorted(dataset,key=lambda F:proj(F))[int(R/2):])

  recurse(dataset)
  return clusters


if __name__=='__main__':
  clusters = where()
  # ----- ::DEBUG:: -----
  set_trace()
