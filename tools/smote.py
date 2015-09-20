#! /Users/rkrsn/anaconda/bin/python
from __future__ import print_function, division
from pdb import set_trace
from collections import Counter
from scipy.spatial.distance import euclidean
from random import choice, seed as rseed, uniform as rand
import pandas as pd
from misc import *

def SMOTE(data=None, atleast=50, atmost=100, k=5, resample=False):

  def knn(a,b):
    return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

  def extrapolate(one, two):
    new = len(one)*[None]
    new[:-1] = [min(a, b) + rand(0,1) * (abs(a - b)) for
                     a, b in zip(one[:-1], two[:-1])]
    new[-1] = int(one[-1])
    return new

  def populate(data):
    newData = []
    for _ in xrange(atleast):
      for one in data:
        neigh = knn(one, data)[1:k + 1]
        try:
          two = choice(neigh)
        except IndexError:
          two = one
        newData.append(extrapolate(one, two))
    return [choice(newData) for _ in xrange(atleast)]

  def depopulate(data):
    if resample:
      newer = []
      for _ in xrange(atmost):
        orig = choice(data)
        newer.append(extrapolate(orig, knn(orig, data)[1]))
      return newer
    else:
      return [choice(data).tolist() for _ in xrange(atmost)]

  newCells = []
  rseed(1)
  klass = lambda df: df[df.columns[-1]]
  count = Counter(klass(data))

  for u in count.keys():
    if count[u] <= atleast:
      newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u]))
    if count[u] >= atmost:
      newCells.extend(depopulate([r for r in data.as_matrix() if r[-1] == u]))
    else:
      newCells.extend([r.tolist() for r in data.as_matrix() if r[-1] == u])

  return pd.DataFrame(newCells, columns=data.columns)

def test_smote():
  dir = '../Data/Jureczko/camel/camel-1.6.csv'
  Tbl = csv2DF([dir], as_mtx=False)
  newTbl = SMOTE(Tbl)
  # ---- ::DEBUG:: -----
  set_trace()

if __name__ == '__main__':
  test_smote()
