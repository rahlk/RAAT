"""
XTREE
"""
from __future__ import print_function, division
import pandas as pd, numpy as np
from pdb import set_trace

from tools.sk import *
from tools.misc import *
from tools.rforest import *
import tools.pyC45 as pyC45
from tools.Discretize import discretize
from tools.where import where
from timeit import time
from random import uniform
from numpy.random import normal as randn

def flatten(x):
  """
  Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
  and returns a single list [a,b,c,d,e,f]
  """
  result = []
  for el in x:
    if hasattr(el, "__iter__") and not isinstance(el, basestring):
      result.extend(flatten(el))
    else:
      result.append(el)
  return result


class changes():
  """
  Record changes.
  """
  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if not old == new:
      self.log.update({name: (old, new)})

class patches:

  def __init__(i,train,test,trainDF,testDF,clst=None):
    i.train=train
    i.trainDF = trainDF
    i.test=test
    i.testDF=testDF
    i.clstr=clst
    i.change =[]

  def closest(i, arr):
    return sorted([])
  def patchIt(i,testInst):
    set_trace()
    pass

  def main(i, reps=10, justDeltas=False):
    newRows = [i.patchIt(i.testDF.iloc[n]) for n in xrange(i.testDF.shape[0]) if i.testDF.iloc[n][-1]>0]
    newRows = pd.DataFrame(newRows, columns=i.testDF.columns)
    before, after = rforest(i.train, newRows)
    # set_trace()
    gain = (1-sum(after)/len(after))*100
    if not justDeltas:
      return gain
    else:
      return i.change

def CD(train, test, justDeltas=False):
  train_DF = csv2DF(train)
  test_DF = csv2DF(test)
  clstr = where(data=train_DF)
  # set_trace()
  return patches(train=train, test=test, trainDF=train_DF, testDF=test_DF, clst=clstr).main(justDeltas=justDeltas)

if __name__ == '__main__':
  E = []
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = [name]
    for _ in xrange(40):
      aft.append(CD(train, test))
    E.append(aft)
  rdivDemo(E)

