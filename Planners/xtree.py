"""
XTREE
"""
from __future__ import print_function, division
import pandas as pd, numpy as np
from pdb import set_trace
import sys
sys.path.append('..')
from tools.sk import *
from tools.misc import *
from tools.rforest import *
import tools.pyC45 as pyC45
from tools.Discretize import discretize
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

  def __init__(i,train,test,trainDF,testDF,tree=None):
    i.train=train
    i.trainDF = trainDF
    i.test=test
    i.testDF=testDF
    i.tree=tree
    i.change =[]

  def leaves(i, node):
    """
    Returns all terminal nodes.
    """
    L = []
    if len(node.kids) > 1:
      for l in node.kids:
        L.extend(i.leaves(l))
      return L
    elif len(node.kids) == 1:
      return [node.kids]
    else:
      return [node]

  def find(i, testInst, t):
    if len(t.kids)==0:
      return t
    for kid in t.kids:
      try:
        if kid.val[0]<=testInst[kid.f].values[0]<kid.val[1]:
          return i.find(testInst,kid)
        elif kid.val[1]==testInst[kid.f].values[0]==i.trainDF.describe()[kid.f]['max']:
          return i.find(testInst,kid)
      except: set_trace()
    return t

  @staticmethod
  def howfar(me, other):
    common = [a for a in me.branch if a not in other.branch]
    return len(me.branch)-len(common)

  def patchIt(i,testInst):
    # 1. Find where t falls
    C = changes() # Record changes
    testInst = pd.DataFrame(testInst).transpose()
    current = i.find(testInst, i.tree)
    node = current

    while node.lvl > -1:
      node = node.up  # Move to tree root

    leaves = flatten([i.leaves(_k) for _k in node.kids])
    try:
      best = sorted([l for l in leaves if l.score<=0.01*current.score], key=lambda F: i.howfar(current,F))[0]
    except:
      return testInst.values.tolist()[0]

    def new(old, range):
      rad = abs(min(range[1]-old, old-range[1]))
      # return randn(old, rad) if rad else old
      # return uniform(old-rad,rad+old)
      return uniform(range[0],range[1])

    for ii in best.branch:
      before = testInst[ii[0]]
      if not ii in current.branch:
        then = testInst[ii[0]].values[0]
        now = new(testInst[ii[0]].values[0], ii[1])
        testInst[ii[0]] = now
        C.save(name=ii[0], old=then, new=now)

    testInst[testInst.columns[-1]] = None
    i.change.append(C.log)
    return testInst.values.tolist()[0]


  def main(i, reps=10, justDeltas=False):
    newRows = [i.patchIt(i.testDF.iloc[n]) for n in xrange(i.testDF.shape[0]) if i.testDF.iloc[n][-1]>0]
    newRows = pd.DataFrame(newRows, columns=i.testDF.columns)
    before, after = rforest(i.train, newRows)
    gain = (1-sum(after)/len(after))*100
    if not justDeltas:
      return gain
    else:
      return i.testDF.columns[:-1], i.change

def xtree(train, test, justDeltas=False, config=False):
  "XTREE"
  if config:
    data = csv2DF(train, toBin=False)
    shuffle(data)
    train_DF, test_DF=data[:int(len(data)/2)], data[int(len(data)/2):].reset_index(drop=True)
    set_trace()
  else:
    train_DF = csv2DF(train, toBin=True)
    test_DF = csv2DF(test)
    tree = pyC45.dtree(train_DF)
    # set_trace()
    return patches(train=train, test=test, trainDF=train_DF, testDF=test_DF, tree=tree).main(justDeltas=justDeltas)

if __name__ == '__main__':
  E = []
  for name in ['ant']:#, 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = [name]
    for _ in xrange(10):
      aft.append(xtree(train, test))
    E.append(aft)
  rdivDemo(E)