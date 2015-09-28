"""
XTREE
"""
from __future__ import print_function, division
import pandas as pd, numpy as np
from pdb import set_trace
from tools.misc import *
from tools import pyC45
from tools.discretize import discretize
from tools.rforest import *

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

  def __init__(self):
    self.log = {}

  def save(self, name=None, old=None, new=None):
    if not old == new:
      self.log.update({name: (old, new)})

class patches:

  def __init__(i,train,test,tree=None):
    i.train=train
    i.test=test
    i.tree=tree
    i.change =[]

  def patchIt(i,testInst):
    # 1. Find where t falls
    def find(t):
      if len(t.kids)==0:
        return t
      for kid in t.kids:
        if kid.val[0]<=testInst[kid.f]<kid.val[1]:
          return find(kid)
        elif kid.val[1]==testInst[kid.f]==i.train.describe()[kid.f]['max']:
          return find(kid)
      return t

    current = find(i.tree)
    # 2. Traverse to  a better location
    # - 2.1 Go up one level
    upper = [kids for kids in current.up.kids if not kids==current]
    # - 2.2 Get all the leaf nodes of the neighbours
    leaf=[[l for l in pyC45.leaves(u)] for u in upper]
    better = [l for l in flatten(leaf) if l.score<=0.5*current.score]
    set_trace()

    # 3. Find and apply the changes
    # 4. Predict defects
    # 5. Return changes and prediction


  def main(i, justDeltas=False):

  # def newTable(justDeltas=False):
    newRows = [i.patchIt(i.test.iloc[n]) for n in xrange(i.test.shape[0]) if i.test.iloc[n][-1]>0]
    after = pd.DataFrame(newRows, columns=test.columns)
    if not justDeltas:
      return after
    else:
      return i.change

def xtree(train, test, justDeltas=False):
  # if mode == "defect":
  train_DF = csv2DF(train)
  test_DF = csv2DF(test)
  tree = pyC45.dtree(train_DF)
  return patches(train=train_DF, test=test_DF, tree=tree).main(justDeltas=justDeltas)

if __name__ == '__main__':
  for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = xtree(train, test)
    # _, pred = rforest(train,aft)
    # _,  bef = rforest(train,csv2DF(test))
    # testDF = csv2DF(test, toBin=True)
    # before = testDF[testDF.columns[-1]]
    # after = aft[aft.columns[-1]]
    # print(name,': 0.2f'%((1-sum(after)/sum(before))*100))
  # set_trace()

