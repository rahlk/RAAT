"""
XTREE
"""
from __future__ import print_function, division
import pandas as pd, numpy as np
from pdb import set_trace
from tools.misc import *
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

def patches(train, test, justDeltas=False):

  train = csv2DF(train)
  test = csv2DF(test)
  change =[]

  def patchIt(t, tbl):
    # 1. Find where t falls
    def find(t):
      for i, _, el in zip(enumerate(tree))
    # 2. Traverse to  a better
    # 3. Find and apply the changes
    # 4. Predict defects
    # 5. Return changes and prediction

  def newTable(justDeltas=False):
    newRows = [patchIt(t) for t in test.as_matrix() if t[-1]>0]
    after = pd.DataFrame(newRows, columns=test.columns)
    if not justDeltas:
      return after
    else:
      return change

def xtree(train, test, name=None, prune=False, mode='defect', justDeltas=False):
  if mode == "defect":
    train_DF = csv2DF(train)
    test_DF = csv2DF(test)
    return patches(train=train, test=test, justDeltas=justDeltas)

if __name__ == '__main__':
  for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = xtree(train, test, prune=True).main()
    # _, pred = rforest(train,aft)
    # _,  bef = rforest(train,csv2DF(test))
    testDF = csv2DF(test, toBin=True)
    before = testDF[testDF.columns[-1]]
    after = aft[aft.columns[-1]]
    print(name,': 0.2f'%((1-sum(after)/sum(before))*100))
  set_trace()

