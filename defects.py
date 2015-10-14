from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore

def temporal():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    e=[]
    train, test = explore(dir='Data/Jureczko/', name=name)
    for planners in [xtree, method1, method2, method3]:
      aft = [planners.__doc__]
      for _ in xrange(1):
        aft.append(planners(train, test, justDeltas=False))
        set_trace()
      e.append(aft)
    rdivDemo(e)

def cross():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    e=[]
    train, test = explore(dir='Data/Jureczko/', name=name)
    for planners in [xtree, method1, method2, method3]:
      aft = [planners.__doc__]
      for _ in xrange(1):
        aft.append(planners(train, test, justDeltas=False))
        set_trace()
      e.append(aft)
    rdivDemo(e)

def deltas():
  from collections import Counter
  counts = {}
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    e=[]
    train, test = explore(dir='Data/Jureczko/', name=name)
    for planners in [xtree, method1, method2, method3]:
      # aft = [planners.__doc__]
      for _ in xrange(1):
        keys=[]
        everything, changes = planners(train, test, justDeltas=True)
        for ch in changes: keys.extend(ch.keys())
        counts.update({planners.__doc__:Counter(keys)})
    set_trace()

if __name__=='__main__':
  deltas()