from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import CD
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore

def main():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    e=[]
    train, test = explore(dir='Data/Jureczko/', name=name)
    for planners in [CD, xtree]:
      aft = [planners.__doc__]
      for _ in xrange(50):
        aft.append(planners(train, test))
      e.append(aft)
    rdivDemo(e)

if __name__=='__main__':
  E = []
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='Data/Jureczko/', name=name)
    aft = [name]
    for _ in xrange(40):
      aft.append(xtree(train, test))
    E.append(aft)
  rdivDemo(E)