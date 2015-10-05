from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore

def main():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    e=[]
    train, test = explore(dir='Data/Jureczko/', name=name)
    for planners in [xtree, method1, method2, method3]:
      aft = [planners.__doc__]
      for _ in xrange(10):
        aft.append(planners(train, test))
      e.append(aft)
    rdivDemo(e)

if __name__=='__main__':
  main()