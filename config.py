from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore

def main():
  for name in ['Apache', 'BDBC', 'BDBJ', 'LLVM', 'X264', 'SQL']:
    print("##", name)
    e=[]
    data = explore(dir='Data/Seigmund/', name=name)
    set_trace()
    for planners in [xtree, method1, method2, method3]:
      aft = [planners.__doc__]
      for _ in xrange(32):
        aft.append(planners(train=data, test=None, config=True))
      e.append(aft)
    rdivDemo(e)

if __name__=='__main__':
  main()