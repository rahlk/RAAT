from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore

def main():
  e=[]
  for name in ['Apache', 'BDBC', 'BDBJ', 'LLVM', 'X264', 'SQL']:
    print("##", name)
    data = explore(dir='Data/Seigmund/', name=name)
    # set_trace()
    for planners in [xtree, method3]:#, method2, method3]:
      # aft = [planners.__doc__]
      aft = [name]
      for _ in xrange(10):
        aft.append(planners(train=data, test=None, config=True))
      e.append(aft)
  rdivDemo(e, isLatex=True)
    # set_trace()

if __name__=='__main__':
  main()