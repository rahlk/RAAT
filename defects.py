from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore, say
from tools.stats import ABCD

class temporal:
  def __init__(self):
    pass
  def deltas(self):
    from collections import Counter
    counts = {}

    def save2plot(header, counts, labels, N):
      for h in header: say(h+' ')
      print('')
      for l in labels:
        say(l[1:]+' ')
        for k in counts.keys():
          say("%0.2f "%(counts[k][l]*100/N))
        print('')


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
      header = ['Features']+counts.keys()
      save2plot(header, counts, everything, N=len(changes))

  def improve(self):
    for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
      print("##", name)
      e=[]
      train, test = explore(dir='Data/Jureczko/', name=name)
      for planners in [xtree, method1, method2, method3]:
        aft = [planners.__doc__]
        for _ in xrange(10):
          aft.append(planners(train, test, justDeltas=False))
          # set_trace()
        e.append(aft)
      rdivDemo(e)

class cross:
  def __init__(self):
    pass
  def improve1(self):
    """
    Learn from all other projects. Compare the results.
    :return:
    """
    names=['ant', 'ivy', 'jedit', 'lucene', 'poi']
    for planners in [xtree]:#, method1, method2, method3]:
      for one in names:
        e=[]
        for two in names:
          print("##", two, one)
          aft = [two]
          train,_ = explore(dir='Data/Jureczko/', name=two)
          test,_  = explore(dir='Data/Jureczko/', name=one)
          for _ in xrange(10):
            aft.append(planners(train, test, justDeltas=False))
            # set_trace()
          e.append(aft)
        rdivDemo(e)
      set_trace()
  def deltas(self):
    from collections import Counter

    def save2plot(header, counts, labels, N):
      for h in header: say(h+' ')
      print('')
      for l in labels:
        say(l[1:]+' ')
        for k in counts.keys():
          say("%0.2f "%(counts[k][l]*100/N))
        print('')

    names=['ant', 'ivy', 'jedit', 'lucene', 'poi']
    for planners in [xtree]:#, method1, method2, method3]:
      counts = {}
      for one in names:
        e=[]
        for two in names:
          print("##", two, one)
          aft = [two]
          train,_ = explore(dir='Data/Jureczko/', name=two)
          test,_  = explore(dir='Data/Jureczko/', name=one)
          for _ in xrange(1):
            keys=[]
            everything, changes = planners(train, test, justDeltas=True)
            for ch in changes: keys.extend(ch.keys())
            counts.update({two:Counter(keys)})
          # set_trace()
        header = ['Features']+counts.keys()
        save2plot(header, counts, everything, N=len(changes))

class accuracy:
  """
  Test prediction accuracy with RF
  """

  def __init__(i):
    pass

  def main(i):
    train,test = explore(dir='Data/Jureczko/')
    for te in test:
      print("# %s"%(te[0].split('/')[-2]))
      E0, E1 = [],[]
      for tr in train:
        Pd=[tr[0].split('/')[-2]]
        Pf=[tr[0].split('/')[-2]]
        G =[tr[0].split('/')[-2]]
        for _ in xrange(1):
          actual, preds = rforest(tr, te)
          abcd = ABCD(before=actual, after=preds)
          F = np.array([k.stats()[-1] for k in abcd()])
          # set_trace()
          G.append(F[0])
          # Pd.append(F[1][0])
        E0.append(G)
        # E1.append(Pf)
      rdivDemo(E0)
      # rdivDemo(E1)
    set_trace()


class mccabe:
  """
  MacCabe Halsted dataset
  """

  def __init__(self):
    pass
  def deltas(self):
    from collections import Counter
    counts = {}

    def save2plot(header, counts, labels, N):
      for h in header: say(h+' ')
      print('')
      for l in labels:
        say(l[1:]+' ')
        for k in counts.keys():
          say("%0.2f "%(counts[k][l]*100/N))
        print('')


    for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
      print("##", name)
      e=[]
      train, test = explore(dir='Data/mccabe/', name=name)
      for planners in [xtree, method1, method2, method3]:
        # aft = [planners.__doc__]
        for _ in xrange(1):
          keys=[]
          everything, changes = planners(train, test, justDeltas=True)
          for ch in changes: keys.extend(ch.keys())
          counts.update({planners.__doc__:Counter(keys)})
      header = ['Features']+counts.keys()
      save2plot(header, counts, everything, N=len(changes))

  def improve(self):
    for name in ['cm', 'ar', 'jm', 'kc', 'mc', 'mw', 'pc']:
      print("##", name)
      train, test = explore(dir='Data/mccabe/', name=name)
      all = train+test
      for test in all:
        if not test==train:
          e=[]
          for planners in [xtree, method1, method2, method3]:
            aft = [planners.__doc__]
            for _ in xrange(10):
              aft.append(planners(train = [aa for aa in all if not aa==test], test=[test], justDeltas=False))
              # set_trace()
            e.append(aft)
          rdivDemo(e)

if __name__=='__main__':
  # accuracy().main()
  mccabe().improve()