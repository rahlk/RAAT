from __future__ import print_function, division
__author__ = 'rkrsn'
from Planners.CD import *
from Planners.xtree import xtree
from tools.sk import rdivDemo
from tools.misc import explore, say
from tools.stats import ABCD
from tools.tune.dEvol import tuner
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
      for planners in [xtree]:#, method1, method2, method3]:
        # aft = [planners.__doc__]
        for _ in xrange(1):
          keys=[]
          everything, changes = planners(train, test, justDeltas=True)
          for ch in changes: keys.extend(ch.keys())
          counts.update({planners.__doc__:Counter(keys)})
      header = ['Features']+counts.keys()
      save2plot(header, counts, everything, N=len(changes))

  def improve(self):
    for name in ['lucene', 'poi', 'ant', 'ivy', 'jedit']:
      print("##", name)
      e=[]
      train, test = explore(dir='Data/Jureczko/', name=name)
      for planners in [xtree]:#, method1, method2, method3]:
        aft = [planners.__doc__]
        for _ in xrange(1):
          aft.append(planners(train, test, justDeltas=False))
          # set_trace()
        e.append(aft)
      rdivDemo(e, isLatex=True, globalMinMax=True, high=100, low=0)

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
          train0,train1 = explore(dir='Data/Jureczko/', name=two)
          rfTrain, test  = explore(dir='Data/Jureczko/', name=one)
          train = list(set([t for t in train0+train1 if not t in test]))
          # set_trace()
          for _ in xrange(10):
            _, new = planners(train, test, rftrain = rfTrain, justDeltas=False)
            aft.append(new)
            # set_trace()
          e.append(aft)
        rdivDemo(e)

  # def accuracy(self):
  #   """
  #   Learn from all other projects. Compare the results.
  #   :return:
  #   """
  #   names=['ant', 'ivy', 'jedit', 'lucene', 'poi']
  #   for planners in [xtree]:#, method1, method2, method3]:
  #     for one in names:
  #       e=[]
  #       for two in names:
  #         print("##", one)
  #         aft = [two]
  #         train = csv2DF(explore(dir='Data/Jureczko/', name=two)[0])
  #         test  = csv2DF(explore(dir='Data/Jureczko/', name=one)[0])
  #         for _ in xrange(10):
  #           actual, predicted = rforest(train, test, bin=True, regress=False)
  #
  #           _, new = planners(train, test, justDeltas=False)
  #           aft.append(new)
  #           # set_trace()
  #         e.append(aft)
  #       rdivDemo(e)

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
          # print("##", two, one)
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
    # set_trace()
    print("Train - Test, Pd, Pf")
    # print("# %s"%(te[0].split('/')[-2]))
    for _,te in zip(train,test):
      E0, E1 = [],[]
      for tr in train:
        Pd=[tr[0].split('/')[-2]]
        Pf=[tr[0].split('/')[-2]]
        G =[tr[0].split('/')[-2]]
        # for _ in xrange(1):
        tunings = tuner(tr)
        actual, preds = rforest(tr, te, tunings=tunings)
        abcd = ABCD(before=actual, after=preds)
        F = np.array([k.stats()[-2] for k in abcd()])
        Pd0 = np.array([k.stats()[0] for k in abcd()])
        Pf0 = np.array([k.stats()[1] for k in abcd()])
        # set_trace()
        G.append(F[0])
        say(tr[0].split('/')[-2]+' - '+te[0].split('/')[-2]+' %0.2f, %0.2f\n'%(Pd0[1], 1-Pf0[1]))
        Pd.append(Pd0)
        Pf.append(Pf0)
        # E0.append(G)
        # E1.append(Pf)
        # rdivDemo(E0)
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
    for name in ['jm', 'cm', 'ar', 'kc', 'jm', 'mw']:
      print("##", name)
      train, test = explore(dir='Data/mccabe/', name=name)
      all = train+test
      for test in all:
        if not test==train:
          e=[]
          for planners in [xtree, method1, method2, method3]:
            aft = [planners.__doc__]
            for _ in xrange(1):
              aft.append(planners(train = [aa for aa in all if not aa==test], test=[test], justDeltas=False))
              # set_trace()
            e.append(aft)
          rdivDemo(e)
  def acc(self):
    for name in ['MC']:#, 'kc', 'cm', 'ar', 'jm', 'mw']:
      print("##", name)
      train, test = explore(dir='Data/mccabe/', name=name)
      # set_trace()
      all = train+test
      E0=[]
      for test in all:
        if not test==train:
          # set_trace()
          G =[test.split('/')[-1][:-4]]
          actual, preds = rforest(train=[aa for aa in all if not aa==test], test=[test])
          abcd = ABCD(before=actual, after=preds)
          F = np.array([k.stats()[-1] for k in abcd()])
          # set_trace()
          G.append(F[0])
        E0.append(G)
      rdivDemo(E0)

if __name__=='__main__':
  accuracy().main()
  # cross().improve1()
  # mccabe().improve()
  # mccabe().acc()
  # temporal().improve()
  # cross().deltas()
