#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division
from numpy import array, asarray, mean, median, percentile, size, sum, sqrt
from pdb import set_trace
from tools.misc import *
from tools.rforest import *
from tools.where import where
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as CART
from scipy.spatial.distance import euclidean as edist

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

class node():

  """
  A data structure to hold all the rows in a cluster.
  Also return an exemplar: the centroid.
  """

  def __init__(self, rows):
    self.rows = []
    for r in rows:
      self.rows.append(r[:-1])

  def exemplar(self, what='centroid'):
    if what == 'centroid':
      return median(array(self.rows), axis=0)
    elif what == 'mean':
      return mean(array(self.rows), axis=0)


class contrast():

  "Identify the nearest enviable node."

  def __init__(self, clusters):
    self.clusters = clusters

  def closest(self, testCase):
    return sorted([f for f in self.clusters],
                  key=lambda F: edist(F.exemplar(), testCase[:-1]))[0]

  def envy(self, testCase, alpha=0.5):
    me = self.closest(testCase)
    others = [o for o in self.clusters if not me == o]
    betters = [f for f in others if f.exemplar()[-1] <= alpha*me.exemplar()[-1]]
    try:
      return sorted([f for f in betters],
                    key=lambda F: edist(F.exemplar(), me.exemplar()))[0]
    except:
      return me


class patches():

  "Apply new patch."

  def __init__(
          self, train, test, clusters, prune=False, B=0.25
          , verbose=False, config=False, models=False, pred=[], name=None):

    if config or models:
      self.train = csv2DF(train)
      self.test = csv2DF(test)
    else:
      self.train = csv2DF(train, toBin=False)
      self.test = csv2DF(test, toBin=False)

    self.name = name
    self.clusters = clusters
    self.Prune = prune
    self.B = B
    self.mask = self.fWeight()
    self.write = verbose
    self.bin = config
    self.pred = pred
    self.change = []

  def min_max(self):
    allRows = pd.concat([self.train, self.test]).as_matrix()
    return np.max(allRows, axis=0)[:-1]-np.min(allRows, axis=0)[:-1]

  def fWeight(self, criterion='Variance'):
    "Sort "
    clf = CART(criterion='entropy')
    features = self.train.columns[:-1]
    klass = self.train[self.train.columns[-1]]
    clf.fit(self.train[features], klass)
    lbs = clf.feature_importances_
    if self.Prune:
      cutoff = sorted(lbs, reverse=True)[int(len(lbs)*self.B)]
      return np.array([cc if cc>=cutoff else 0 for cc in lbs])
    else:
      return lbs

  def delta0(self, node1, node2):
    if not self.bin:
      return array([el1 - el2 for el1, el2 in zip(node1.exemplar()
                                                  , node2.exemplar())]) / self.min_max() * self.mask

    else:
      return array([el1 == el2 for el1, el2 in zip(node1.exemplar()
                                                   , node2.exemplar())])

  def delta(self, t):
    C = contrast(self.clusters)
    closest = C.closest(t)
    better = C.envy(t, alpha=0.5)
    return self.delta0(closest, better)

  def patchIt(self, t):
    C = changes()
    if not self.bin:
      for i, old, delt in zip(range(len(t[:-1])), t[:-1], self.delta(t)):
        C.save(self.train.columns[i][1:], old, new=old + delt)
      self.change.append(C.log)
      return (array(t[:-1]) + self.delta(t)).tolist()+[None]
    else:
      for i, old, delt, m in zip(range(len(t.cells[:-2])), t.cells[:-2], self.delta(t), self.mask.tolist()):
        C.save(
            self.train.headers[i].name[
                1:],
            old,
            new=(
                1 -
                old if delt and m > 0 else old))
      self.change.append(C.log)
      return [1 - val if d and m > 0 else val for val, m,
              d in zip(t.cells[:-2], self.mask, self.delta(t))]

  def newTable(self, justDeltas=False):
    if not self.bin:
      oldRows = [r for r in self.test.as_matrix() if abs(r[-1]) > 0]
    else:
      oldRows = self.test
    newRows = [self.patchIt(t) for t in oldRows]

    after = pd.DataFrame(newRows, columns=self.test.columns)
    before = pd.DataFrame(oldRows, columns=self.test.columns)
    if not justDeltas:
      return after
    else:
      return self.change

class strawman():

  def __init__(self, train, test, name=None, prune=False):
    self.train, self.test = train, test
    self.prune = prune
    self.name = name

  def main(self, mode='defect', justDeltas=False):
    if mode == "defect":
      train_DF = csv2DF(self.train)
      test_DF = csv2DF(self.train)
      before = rforest(train=train_DF, test=test_DF)
      clstr = [node(c) for c in where(data=train_DF)]
      return patches(train=self.train,
                     test=self.test,
                     clusters=clstr,
                     prune=self.prune,
                     pred=before).newTable(justDeltas=justDeltas)

    elif mode == "models":
      train_DF = createTbl(self.train, isBin=False)
      test_DF = createTbl(self.test, isBin=False)
      before = rforest(train=train_DF, test=test_DF)
      clstr = [c for c in self.nodes(train_DF._rows)]
      return patches(train=self.train,
                     test=self.test,
                     clusters=clstr,
                     prune=self.prune,
                     models=True,
                     pred=before).newTable(justDeltas=justDeltas)
    elif mode == "config":
      train_DF = createTbl(self.train, isBin=False)
      test_DF = createTbl(self.test, isBin=False)
      before = rforest2(train=train_DF, test=test_DF)
      clstr = [c for c in self.nodes(train_DF._rows)]
      return patches(train=self.train,
                     test=self.test,
                     clusters=clstr,
                     name=self.name,
                     prune=self.prune,
                     pred=before,
                     config=True).newTable(justDeltas=justDeltas)


def categorize(dataName):
  dir = '../Data/Jureczko'
  projects = [Name for _, Name, __ in walk(dir)][0]
  numData = len(projects)  # Number of data
  one, two = explore(dir)
  data = [one[i] + two[i] for i in xrange(len(one))]

  def withinClass(data):
    N = len(data)
    return [(data[:n], [data[n]]) for n in range(1, N)]

  def whereis():
    for indx, name in enumerate(projects):
      if name == dataName:
        return indx

  try:
    return [
        dat[0] for dat in withinClass(data[whereis()])], [
        dat[1] for dat in withinClass(data[whereis()])]  # Train, Test
  except:
    set_trace()

if __name__ == '__main__':
  for name in ['ivy', 'jedit', 'lucene', 'poi', 'ant']:
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = strawman(train, test, prune=False).main()
    _, pred = rforest(train,aft)
    testDF = csv2DF(test, toBin=True)
    before = testDF[testDF.columns[-1]]
    print(name,': ', (1-sum(pred)/sum(before))*100)
  set_trace()

