from __future__ import division
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter
from scipy.spatial.distance import euclidean
from random import choice, seed as rseed, uniform as rand
import pandas as pd
import numpy as np
from stats import ABCD
from misc import *
from pdb import set_trace

def SMOTE(data=None, atleast=50, atmost=100, k=5, resample=False):
  "Synthetic Minority Oversampling Technique"
  def knn(a,b):
    return sorted(b, key=lambda F: euclidean(a[:-1], F[:-1]))

  def extrapolate(one, two):
    new = len(one)*[None]
    new[:-1] = [min(a, b) + rand(0,1) * (abs(a - b)) for
                     a, b in zip(one[:-1], two[:-1])]
    new[-1] = int(one[-1])
    return new

  def populate(data):
    newData = []
    for _ in xrange(atleast):
      for one in data:
        neigh = knn(one, data)[1:k + 1]
        try:
          two = choice(neigh)
        except IndexError:
          two = one
        newData.append(extrapolate(one, two))
    return [choice(newData) for _ in xrange(atleast)]

  def depopulate(data):
    if resample:
      newer = []
      for _ in xrange(atmost):
        orig = choice(data)
        newer.append(extrapolate(orig, knn(orig, data)[1]))
      return newer
    else:
      return [choice(data).tolist() for _ in xrange(atmost)]

  newCells = []
  rseed(1)
  klass = lambda df: df[df.columns[-1]]
  count = Counter(klass(data))

  for u in count.keys():
    if count[u] <= atleast:
      newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u]))
    if count[u] >= atmost:
      newCells.extend(depopulate([r for r in data.as_matrix() if r[-1] == u]))
    else:
      newCells.extend([r.tolist() for r in data.as_matrix() if r[-1] == u])

  return pd.DataFrame(newCells, columns=data.columns)

def _smote():
  "Test SMOTE"
  dir = '../Data/Jureczko/camel/camel-1.6.csv'
  Tbl = csv2DF([dir], as_mtx=False)
  newTbl = SMOTE(Tbl)
  print('Before SMOTE: ', Counter(Tbl[Tbl.columns[-1]]))
  print('After  SMOTE: ', Counter(newTbl[newTbl.columns[-1]]))
  # ---- ::DEBUG:: -----
  set_trace()

def rforest(train, test, tunings=None, smoteit=True, bin=True, regress=False):
  "RF "
  if not isinstance(train, pd.core.frame.DataFrame):
    train = csv2DF(train, as_mtx=False, toBin=bin)

  if not isinstance(test, pd.core.frame.DataFrame):
    test_DF = csv2DF(test, as_mtx=False, toBin=True)

  if smoteit:
    train = SMOTE(train, resample=True)
  if not tunings:
    if regress:
      clf = RandomForestRegressor(n_estimators=100, random_state=1)
    else:
      clf = RandomForestClassifier(n_estimators=100, random_state=1)
  else:
    if regress:
      clf = RandomForestRegressor(n_estimators=int(tunings[0]),
                                   max_features=tunings[1] / 100,
                                   min_samples_leaf=int(tunings[2]),
                                   min_samples_split=int(tunings[3]))
    else:
      clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                   max_features=tunings[1] / 100,
                                   min_samples_leaf=int(tunings[2]),
                                   min_samples_split=int(tunings[3]))
  features = train.columns[:-1]
  klass = train[train.columns[-1]]
  clf.fit(train[features], klass)
  actual = test[test.columns[-1]].as_matrix()
  preds = clf.predict(test[test.columns[:-1]])
  return actual, preds

def _RF():
  dir = '../Data/Jureczko/'
  train, test = explore(dir)
  print('Dataset, Expt(F-Score)')
  for tr,te in zip(train, test):
    say(tr[0].split('/')[-1][:-8])
    actual, predicted = rforest(tr, te)
    abcd = ABCD(before=actual, after=predicted)
    F = np.array([k.stats()[-2] for k in abcd()])
    tC = Counter(actual)
    FreqClass=[tC[kk]/len(actual) for kk in list(set(actual))]
    ExptF = np.sum(F*FreqClass)
    say(', %0.2f\n'%(ExptF))
  # ---- ::DEBUG:: -----
  set_trace()

if __name__ == '__main__':
  _RF()
