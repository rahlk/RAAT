"""
XTREE
"""
from __future__ import print_function, division
import sys
sys.path.append('..')
from tools.sk import *
from tools.rforest import *
from tools.where import where
from random import choice
from scipy.spatial.distance import euclidean as edist
from sklearn.tree import DecisionTreeClassifier as CART

def fWeight(tbl):
    """
    Sort features based on entropy
    """
    clf = CART(criterion='entropy')
    features = tbl.columns[:-1]
    klass = tbl[tbl.columns[-1]]
    clf.fit(tbl[features], klass)
    lbs = clf.feature_importances_
    return np.argsort(lbs)[::-1]

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



class node:
  """
  A data structure to hold all the rows in a cluster.
  Also return an exemplar: the centroid.
  """

  def __init__(i, rows):
    i.rows = []
    for r in rows:
      i.rows.append(r)
    i.sample = i.exemplar()

  def exemplar(i, what='choice'):
    if what == 'choice':
      return choice(i.rows)
    if what == 'centroid':
      return median(np.array(i.rows), axis=0)
    elif what == 'mean':
      return np.mean(np.array(i.rows), axis=0)


class changes():
  """
  Record changes.
  """
  def __init__(i):
    i.log = {}

  def save(i, name=None, old=None, new=None):
    if not old == new:
      i.log.update({name: (old, new)})

class patches:

  def __init__(i,train,test,trainDF,testDF,fsel=True,clst=None):
    i.train=train
    i.trainDF = trainDF
    i.test=test
    i.testDF=testDF
    i.clstr=clst
    i.change =[]
    if fsel:
      i.fsel=fsel
      i.lbs = fWeight(trainDF)

  def closest(i, arr):
    """
    :param arr: np array (len=No. Indep var + No. Depen var)
    :return: float
    """
    return sorted(i.clstr, key= lambda x: edist(arr[:-1], x.sample[:-1]))

  def patchIt(i,testInst):
    testInst = testInst.values
    close = i.closest(testInst)[0]
    better = sorted(i.closest(close.sample), key=lambda x: x.sample[-1])[0]
    newInst = testInst + (better.sample-close.sample)
    if i.fsel:
      indx = i.lbs[:int(len(i.lbs)*0.33)]
      for n in indx:
        testInst[n] = newInst[n]
    return testInst

  def main(i, reps=10, justDeltas=False):
    newRows = [i.patchIt(i.testDF.iloc[n]) for n in xrange(i.testDF.shape[0]) if i.testDF.iloc[n][-1]>0]
    newRows = pd.DataFrame(newRows, columns=i.testDF.columns)
    before, after = rforest(i.train, newRows)
    gain = (1-sum(after)/len(after))*100
    # set_trace()
    # set_trace()
    if not justDeltas:
      return gain
    else:
      return i.change

def CD(train, test, justDeltas=False):
  "CD"
  train_DF = csv2DF(train, toBin=True)
  test_DF = csv2DF(test)
  clstr = [node(x) for x in where(data=train_DF)]
  # set_trace()
  return patches(train=train, test=test, trainDF=train_DF, testDF=test_DF, clst=clstr).main(justDeltas=justDeltas)

if __name__ == '__main__':
  E = []
  for name in ['ant']:#, 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    aft = [name]
    for _ in xrange(10):
      aft.append(CD(train, test))
    E.append(aft)
  rdivDemo(E)