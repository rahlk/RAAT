"""
XTREE
"""
from __future__ import print_function, division
# import pandas as pd, numpy as np
# from pdb import set_trace

import sys
sys.path.append('..')
from tools.oracle import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, f_regression
from random import uniform
from xtree import  xtree
from tools.sk import rdivDemo

# from tools.sk import *
# from tools.misc import *
# import tools.pyC45 as pyC45
# from tools.Discretize import discretize
# from timeit import time
# from numpy.random import normal as randn
# from tools.tune.dEvol import tuner

def VARL(coef,inter,p0=0.05):
  """
  :param coef: Slope of   (Y=aX+b)
  :param inter: Intercept (Y=aX+b)
  :param p0: Confidence Interval. Default p=0.05 (95%)
  :return: VARL threshold

            1   /     /  p0   \             \
  VARL = ----- | log | ------ | - intercept |
         slope \     \ 1 - p0 /             /

  """
  return (np.log(p0/(1-p0))-inter)/coef

def apply(changes, row):
  all = []
  for idx, thres in enumerate(changes):
    newRow = row
    if thres>0:
      if newRow[idx]>thres:
        newRow[idx] = uniform(0, thres)

      all.append(newRow)

  return all

def planner(train, test, rftrain=None, tunings=None, verbose=False):

  "Threshold based planning"

  """ Compute Thresholds
  """
  data_DF=csv2DF(train, toBin=True)
  metrics=[str[1:] for str in data_DF[data_DF.columns[:-1]]]
  ubr = LogisticRegression() # Init LogisticRegressor
  X = data_DF[data_DF.columns[:-1]].values # Independent Features (CK-Metrics)
  y = data_DF[data_DF.columns[-1]].values  # Dependent Feature (Bugs)
  ubr.fit(X,y)                # Fit Logit curve
  inter = ubr.intercept_[0]   # Intercepts
  coef  = ubr.coef_[0]        # Slopes
  pVal  = f_classif(X,y)[1]   # P-Values
  changes = len(metrics)*[-1]

  if verbose:
    "Pretty Print Thresholds"
    table = Texttable()
    table.set_cols_align(["l","l","l"])
    table.set_cols_valign(["m","m","m"])
    table.set_cols_dtype(['t', 't', 't'])
    table_rows=[["Metric", "Threshold", "P-Value"]]

  "Find Thresholds using VARL"
  for Coeff, P_Val, idx in zip(coef, pVal, range(len(metrics))): #xrange(len(metrics)):
    thresh = VARL(Coeff, inter, p0=0.065) # VARL p0=0.05 (95% CI)
    if thresh>0 and P_Val<0.05:
      if verbose: table_rows.append([metrics[idx], "%0.2f"%thresh, "%0.3f"%P_Val])
      changes[idx]=thresh


  if verbose:
    table.add_rows(table_rows)
    print(table.draw(), "\n")

  """ Apply Plans Sequentially
  """
  nChange = len(table_rows)-1
  testDF = csv2DF(test, toBin=True)
  buggy = [testDF.iloc[n].values.tolist() for n in xrange(testDF.shape[0]) if testDF.iloc[n][-1]>0]
  before = len(buggy)
  new =[]
  for n in xrange(nChange):
    new.append(["Reduce "+table_rows[n+1][0]])
    for _ in xrange(5):
      modified=[]
      for attr in buggy:
        modified.append(apply(changes, attr)[n])

      modified=pd.DataFrame(modified, columns = data_DF.columns)
      before, after = rforest(train, modified, tunings=None, bin = True, regress=False)
      gain = (1 - sum(after)/sum(before))*100
      new[n].append(gain)

  return new

def __test():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    E = planner(train, test, verbose=True)

    E.append(['RANK'])
    E[-1].extend([xtree(train, test, justDeltas=False) for _ in xrange(5)])

    rdivDemo(E, isLatex=False, globalMinMax=True, high=100, low=0)
    print("\n")
    # set_trace()

if __name__=="__main__":
  from logo import logo
  __test()









