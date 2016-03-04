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

# from tools.sk import *
# from tools.misc import *
# import tools.pyC45 as pyC45
# from tools.Discretize import discretize
# from timeit import time
# from random import uniform
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

def planner(train, test, rftrain=None, tunings=None):

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

  "Find Thresholds using VARL"
  for Coeff, P_Val, idx in zip(coef, pVal, range(len(metrics))): #xrange(len(metrics)):
    thresh = VARL(Coeff, inter, p0=0.05) # VARL p0=0.05 (95% CI)
    if thresh>0 and P_Val<0.05:
      changes[idx]=thresh

  """ Apply Plans Sequentially
  """
  test = csv2DF(test, toBin=False)
  set_trace()
  buggy = [test]

def __test():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    planner(train, test)

if __name__=="__main__":
 __test()









