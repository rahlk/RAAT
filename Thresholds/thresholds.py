"""
CK Thresholds
"""
from __future__ import print_function, division
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, f_regression
import pandas as pd, numpy as np
from pdb import set_trace
import sys
sys.path.append('..')
from tools.sk import *
from tools.misc import *
from tools.oracle import *
import tools.pyC45 as pyC45
from tools.Discretize import discretize
from timeit import time
from random import uniform
from numpy.random import normal as randn
from tools.tune.dEvol import tuner


def VARL(coef,inter,p0=0.05):
  return (np.log(p0/(1-p0))-inter)/coef

# def VARG(coef,inter,p0=0.05):
#

def thresholds():
  for name in ['ant', 'ivy', 'jedit', 'lucene', 'poi']:
    print("##", name)
    train, test = explore(dir='../Data/Jureczko/', name=name)
    data_DF=csv2DF(train, toBin=True)
    metrics=[str[1:] for str in data_DF[data_DF.columns[:-1]]]
    ubr = LogisticRegression()
    X = data_DF[data_DF.columns[:-1]].values
    y = data_DF[data_DF.columns[-1]].values
    ubr.fit(X,y)
    inter, coef, pVal = ubr.intercept_[0], ubr.coef_[0], f_classif(X,y)[1]
    for i in xrange(len(metrics)):
      thresh="%0.2f"%VARL(coef[i], inter, p0=0.1) if VARL(coef[i], inter, p0=0.1)>0 else "-"
      print(metrics[i]+"\t"+thresh+"\t"+"%0.3f"%pVal[i])
    # set_trace()
  return None

if __name__=="__main__":
  thresholds()
  pass
