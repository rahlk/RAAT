from __future__ import division, print_function
from tools.containers import *
from tools.Discretize import discretize
from pdb import set_trace
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier as CART

def fselect(tbl, B=0.33):
  "Sort "
  clf = CART(criterion='entropy')
  features = tbl.columns[3:-1]
  klass = tbl[tbl.columns[-1]]
  clf.fit(tbl[features], klass)
  lbs = clf.feature_importances_
  cutoff = sorted(lbs, reverse=True)[int(len(lbs)*B)]
  return [features[i] for i,cc in enumerate(lbs) if cc>=cutoff]

def prune(n):
  if nmodes(n) == 1:
    n.kids = []
  for kid in n.kids:
    prune(kid)


def classStats(n):
  depen = lambda x: x.cells[n.t.klass[0].col]
  return Sym(depen(x) for x in n.rows)


def showTdiv(n, lvl=-1):
  if n.f:
    say(('|..' * lvl) + str(n.f.name) + "=" + str(n.val) +
        "\t:" + str(n.mode) + " #" + str(nmodes(n)))
  if n.kids:
    nl()
    for k in n.kids:
      showTdiv(k, lvl + 1)
  else:
    s = classStats(n)
    print(' ' + str(int(100 * s.counts[s.mode()] / len(n.rows))) + '% * ' + str(len(n.rows)))


def dtnodes(tree, lvl=0):
  if tree:
    yield tree, lvl
    for kid in tree.kids:
      lvl1 = lvl
      for sub, lvl1 in dtnodes(kid, lvl1 + 1):
        yield sub, lvl1


def dtleaves(tree):
  for node, _ in dtnodes(tree):
    # print "K>", tree.kids[0].__dict__.keys()
    if not node.kids:
      yield node


def apex(test, tree):
  """apex=  leaf at end of biggest (most supported)
   branch that is selected by test in a tree"""
  def equals(val, span):
    if val == opt.missing or val == span:
      return True
    else:
      if isinstance(span, tuple):
        lo, hi = span
        return lo <= val <= hi  # <hi
      else:
        return span == val

  def apex1(cells, tree):
    found = False
    for kid in tree.kids:
      val = cells[kid.f.col]
      if equals(val, kid.val):
        for leaf in apex1(cells, kid):
          found = True
          yield leaf
    if not found:
      yield tree
  leaves = [(len(leaf.rows), leaf)
            for leaf in apex1(opt.cells(test), tree)]
  return second(last(sorted(leaves)))

def rankedFeatures(rows, t, features=None, klass=-1):

  # features = tbl.columns
  # klass = tbl.columns[-1]

  def ranked(f):
    syms, at, n = {}, {}, len(rows)
    for x in f.counts.keys():
      syms[x] = Sym()
    for row in rows:
      key = row[i]
      val = row[klass]
      set_trace()
      syms[key] + val
      at[key] = at.get(key, []) + [row]
    e = 0
    for val in syms.values():
      if val.n:
        e += val.n / n * val.ent()
    return e, f, syms, at
  return sorted(ranked(f) for f in enumerate(features))

def builder(tbl, rows=None, lvl=-1, asIs=10 ** 32, up=None, features=None, klass = -1, branch=[],
          f=None, val=None, opt=None):

  if rows==None:
    rows=discretize(tbl)
  here = Thing(t=tbl, kids=[], f=f, val=val, up=up, lvl=lvl, rows=rows, modes={},
               branch=branch)
  set_trace()
  def mode(lst):
    return np.max([Counter(lst)[k] for k in Counter(lst).keys()])
  here.mode = mode(tbl[tbl.columns[klass]])
  if lvl > 10:
    return here
  if asIs == 0:
    return here
  _, splitter, syms, splits = rankedFeatures(rows, tbl, features)[0]
  for key in sorted(splits.keys()):
    someRows = splits[key]
    toBe = syms[key].ent()
    if opt.variancePrune and lvl > 1 and toBe >= asIs:
      continue
    if opt.min <= len(someRows) < len(rows):
      here.kids += [builder(tbl, someRows, lvl=lvl + 1, asIs=toBe, features=features,
                          up=here, f=splitter,
                          val=key, branch=branch + [(splitter, key)], opt=opt)]
  return here


def dtree(tbl):
  features = fselect(tbl)
  tree = builder(tbl, features=features, branch=[])
  if opt.prune:
    modes(tree)
    prune(tree)
  return tree


if __name__=='__main__':
  import sys
  sys.path.append(['~/git/axe/axe'])
  import table as t
  import dtree as dt
  tbl_loc = '../Data/Jureczko/ant/ant-1.3.csv'
  tbl = pd.read_csv(tbl_loc)
  a = dtree(tbl)
  set_trace()