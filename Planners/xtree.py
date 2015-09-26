from __future__ import division, print_function
from tools.containers import *
from tools.Discretize import discretize
from pdb import set_trace
import pandas as pd
from collections import Counter
import numpy as np
from tools.misc import *
from sklearn.tree import DecisionTreeClassifier as CART

def settings(**d):
  return Thing(
      min=1,
      infoPrune=0.33,
      variancePrune=True,
      debug=False,
      m=5,
      n=5,
      klass=-1,
      missing='?',
      better=lambda x: x.better,
      worse=lambda x: x.worse,
      cells=lambda x: x.cells,
      prune=False).override(d)


def prune(n):
  if nmodes(n) == 1:
    n.kids = []
  for kid in n.kids:
    prune(kid)


def classStats(n):
  depen = lambda x: x.cells[n.t.klass[0].col]
  return Sym(depen(x) for x in n.rows)


def showTdiv(n, lvl=-1):
  set_trace()
  import sys
  def say(x):
    sys.stdout.write(x)
  if n.f:
    say(('|..' * lvl) + str(n.f.name) + "=" + str(n.val) +
        "\t:" + str(n.mode))
  if n.kids:
    print('')
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

def infogain(t, opt=settings()):
  lst = rankedFeatures(t[t.columns].values.tolist(), t, features=t.columns[:opt.klass])
  n = int(len(lst)*opt.infoPrune)
  n = max(n, 1)
  # set_trace()
  return [f for e, f, syms, at in lst[:n]]

def rankedFeatures(rows, tbl, features=None, klass=-1):
  def ranked(i,f):
    syms, at, n = {}, {}, len(rows)
    def keys(i):
      return Counter(np.array(rows).transpose()[i]).keys()
    for x in keys(i):
      syms[x] = Sym()
    for row in rows:
      key = row[i]
      val = row[klass]
      syms[key] + val
      at[key] = at.get(key, []) + [row]
    e = 0
    for val in syms.values():
      if val.n:
        e += val.n / n * val.ent()
    return e, f, syms, at
  return sorted(ranked(i,f) for i,f in enumerate(features))

def builder(dtbl, rows=None, lvl=-1, asIs=10 ** 32, up=None, features=None, klass = -1, branch=[],
          f=None, val=None, opt=settings()):

  if not isinstance(rows, list): rows = dtbl[dtbl.columns].values.tolist()
  here = Thing(t=dtbl, kids=[], f=f, val=val, up=up, lvl=lvl, rows=rows, modes={},
               branch=branch)
  def mode(lst):
    return np.max([Counter(lst)[k] for k in Counter(lst).keys()])

  here.mode = mode(dtbl[dtbl.columns[klass]])
  if lvl > 10:
    return here
  if asIs == 0:
    return here
  _, splitter, syms, splits = rankedFeatures(rows, dtbl, features)[0]
  for key in sorted(splits.keys()):
    someRows = splits[key]
    toBe = syms[key].ent()
    if opt.min <= len(someRows) < len(rows):
      here.kids += [builder(dtbl, someRows, lvl=lvl + 1, asIs=toBe, features=features,
                          up=here, f=splitter,
                          val=key, branch=branch + [(splitter, key)], opt=opt)]
  return here


def dtree(tbl):
  features = infogain(discretize(tbl))
  tree = builder(discretize(tbl), features=features, branch=[])
  if settings().prune:
    modes(tree)
    prune(tree)
  return tree

def old():
  import sys
  sys.path.append(['~/git/axe/axe'])
  import table as t
  import dtree as dt
  tbl_loc = '../Data/Jureczko/ant/ant-1.3.csv'
  tbl = t.discreteTable(tbl_loc)
  dt = dt.tdiv(tbl)
  set_trace()

def new():
  tbl_loc = explore(name='ant')[0]
  tbl = csv2DF(tbl_loc)
  tree = dtree(tbl)
  set_trace()

if __name__=='__main__':
  # old()
  new()