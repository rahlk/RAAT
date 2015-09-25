from __future__ import division, print_function
from tools.containers import *
from pdb import set_trace
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeRegressor as CART

def ediv(lst, lvl=0,tiny=1,
         dull=0.3,
         num=lambda x:x[0], sym=lambda x:x[1]):
  "Divide lst of (numbers,symbols) using entropy."""
  #----------------------------------------------
  #print watch
  def divide(this,lvl): # Find best divide of 'this' lst.
    def ke(z): return k(z)*ent(z)
    def k(x): return len(Counter(x).keys())
    def ent(x): return sum([-Counter(x)[n]/len(x)*np.log2(Counter(x)[n]/len(x)) for n in Counter(x).keys()])
    lhs,rhs   = [], [x[-1] for x in this]
    n0,k0,e0,ke0= len(rhs),k(rhs),ent(rhs),k(rhs)*ent(rhs)
    cut, least  = None, e0
    last = num(this[0])
    for j,x  in enumerate(this):
      # rhs - sym(x); #nRhs - num(x)
      lhs.append(rhs.pop()); #nLhs + num(x)
      now = num(x)
      if now != last:
        if len(lhs) > tiny and len(rhs) > tiny:
          maybe= len(lhs)/n0*ent(lhs)+ len(rhs)/n0*ent(rhs)
          if maybe < least :
            gain = e0 - maybe
            delta= np.log2(float((3**k0-2)-(ke0- ke(rhs)-ke(lhs))))
            if gain >= (np.log2(n0-1) + delta)/n0:
              cut,least = j,maybe
      last= now
    return cut,least
  #--------------------------------------------
  def recurse(this, cuts,lvl):
    cut,e = divide(this,lvl)
    if cut:
      recurse(this[:cut], cuts, lvl+1);
      recurse(this[cut:], cuts, lvl+1)
    else:
      lo    = num(this[0])
      hi    = num(this[-1])
      cuts += [Thing(at=lo,
                     e=e,_has=this,
                     range=(lo,hi))]
    return cuts
  #---| main |-----------------------------------
  return recurse(sorted(lst,key=num),[],0)


def discreteNums(tbl):
   therows=[r.tolist() for r in tbl[tbl.columns[3:]].as_matrix()]
   for i, num in enumerate(tbl.columns[3:]):
      for cut in  ediv(therows, num=lambda x:x[i], sym=lambda x:x[-1]):
        for row in cut._has:
          row[i] = cut.range
   return np.array(therows, dtype=object)

def fselect(tbl, using='entropy', B=0.33):
  "Sort "
  clf = CART()
  features = tbl.columns[3:-1]
  klass = tbl[tbl.columns[-1]]
  clf.fit(tbl[features], klass)
  lbs = clf.feature_importances_
  cutoff = sorted(lbs, reverse=True)[int(len(lbs)*B)]
  return [features[i] for i,cc in enumerate(lbs) if cc>=cutoff]


def builder(tbl, lvl=-1, asIs=10 ** 32, up=None, features=None, klass = -1, branch=[],
          f=None, val=None, opt=None):
  rows = lambda t: [r.tolist() for r in t[features].as_matrix()]
  here = Thing(t=tbl, kids=[], f=f, val=val, up=up, lvl=lvl, rows=rows, modes={},
               branch=branch)

  def mode(lst):
    return np.max([Counter(lst)[k] for k in Counter(lst).keys()])
  here.mode = mode(tbl[tbl.columns[klass]])
  if lvl > 10:
    return here
  if asIs == 0:
    return here
  _, splitter, syms, splits = rankedFeatures(rows(tbl), tbl, features)[0]
  for key in sorted(splits.keys()):
    someRows = splits[key]
    toBe = syms[key].ent()
    if opt.variancePrune and lvl > 1 and toBe >= asIs:
      continue
    if opt.min <= len(someRows) < len(rows):
      here.kids += [tdiv1(t, someRows, lvl=lvl + 1, asIs=toBe, features=features,
                          up=here, f=splitter,
                          val=key, branch=branch + [(splitter, key)], opt=opt)]
  return here


def rankedFeatures(rows, t, features=None):
  features = features if features else t.indep
  klass = t.klass[0].col

  def ranked(f):
    syms, at, n = {}, {}, len(rows)
    for x in f.counts.keys():
      syms[x] = Sym()
    for row in rows:
      key = row.cells[f.col]
      val = row.cells[klass]
      syms[key] + val
      at[key] = at.get(key, []) + [row]
    e = 0
    for val in syms.values():
      if val.n:
        e += val.n / n * val.ent()
    return e, f, syms, at
  set_trace()
  return sorted(ranked(f) for f in features)


def dtree(tbl):
  features = fselect(tbl)
  tree = builder(tbl, features=features, branch=[])
  if opt.prune:
    modes(tree)
    prune(tree)
  return tree


def modes(n):
  if not n.modes:
    n.modes = {n.mode: True}
    for kid in n.kids:
      for mode in modes(kid):
        n.modes[mode] = True
  return n.modes


def nmodes(n):
  return len(n.modes.keys())


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

if __name__=='__main__':
  import sys
  sys.path.append(['~/git/axe/axe'])
  from table import *
  from dtree import *
  tbl_loc = '../Data/Jureczko/ant/ant-1.3.csv'
  tbl1 = discreteTable(tbl_loc)
  #
  tbl = pd.read_csv(tbl_loc)
  a = tdiv(tbl1)

  # for i,_ in enumerate(tbl.columns[4:-1]):
  #   for cut in  ediv(tbl[tbl.columns[4:-1]].as_matrix(),
  #                  num=lambda x:x[i],
  #                  sym=lambda x:x[-1]):
  #     for row in cut._has:
  #       row.tolist()[i]=cut.range
  #       print(row[i] , cut.range)
  set_trace()