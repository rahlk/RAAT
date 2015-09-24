from __future__ import division, print_function
from tools.containers import *
from pdb import set_trace
import pandas as pd


# def rankedFeatures(rows, t, features=None):
#   features = features if features else t.indep
#   klass = t.klass[0].col
#
#   def ranked(f):
#     syms, at, n = {}, {}, len(rows)
#     for x in f.counts.keys():
#       syms[x] = Sym()
#     for row in rows:
#       key = row.cells[f.col]
#       val = row.cells[klass]
#       syms[key] + val
#       at[key] = at.get(key, []) + [row]
#     e = 0
#     for val in syms.values():
#       if val.n:
#         e += val.n / n * val.ent()
#     return e, f, syms, at
#
#   return sorted(ranked(f) for f in features)


# def infogain(t, opt=The.tree):
#   def norm(x):
#     return (x - lo) / (hi - lo + 0.0001)
#   for f in t.headers:
#     f.selected = False
#   lst = rankedFeatures(t._rows, t)
#   tmp = [l[0] for l in lst]
#   n = int(len(lst))
#   n = max(n, 1)
#   for _, f, _, _ in lst[:n]:
#     f.selected = True
#   return [f for e, f, syms, at in lst[:n]]

def ediv(lst, lvl=0,tiny=1,
         dull=0.33,
         num=lambda x:x[x.columns[0]], sym=lambda x:x[x.columns[-1]]):
  "Divide lst of (numbers,symbols) using entropy."""
  #----------------------------------------------
  #print watch
  def divide(this,lvl): # Find best divide of 'this' lst.
    def ke(z): return z.k()*z.ent()
    lhs,rhs   = Sym(), Sym(sym(x) for x in this)
    n0,k0,e0,ke0= 1.0*rhs.n,rhs.k(),rhs.ent(),ke(rhs)
    cut, least  = None, e0
    last = num(this[0])
    for j,x  in enumerate(this):
      rhs - sym(x); #nRhs - num(x)
      lhs + sym(x); #nLhs + num(x)
      now = num(x)
      if now != last:
        if lhs.n > tiny and rhs.n > tiny:
          maybe= lhs.n/n0*lhs.ent()+ rhs.n/n0*rhs.ent()
          if maybe < least :
            gain = e0 - maybe
            delta= log2(float((3**k0-2)-(ke0- ke(rhs)-ke(lhs))))
            if gain >= (log2(n0-1) + delta)/n0:
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

if __name__=='__main__':
  import sys
  sys.path.append(['~/git/axe/axe'])
  # from table import *
  # from dtree import *
  tbl_loc = '../Data/Jureczko/ant/ant-1.3.csv'
  # tbl = discreteTable(tbl_loc)

  tbl = pd.read_csv(tbl_loc)
  for i,_ in enumerate(tbl.columns[4:-1]):
    for cut in  ediv(tbl[tbl.columns[4:-1]].as_matrix(),
                   num=lambda x:x[i],
                   sym=lambda x:x[-1]):
      for row in cut._has:
        row.tolist()[i]=cut.range
        print(row[i] , cut.range)
  set_trace()