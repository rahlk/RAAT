from __future__ import division, print_function
from rf import rf
import numpy as np
import random
from time import time
from pdb import set_trace

class settings:
  iter=50,
  N=100,
  f=0.5,
  cf=0,
  maxIter=100,
  lives=10

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

def de0(model, new=[], pop=int(1e4), iter=1000, lives=5, settings=settings):
  frontier = model.generate(pop)

  def cdom(x, y, better='more'):

    def loss1(i,x,y):
      return (x - y) if better[i] == 'less' else (y - x)

    def expLoss(i,x,y,n):
      return np.exp(loss1(i,x,y) / n)

    def loss(x, y):
      n      = min(len(x), len(y)) #lengths should be equal
      losses = [expLoss(i,xi,yi,n) for i, (xi, yi) in enumerate(zip(x,y))]
      return sum(losses)/n

    "x dominates y if it losses least"
    if not isinstance(x,list):
      return x<y if better=='less' else x>y
    else:
      return loss(x,y) < loss(y,x)


  def extrapolate(current, l1, l2, l3):
    def extrap(i,a,x,y,z):
      return (max(model.dec_lim[i][0], min(model.dec_lim[i][1], x + settings.f * (z - y)))) if random.random()>settings.cf else a
    return [extrap(i, a,x,y,z) for i, a, x, y, z in zip(range(model.n_dec), current, l3, l1, l2)]

  def one234(one, pop):
    ids = [i for i,p in enumerate(pop) if not p==one]
    a = np.random.choice(ids, size=3, replace=False)
    return one, pop[a[0]], pop[a[1]], pop[a[2]]

  while lives > 0 and iter>0:
    better = False
    xbest = random.choice(frontier)
    for pos in xrange(len(frontier)):
      iter -= 1
      lives -= 1
      now, l1, l2, l3 = one234(frontier[pos], frontier)
      # set_trace()
      new = extrapolate(now, l1, l2, l3)
      newVal=model.solve(new)
      oldVal=model.solve(now)
      # print(iter, lives)
      if cdom(newVal, oldVal):
        frontier.pop(pos)
        frontier.insert(pos, new)
        lives += 1
      elif cdom(model.solve(frontier[pos]), model.solve(new)):
        better = False
      else:
        frontier.append(new)
        lives += 1

  return sorted(frontier, key=lambda F: model.solve(F))[-1]

def tuner(data):
  if len(data)==1:
    return None
  else:
    return de0(model = rf(data=data),pop=10, iter=100)
