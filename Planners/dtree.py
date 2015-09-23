__author__ = 'rkrsn'


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

  return sorted(ranked(f) for f in features)


def infogain(t, opt=The.tree):
  def norm(x):
    return (x - lo) / (hi - lo + 0.0001)
  for f in t.headers:
    f.selected = False
  lst = rankedFeatures(t._rows, t)
  tmp = [l[0] for l in lst]
  n = int(len(lst))
  n = max(n, 1)
  for _, f, _, _ in lst[:n]:
    f.selected = True
  return [f for e, f, syms, at in lst[:n]]

