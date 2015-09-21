from pandas import DataFrame, read_csv, concat
from os import walk
from pdb import set_trace
import sys

def say(text):
  sys.stdout.write(str(text))

def csv2DF(dir, as_mtx=False, toBin=False):
  files=[]
  for f in dir:
    df=read_csv(f)
    headers = [h for h in df.columns if '?' not in h]
    if toBin:
      df[df.columns[-1]]=DataFrame([1 if d > 0 else 0 for d in df[df.columns[-1]]])
    files.append(df[headers])
  "For N files in a project, use 1 to N-1 as train."
  data_DF = concat(files)
  if as_mtx: return data_DF.as_matrix()
  else: return data_DF

def explore(dir='../Data/Jureczko/', name=None):
  datasets = []
  for (dirpath, dirnames, filenames) in walk(dir):
    datasets.append(dirpath)
  training = []
  testing = []
  if name:
    for k in datasets[1:]:
      if name in k:
        train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
        test = [train[0][0] + '/' + train[0][1].pop(-1)]
        training = [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store']
        testing = test
        return training, testing
  else:
    for k in datasets[1:]:
      train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
      test = [train[0][0] + '/' + train[0][1].pop(-1)]
      training.append(
          [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store'])
      testing.append(test)
    return training, testing
