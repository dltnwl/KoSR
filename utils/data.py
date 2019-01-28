"""Utility module for loading data."""

import re as _re
import json as _json
import os as _os
import os.path as _path
import random as _random
from itertools import groupby as _groupby


def _rawdata_loader(f):
  """Decorator for raw data lodaers.

  Load actual audio data from given audio file paths.

  Note:
    All data loaders must be decorated with ``@_rawdata_loader``.
  """
  def decorated(data_dir, batch_size, test_max_size, seed, **kwargs):
    """
  Args:
    data_dir (str): Root directory of data folder.

  Return:
    (str, str) list: List of audio paths and corresponding texts.
    """
    _random.seed(seed)
    train, valid, test = f(data_dir, batch_size, test_max_size, **kwargs)
    return train, valid, test
  decorated.__doc__ = "\n\n".join([f.__doc__, decorated.__doc__])
  return decorated

@_rawdata_loader
def opendict(data_dir, batch_size, test_max_size, **kwargs):
  """Load dictionary pronounciation data."""
  join = lambda f: _path.join(data_dir, f)
  with open(join("list.jl")) as f:
    data = [(join(e["files"][0]["path"]), e["word"].strip())
            for e in filter(lambda x: x.get("files"), map(_json.loads, f))]
  _random.shuffle(data)
  if not test_max_size:
    test_max_size = int(len(data) / 100)
  tmp = - (batch_size + test_max_size)
  return data[:tmp], data[tmp:-test_max_size], data[-test_max_size:]

@_rawdata_loader
def nangdok(data_dir, batch_size, test_max_size, **kwargs):
  """Load Nangdock corpus data."""
  join = lambda f: _path.join(data_dir, f)
  texts = []
  with open(join("script_nmbd_by_sentence.txt"), encoding="utf-16-le") as f:
    tmp = []
    for line in f.readlines():
      if line.startswith("<"):
        texts.append(tmp)
        tmp = []
      elif _re.match(r"^\d+\..*", line):
        tmp.append(line)
  texts.append(tmp)
  del texts[0]
  participants = sorted(filter(lambda l: _re.match("^[fm][v-z][0-9]+", l),
                               _os.listdir(data_dir)))
  test_sentences = kwargs.get("test_sentences",
                              [_random.choice(ts) for ts in texts])
  test_participants = kwargs.get("test_participants",
                                 [_random.choice(list(g))
                                  for _, g in _groupby(participants, lambda p: p[:2])])
  train = []
  test = []
  for participant in sorted(participants):
    for i, _ in enumerate(texts):
      for j, text in enumerate(_):
        f = join("{0}/{0}_t{1:0>2}_s{2:0>2}.wav".format(participant, i+1, j+1))
        if _path.isfile(f):
          if text in test_sentences or participants in test_participants:
            test.append((f, text))
          else:
            train.append((f, text))
  _random.shuffle(test)
  valid = test[:batch_size]
  if test_max_size and batch_size + test_max_size < len(test):
    test = test[batch_size:(batch_size + test_max_size)]
  else:
    test = test[batch_size:]
  return train, valid, test

@_rawdata_loader
def zeroth(data_dir, batch_size, test_max_size, **kwargs):
  """Load Zeroth project data."""
  data = {"train": [], "test": []}
  for phase in data:
    root = _path.join(data_dir, "{}_data_01/003/".format(phase))
    for i in _os.listdir(root):
      with open(_path.join(root, "{0}/{0}_003.trans.txt".format(i)),
                encoding="utf-8") as f:
        for line in f.readlines():
          p = _re.compile(r'[0-9]+[_][0-9]+[_][0-9]+')
          num = p.match(line)
          idx = num.group()
          p = _re.compile('[^0-9]+[^_0-9].')
          lab = p.findall(line)
          audio_path = _path.join(root, "{0}/{1}.flac".format(i, idx))
          data[phase].append((audio_path, lab[0][1:]))
  _random.shuffle(data["test"])
  valid = data["test"][:batch_size]
  if test_max_size and batch_size + test_max_size < len(data["test"]):
    test = data["test"][batch_size:(batch_size+test_max_size)]
  else:
    test = data["test"][batch_size:]
  return data["train"], valid, test
