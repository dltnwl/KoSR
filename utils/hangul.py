"""Utility module for handling Hangul characters."""

import re as _re

INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
"char list: Hangul initials (초성)"

MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
"char list: Hangul medials (중성)"

FINALS = list("∅ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
"char list: Hangul finals (종성), with an empty final character included."

JAMOS = sorted(set(INITIALS + MEDIALS + FINALS))
"char list: Hangul Jamos (자모)."

SYLLABLES = list(map(chr, range(0xAC00, 0xD7A3)))
"char list: Hangul syllables (가~힣)."

_INITIALS_IDX = {c: i for i, c in enumerate(INITIALS)}
_MEDIALS_IDX = {c: i for i, c in enumerate(MEDIALS)}
_FINALS_IDX = {c: i for i, c in enumerate(FINALS)}
_JAMOS_IDX = {c: i for i, c in enumerate(JAMOS)}
_SYLLABLES_IDX = {c: i for i, c in enumerate(SYLLABLES)}

def check_syllable(char):
  """Check whether given character is valid Hangul syllable."""
  return 0xAC00 <= ord(char) <= 0xD7A3

def split_syllable(char):
  """Split Hangul syllable into Jamos."""
  assert check_syllable(char)
  diff = ord(char) - 0xAC00
  m = diff % 28
  d = (diff - m) // 28
  return (INITIALS[d // 21], MEDIALS[d % 21], FINALS[m])

def _merge_jamos(initial, medial, final=None):
  """Merge Jamos into Hangul syllable.

  Raises:
    AssertionError: If ``initial``, ``medial``, and ``final`` are not in
      ``INITIAL``, ``MEDIAL``, and ``FINAL`` respectively.
  """
  assert initial in INITIALS
  assert medial in MEDIALS
  final = "∅" if final is None else final
  assert final in FINALS
  return chr(0xAC00 +
             588 * _INITIALS_IDX[initial] +
             28 * _MEDIALS_IDX[medial] +
             _FINALS_IDX[final])

def recover_splitted(text):
  """Recover splitted Hangul string.

  Raises:
    AssertionError: If ``text`` does not consist of Jamo, or does not obey the
      Jamo order.
  """
  result = ""
  i = 0
  l = len(text)
  while i < l:
    if text[i] == " ":
      result += " "
      i += 1
    else:
      if i+1 == l:
        result += text[i]
      elif i+2 == l:
        result += _merge_jamos(text[i], text[i+1])
      else:
        result += _merge_jamos(text[i], text[i+1], text[i+2])
      i += 3
  return result

def _text_preprocessor(f):
  """Decorator for text pre-processors.

  Note:
    All text pre-processors must be decorated with ``@_text_preprocessor``.
  """
  def decorated(text, **kwargs):
    """
  Args:
    text (str): Raw text.

  Returns:
    int list: Pre-processed text.
    """
    return f(text, **kwargs)
  decorated.__doc__ = "\n\n".join([f.__doc__, decorated.__doc__])
  return decorated

def _preprocess(text, split_syllables):
  """Hangul pre-processor.

  Drop invalid characters and adjacent whitespaces, then split Hangul
  syllables if you want. Finally, encode the string into integer list.

  Args:
    text (str): Raw text.
    split_syllables (boolean): Split Hangul syllable to Jamos or not.

  Returns:
    int list: Pre-processed text.
  """
  tmp = ""
  result = []
  for char in text:
    if char == " ":
      tmp += " "
    elif check_syllable(char):
      if split_syllables:
        tmp += "".join(split_syllable(char))
      else:
        tmp += char
  for char in _re.sub("\\s+", " ", tmp.strip()):
    if char == " ":
      if split_syllables:
        result.append(len(JAMOS) + 1)
      else:
        result.append(len(SYLLABLES) + 1)
    else:
      if split_syllables:
        result.append(_JAMOS_IDX[char] + 1)
      else:
        result.append(_SYLLABLES_IDX[char] + 1)
  return result

@_text_preprocessor
def jamo_token(text, **kwargs):
  """Jamo-based tokenizer."""
  return _preprocess(text, True)

@_text_preprocessor
def syllable_token(text, **kwargs):
  """Syllable-based tokenizer."""
  return _preprocess(text, False)
