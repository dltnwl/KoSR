"""Utility module for pre-processing audio."""

import logging as _logging
import numpy as _np
from scipy.signal import resample_poly as _resample_poly
import python_speech_features as _features


def _audio_preprocessor(f):
  """Decorator for audio pre-processors.

  Before applying given pre-processor, convert audio to mono. After
  pre-processing, pad 0 to the processed values to ensure that the time is
  divisible by 4.

  Note:
    All audio pre-processors must be decorated with ``@_audio_preprocessor``.
  """
  def decorated(audio, samplerate, n_features, n_channels, **kwargs):
    """
  Args:
    audio (:obj:`np.ndarray`): Raw audio data.
    samplerate (int): Sample rate of given audio.
    n_features (int): Number of features for pre-processed audio.
    n_channels (int): Number of channels for pre-processed audio.

  Returns:
    :obj:`np.array`: Pre-processed audio (Time-major).
    """
    max_sample_rate = 16000
    try:
      if audio.ndim == 2:
        audio = _np.mean(audio, 1)
      if samplerate > max_sample_rate:
        audio = _resample_poly(audio, max_sample_rate, samplerate)
        samplerate = max_sample_rate
      processed = f(audio, samplerate, n_features, n_channels, **kwargs)
      processed = _np.pad(processed,
                          ((0, (-processed.shape[0]) % 4), (0, 0), (0, 0)),
                          'constant')
      return processed
    except:
      _logging.exception("Exception raised while preprocessing audio:")
      return None
  decorated.__doc__ = "\n\n".join([f.__doc__, decorated.__doc__])
  return decorated

@_audio_preprocessor
def mfcc_with_delta(audio, samplerate, n_features, n_channels, **kwargs):
  """Calculate Mel-frequency cepstral coefficients, and calculate delta
  features if requested."""
  tmp = _features.mfcc(audio, samplerate, numcep=n_features, **kwargs)
  tmp -= _np.mean(tmp, axis=0) + 1e-8
  result = [tmp]
  for _ in range(1, n_channels):
    tmp = _features.delta(tmp, 2)
    result.append(tmp)
  result = _np.stack(result, axis=2)
  return result
