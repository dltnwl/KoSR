"""Utility module; various helper functions are implemented."""

import os as _os
import os.path as _path
import random as _random
import logging as _logging
from importlib import import_module as _import_module
import yaml as _yaml
import tensorflow as _tf
import soundfile as _sf
import utils.hangul as _hangul


CONFIG = _tf.contrib.training.HParams()
""":obj:`tf.contrib.training.HParams`: All configurations.

List of all configurable parameters are located in ``utils/__init__.py``."""


def load_config(filename):
  """Load a config file and update current configurations.

  Note:
    The config file should be in YAML format.

  Args:
    filename (str): Path of the config file.
  """
  # default configurations
  CONFIG.add_hparam('ALLOW_SOFT_PLACEMENT', True)
  CONFIG.add_hparam('LOG_DEVICE_PLACEMENT', False)
  CONFIG.add_hparam('DATA_TYPE', None)
  CONFIG.add_hparam('DATA_DIR', None)
  CONFIG.add_hparam('DATA_SEED', None)
  CONFIG.add_hparam('DATA_ARGS', {})
  CONFIG.add_hparam('BATCH_SIZE', 128)
  CONFIG.add_hparam('TEST_MAX_SIZE', 500)
  CONFIG.add_hparam('N_EPOCHS', 100)
  CONFIG.add_hparam('AUDIO_PREPROCESS', 'mfcc_with_delta')
  CONFIG.add_hparam('AUDIO_N_FEATURES', 13)
  CONFIG.add_hparam('AUDIO_N_CHANNELS', 3)
  CONFIG.add_hparam('AUDIO_PREPROCESS_ARGS', {})
  CONFIG.add_hparam('TEXT_PREPROCESS', 'jamo_token')
  CONFIG.add_hparam('TEXT_PREPROCESS_ARGS', {})
  CONFIG.add_hparam('ENCODER_ARGS', {'N_UNITS': 1024,
                                     'DROPOUT_PROB': 0.5,
                                     'USE_BAYES_RNN': True})
  CONFIG.add_hparam('DECODER', 'CTCAtt')
  CONFIG.add_hparam('DECODER_ARGS', {'EMBED_DIM': 16,
                                     'N_UNITS': 128,
                                     'LAMBDA': 0.2,
                                     'USE_JAMO_FSM': False,
                                     'BEAM_WIDTH': 64,
                                     'N_SAMPLES': 2})
  CONFIG.add_hparam('OPTIM', 'AdamOptimizer')
  CONFIG.add_hparam('OPTIM_GRADIENT_CLIP', 100.0)
  CONFIG.add_hparam('OPTIM_ARGS', {'learning_rate': 0.00025})
  CONFIG.add_hparam('CHECKPOINT_STEP', 5000)
  CONFIG.add_hparam('REPORT_STEP', 50)
  log = _logging.getLogger(__name__)
  try:
    with open(filename) as f:
      CONFIG.override_from_dict(_yaml.load(f.read()))
  except FileNotFoundError:
    log.warning("Config file not found; use default configurations.")
  except (_yaml.YAMLError, AttributeError, KeyError):
    log.warning("Error in config file; use default configurations.")
  finally:
    if CONFIG.TEXT_PREPROCESS == "jamo_token":
      CONFIG.add_hparam("LABELS", [""] + _hangul.JAMOS + [" "])
    elif config['TEXT_PREPROCESS'] == "syllable_token":
      CONFIG.add_hparam("LABELS", [""] + _hangul.SYLLABLES + [" "])
    log.info("Config file %s loaded.", filename)

def _preprocess(data):
  """Pre-processing raw data.

  Args:
    data ((str, str) list): Raw data.

  Yields:
    (:obj:`np.array`, int, int list, int list, int): Pre-processed data.

    Each element of the tuple denotes:
      - Audio.
      - Length of audio.
      - Text with prepended `<sos>` token.
      - Text with appended `<eos>` token.
      - Length of text.
  """
  preprocess_audio = getattr(_import_module("utils.audio"),
                             CONFIG.AUDIO_PREPROCESS)
  preprocess_text = getattr(_import_module("utils.hangul"),
                            CONFIG.TEXT_PREPROCESS)
  for (audio_path, text) in data:
    try:
      audio, samplerate = _sf.read(audio_path)
    except RuntimeError:
      _logging.exception("Exception raised while loading %s:", audio_path)
      try:
        _os.remove(audio_path)
      except FileNotFoundError:
        pass
    else:
      t = int(100 * audio.shape[0] / samplerate) + 1
      text = preprocess_text(text, **CONFIG.TEXT_PREPROCESS_ARGS)
      if t < 2000 and t >= 4 * len(text):
        audio = preprocess_audio(audio, samplerate,
                                 CONFIG.AUDIO_N_FEATURES,
                                 CONFIG.AUDIO_N_CHANNELS,
                                 **CONFIG.AUDIO_PREPROCESS_ARGS)
        if audio is not None:
          yield (audio, audio.shape[0], [0] + text, text + [0], len(text) + 1)

def _create_dataset(data, batch_size, n_epochs):
  """Create a Tensorflow dataset with given raw data.

  Args:
    data ((str, str) list): Raw data.
    batch_size (int): Batch size.
    n_epochs (int): Number of epochs.

  Return:
    :obj:`tf.data.Dataset`: Tensorflow dataset.
  """
  shape = ([None, CONFIG.AUDIO_N_FEATURES, CONFIG.AUDIO_N_CHANNELS],
           [], [None], [None], [])
  dataset = _tf.data.Dataset.from_generator(
      lambda: _preprocess(data),
      (_tf.float32, _tf.int32, _tf.int32, _tf.int32, _tf.int32), shape).cache()
  if n_epochs and n_epochs > 1:  # training
    dataset = dataset.apply(
        _tf.contrib.data.shuffle_and_repeat(len(data) // 10, n_epochs))
    dataset = dataset.apply(
        _tf.contrib.data.bucket_by_sequence_length(
            lambda audio, audio_time, sos_text, text_eos, text_length: audio_time,
            [25 * x for x in range(1, 80)],
            [batch_size] * 20 + [batch_size // 2] * 20 + [batch_size // 4] * 40))
  else:
    dataset =  dataset.repeat(n_epochs).padded_batch(batch_size, shape)
  return dataset.prefetch(batch_size)

def load_batches():
  """Create three Tensorflow tensors which fetches a next batch from datasets.
  Each tensors fetch from training set, valid set, and test set respectively.

  Note that the batch size and the epochs of training set are determined by the
  corresponding values in ``utils.CONFIG``.

  The batch size of valid set and test set is fixed to 1. The epochs of test
  set is fixed to 1, while valid set has no limit in number of epochs.

  Returns:
    (:obj:`tf.Tensor`, :obj:`tf.Tensor`, :obj:`tf.Tensor`): Tensorflow tensors
    that fetches next batch from training set, valid set, and test set
    respectively.
  """
  assert _path.isdir(CONFIG.DATA_DIR)
  load_rawdata = getattr(_import_module("utils.data"), CONFIG.DATA_TYPE)
  train_set, valid_set, test_set = load_rawdata(CONFIG.DATA_DIR,
                                                CONFIG.BATCH_SIZE,
                                                CONFIG.TEST_MAX_SIZE,
                                                CONFIG.DATA_SEED,
                                                **CONFIG.DATA_ARGS)
  train_set = _create_dataset(train_set, CONFIG.BATCH_SIZE, CONFIG.N_EPOCHS)
  valid_set = _create_dataset(valid_set, CONFIG.BATCH_SIZE, None)
  test_set = _create_dataset(test_set, CONFIG.BATCH_SIZE, 1)
  return tuple(map(lambda dataset: dataset.make_one_shot_iterator().get_next(),
                   (train_set, valid_set, test_set)))
