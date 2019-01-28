"""Build automated speech recognition model."""

import os.path as _path
import logging as _logging
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import sparse_tensor as _sparse_tensor
import editdistance as _editdistance
from utils import CONFIG as _CONFIG
from utils.hangul import recover_splitted as _recover_splitted

class KoSR:
  """Tensorflow implementation of Korean speech recognition model.

  Attributes:
    audio (:obj:`tf.Tensor`): Placeholder for pre-processed audio.
    audio_time (:obj:`tf.Tensor`): Placeholder for length of pre-processed
      audio.
    sos_text (:obj:`tf.Tensor`): Placeholder for pre-processed text with
      prepended `<sos>` token.
    text_eos (:obj:`tf.Tensor`): Placeholder for pre-processed text with
      appended `<eos>` token.
    text_length (:obj:`tf.Tensor`): Placeholder for length of pre-processed
      text.
    dropout (:obj:`tf.Tensor`): Placeholder for enable dropout.
    decoded (:obj:`tf.Tensor`): Decoded tensor.
    _loss (:obj:`tf.Tensor`): Loss tensor.
    _error (:obj:`tf.Tensor`): Error rate tensor.
    _train_op (:obj:`tf.Tensor`): Training step tensor.
    path (str): File path of saved model.
    config (:obj:`tf.ConfigProto`): Tensorflow configurations.
    device (callable): Tensorflow device function.
  """
  def __init__(self, sess_dir):
    if _CONFIG.DECODER not in ("CTC", "Att", "CTCAtt"):
      raise NotImplementedError
    enc_n_units = _CONFIG.ENCODER_ARGS.get("N_UNITS", 1024)
    dropout_prob = _CONFIG.ENCODER_ARGS.get("DROPOUT_PROB", 0.5)
    bayes_rnn = _CONFIG.ENCODER_ARGS.get("USE_BAYES_RNN", True)
    dec_n_units = _CONFIG.DECODER_ARGS.get("N_UNITS", 128)
    embed_dim = _CONFIG.DECODER_ARGS.get("EMBED_DIM", 16)
    weight = _CONFIG.DECODER_ARGS.get("LAMBDA", 0.2)
    use_jamo_fsm = _CONFIG.DECODER_ARGS.get("USE_JAMO_FSM", False)
    self.labels = _CONFIG.LABELS
    self.beam_width = _CONFIG.DECODER_ARGS.get("BEAM_WIDTH", 64)
    self.n_samples = _CONFIG.DECODER_ARGS.get("N_SAMPLES", 2)

    self.path = _path.join(sess_dir, "model")
    self.config = _tf.ConfigProto(
        gpu_options=_tf.GPUOptions(allow_growth=True),
        allow_soft_placement=_CONFIG.ALLOW_SOFT_PLACEMENT,
        log_device_placement=_CONFIG.LOG_DEVICE_PLACEMENT)
    self.device = _tf.train.replica_device_setter(worker_device="/gpu:0",
                                                  ps_device="/cpu:0",
                                                  ps_tasks=1)

    with _tf.device(self.device):
      global_step = _tf.train.get_or_create_global_step()
      self.audio = _tf.placeholder(_tf.float32, [None, None,
                                                 _CONFIG.AUDIO_N_FEATURES,
                                                 _CONFIG.AUDIO_N_CHANNELS])
      self.audio_time = _tf.placeholder(_tf.int32, [None])
      self.sos_text = _tf.placeholder(_tf.int32, [None, None])
      self.text_eos = _tf.placeholder(_tf.int32, [None, None])
      self.text_length = _tf.placeholder(_tf.int32, [None])
      self.dropout = _tf.placeholder(_tf.bool)

      keep_prob = _tf.cond(self.dropout, lambda: 1.0-dropout_prob, lambda: 1.0)
      n_labels = len(self.labels)
      x, xs, h = self._set_encoder(self.audio, self.audio_time,
                                   _CONFIG.AUDIO_N_FEATURES, enc_n_units,
                                   keep_prob, bayes_rnn)

      if _CONFIG.DECODER[:3] == "CTC":
        x = _tf.reshape(x, [-1, 2 * enc_n_units])
        x = _tf.contrib.layers.fully_connected(x, n_labels,
                                               biases_initializer=None,
                                               activation_fn=None)
        x = _tf.reshape(x, [_tf.shape(xs)[0], -1, n_labels])
        x_t = _tf.transpose(x, (1, 0, 2))
        y_eos_sp = _tf.contrib.layers.dense_to_sparse(self.text_eos-1,
                                                      eos_token=-1)
        ctc_loss = self._set_ctc_loss(x_t, xs, y_eos_sp)

      if _CONFIG.DECODER[-3:] == "Att":
        embed_var = _tf.get_variable("embed", [n_labels, embed_dim])
        decode_var = _tf.nn.rnn_cell.GRUCell(2 * enc_n_units)

        x = _tf.contrib.seq2seq.tile_batch(x, self.beam_width)
        xs_tile = _tf.contrib.seq2seq.tile_batch(xs, self.beam_width)
        h = _tf.contrib.seq2seq.tile_batch(h, self.beam_width)
        yi = _tf.contrib.seq2seq.tile_batch(self.sos_text, self.beam_width)
        y = _tf.contrib.seq2seq.tile_batch(self.text_eos, self.beam_width)
        ys = _tf.contrib.seq2seq.tile_batch(self.text_length, self.beam_width)

        attention = _tf.contrib.seq2seq.BahdanauMonotonicAttention(dec_n_units, x, xs_tile)
        cell = _tf.contrib.seq2seq.AttentionWrapper(decode_var, attention)
        initial = cell.zero_state(_tf.shape(xs_tile)[0], _tf.float32).clone(cell_state=h)
        fc = _tf.layers.Dense(n_labels, use_bias=False)
        att_loss, self._error = self._set_att_loss_error(
            embed_var, cell, initial, fc, yi, y, ys)
        self.decoded, self.prob = \
            self._set_att_decoder(embed_var, cell, initial, fc,
                                  xs, self.beam_width, use_jamo_fsm)
      elif _CONFIG.DECODER[-3:] == "CTC":
        self._error = self._set_ctc_error(x_t, xs, y_eos_sp)
        self.decoded, self.prob = \
            self._set_ctc_decoder(x_t, xs, self.beam_width, use_jamo_fsm)

      if _CONFIG.DECODER == "CTC":
        self._loss = ctc_loss
      elif _CONFIG.DECODER == "Att":
        self._loss = att_loss
      else:
        self._loss = weight * ctc_loss + (1 - weight) * att_loss
      self._train_op = self._set_optim(_CONFIG.OPTIM, _CONFIG.OPTIM_ARGS,
                                       self._loss, _CONFIG.OPTIM_GRADIENT_CLIP,
                                       global_step)

  def _set_encoder(self, x, xs, n_features, n_units, keep_prob, bayes_rnn):
    bs = _tf.shape(xs)[0]
    xs = xs // 4
    x = _tf.contrib.layers.repeat(x, 2, _tf.contrib.layers.conv2d, 64, 3)
    x = _tf.contrib.layers.max_pool2d(x, (2, 1), (2, 1))
    x = _tf.contrib.layers.repeat(x, 2, _tf.contrib.layers.conv2d, 128, 3)
    x = _tf.contrib.layers.max_pool2d(x, (2, 1), (2, 1))
    x = _tf.reshape(x, [bs, -1, 128 * n_features])
    cells = [
        _tf.contrib.rnn.DropoutWrapper(_tf.contrib.rnn.BasicLSTMCell(n_units),
                                       output_keep_prob=keep_prob,
                                       state_keep_prob=keep_prob,
                                       input_keep_prob=keep_prob,
                                       variational_recurrent=bayes_rnn,
                                       input_size=128 * n_features,
                                       dtype=_tf.float32)
        for _ in range(2)]
    x, h = _tf.nn.bidirectional_dynamic_rnn(*cells, x, sequence_length=xs,
                                            dtype=_tf.float32)
    h = _tf.concat((h[0][0], h[1][0]), 1)
    x = _tf.concat(x, 2)
    return x, xs, h

  def _set_ctc_loss(self, x_t, xs, y_sp):
    return _tf.reduce_mean(_tf.nn.ctc_loss(y_sp, x_t, xs))

  def _set_ctc_error(self, x_t, xs, y_sp):
    error, _ = _tf.nn.ctc_greedy_decoder(x_t, xs)
    return _tf.reduce_mean(_tf.edit_distance(_tf.cast(error[0], _tf.int32),
                                             y_sp))

  def _set_att_loss_error(self, embed_var, cell, initial, fc, yi, y, ys):
    helper = _tf.contrib.seq2seq.TrainingHelper(
        _tf.nn.embedding_lookup(embed_var, yi), ys)
    train = _tf.contrib.seq2seq.BasicDecoder(cell, helper, initial, fc)
    train, _, _ = _tf.contrib.seq2seq.dynamic_decode(
        train, maximum_iterations=_tf.reduce_max(ys))
    loss = _tf.reduce_mean(_tf.reduce_sum(
        _tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=train.rnn_output), axis=1))
    error = _tf.reduce_mean(
        _tf.edit_distance(_tf.contrib.layers.dense_to_sparse(train.sample_id),
                          _tf.contrib.layers.dense_to_sparse(y)))
    return loss, error

  def _set_ctc_decoder(self, x_t, xs, beam_width, use_jamo_fsm):
    if use_jamo_fsm:
      custom = _tf.load_op_library('custom/jamo_search.so')
      _ops.NotDifferentiable("CTCJamoSearchDecoder")
      ixs, vals, shapes, log_prob = (
        custom.ctc_jamo_search_decoder(x_t, xs, beam_width=beam_width,
                                       top_paths=beam_width, merge_repeated=False))
      decoded = [_sparse_tensor.SparseTensor(ix, val, shape)
                 for ix, val, shape in zip(ixs, vals, shapes)]
    else:
      decoded, log_prob = _tf.nn.ctc_beam_search_decoder(x_t, xs,
                                                         beam_width, beam_width)
    decoded = [_tf.sparse_tensor_to_dense(d, -1) + 1 for d in decoded]
    # FIXME: log_prob is *incorrect* (see tensorflow Issue #6034)
    return decoded, _tf.transpose(_tf.exp(log_prob))

  def _set_att_decoder(self, embed_var, cell, initial, fc, xs, beam_width,
                       use_jamo_fsm):
    if use_jamo_fsm:
      from custom.jamo_beam_search_decoder import BeamSearchDecoder as _BeamSearchDecoder
      infer = _BeamSearchDecoder(
          cell, embed_var, _tf.fill([_tf.shape(xs)[0]], 0), 0, initial,
          beam_width, fc)
    else:
      infer = _tf.contrib.seq2seq.BeamSearchDecoder(
          cell, embed_var, _tf.fill([_tf.shape(xs)[0]], 0), 0, initial,
          beam_width, fc)
    infer, _, _ = _tf.contrib.seq2seq.dynamic_decode(
        infer, maximum_iterations=_tf.reduce_max(xs))
    decoded = _tf.transpose(infer.predicted_ids, (2, 0, 1))
    log_prob = infer.beam_search_decoder_output.scores
    # FIXME: is log_prob correct?
    prob = _tf.exp(_tf.reduce_sum(_tf.transpose(log_prob, (2, 0, 1)), axis=2))
    return decoded, prob

  def _set_optim(self, optim, optim_args, loss, grad_clip, step):
    optim = getattr(_tf.train, optim)(**optim_args)
    if grad_clip and grad_clip > 0.0:
      grad, variables = zip(*optim.compute_gradients(loss))
      grad, _ = _tf.clip_by_global_norm(grad, grad_clip)
      return optim.apply_gradients(zip(grad, variables), step)
    return optim.minimize(loss, step)

  def train(self, sess, train_batch, valid_batch, train_writer, valid_writer, saver):
    """Train the model.

    Args:
      sess (:obj:`tf.Session`): Tensorflow session.
      train_batch (:obj:`tf.Tensor`): Tensorflow tensor which fetches a next batch
        from training set.
      valid_batch (:obj:`tf.Tensor`): Tensorflow tensor which fetches a next batch
        from validation set.
      train_writer (:obj:`tf.summary.FileWriter`): Tensorflow summary writer.
      valid_writer (:obj:`tf.summary.FileWriter`): Tensorflow summary writer.
      saver (:obj:`tf.Saver`): Tensorflow saver.
    """
    log = _logging.getLogger(__name__)
    summary_proto = _tf.Summary()
    _tf.summary.scalar("Loss", self._loss)
    _tf.summary.scalar("Error_rate", self._error)
    summary = _tf.summary.merge_all()
    global_step = _tf.train.get_global_step()
    sess.run(_tf.global_variables_initializer())
    while True:
      try:
        audio, audio_time, sos_text, text_eos, text_length = \
          sess.run(train_batch)
        feed = {self.audio: audio,
                self.audio_time: audio_time,
                self.sos_text: sos_text,
                self.text_eos: text_eos,
                self.text_length: text_length,
                self.dropout: True}
        _, step = sess.run([self._train_op, global_step], feed)
        if _CONFIG.REPORT_STEP and step % _CONFIG.REPORT_STEP == 0:
          summaries = sess.run(summary, feed)
          train_writer.add_summary(summaries, step)
          summary_proto.ParseFromString(summaries)
          log.info("Step %-5d: Train Loss % 6.2f / Train LER % 6.4f", step,
                   summary_proto.value[0].simple_value,
                   summary_proto.value[1].simple_value)
          audio, audio_time, sos_text, text_eos, text_length = \
            sess.run(valid_batch)
          summaries = sess.run(summary, {self.audio: audio,
                                         self.audio_time: audio_time,
                                         self.sos_text: sos_text,
                                         self.text_eos: text_eos,
                                         self.text_length: text_length,
                                         self.dropout: False})
          valid_writer.add_summary(summaries, step)
          summary_proto.ParseFromString(summaries)
          log.info("            Valid Loss % 6.2f / Valid LER % 6.4f",
                   summary_proto.value[0].simple_value,
                   summary_proto.value[1].simple_value)
        if _CONFIG.CHECKPOINT_STEP and step % _CONFIG.CHECKPOINT_STEP == 0:
          saver.save(sess, self.path)
      except _tf.errors.OutOfRangeError:
        break

  def test(self, sess, batch):
    """Test the model.

    Args:
      sess (:obj:`tf.Session`): Tensorflow session.
      batch (:obj:`tf.Tensor`): Tensorflow tensor which fetches a next batch
        from training set.
    """
    log = _logging.getLogger(__name__)
    n, ler, wer = 0, 0, 0
    print("Original,Decoded")
    while True:
      try:
        audio, audio_time, sos_text, text_eos, text_length = \
          sess.run(batch)
        bs = audio_time.shape[0]
        results = [{} for _ in range(bs)]
        for _ in range(self.n_samples):
          decoded, prob = sess.run([self.decoded, self.prob],
                                   {self.audio: audio,
                                    self.audio_time: audio_time,
                                    self.sos_text: sos_text,
                                    self.text_eos: text_eos,
                                    self.text_length: text_length,
                                    self.dropout: self.n_samples > 1})
          for i in range(self.beam_width):
            for j in range(bs):
              d, p = decoded[i][j], prob[i,j]
              d = "".join([self.labels[i] for i in d])
              results[j][d] = results[j].get(d, 0) + (p / self.n_samples)
        for d, t in zip(map(lambda result: max(result, key=result.get), results),
                        text_eos):
          n += 1
          t = "".join([self.labels[i] for i in t])
          ler += _editdistance.eval(t, d)/len(t)
          tmp = t.split()
          wer += _editdistance.eval(tmp, d.split())/len(tmp)
          try:
            d = _recover_splitted(d)
          except (AssertionError, KeyError):
            pass
          finally:
            print(_recover_splitted(t) + "," + d)
      except _tf.errors.OutOfRangeError:
        break
    if n > 0:
      log.info("LER: %f", ler / n)
      log.info("WER: %f", wer / n)
