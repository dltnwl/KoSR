"""Main entry of DLKoASR."""

import os
import os.path
import logging
import argparse
import tensorflow as tf
import model
import utils
import utils.hangul


parser = argparse.ArgumentParser()
parser.add_argument("--sess-dir", type=str, required=True,
                    help="Session directory")
parser.add_argument("--config", type=str, default="configs/default.yml",
                    help="Configuration file.")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s:%(lineno)d %(message)s")
log = logging.getLogger("DLKoASR")


if __name__ == "__main__":
  args = parser.parse_args()
  try:
    os.makedirs(args.sess_dir)
  except FileExistsError as e:
    if not os.path.isdir(args.sess_dir):
      raise e
  writers = {phase: tf.summary.FileWriter(os.path.join(args.sess_dir, phase))
             for phase in ["train", "valid"]}
  utils.load_config(args.config)

  model = model.KoSR(args.sess_dir)
  with tf.device(model.device):
    train_batch, valid_batch, test_batch = utils.load_batches()

  with tf.Session(config=model.config) as sess:
    saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)
    try:
      saver.restore(sess, tf.train.latest_checkpoint(args.sess_dir))
    except ValueError:
      log.info("==== TRAINING START ====")
      writers["train"].add_graph(sess.graph)
      model.train(sess, train_batch, valid_batch,
                  writers["train"], writers["valid"], saver)
      saver.save(sess, model.path)
    log.info("===== TESTING START ====")
    model.test(sess, test_batch)
