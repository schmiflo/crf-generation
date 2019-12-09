import argparse
import tensorflow as tf
from toy_language import get_toy_dataset
import numpy as np
from tensorflow.python.framework.errors_impl import OutOfRangeError

from models.iid import IIDModel
from models.tf_crf import TFCRF
from models.emb_crf import ContextualEmbCRF
import random
from models.tf_utils import add_optimizer_graph, get_positional_states,\
  load_asymmetric_glove_embedding


parser = argparse.ArgumentParser()

parser.add_argument("--model", default="emb-crf", choices=["iid", "tf-crf", "emb-crf"], type=str)
parser.add_argument("--model_dir", default=None, type=str)


# Data parameters
parser.add_argument("--vocab_size", default=20, type=int)
parser.add_argument("--min_sequence_length", default=2, type=int)
parser.add_argument("--max_sequence_length", default=10, type=int)


# Training and Optimization
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--initial_learning_rate", default=0.001, type=float)
parser.add_argument("--max_grad_norm", default=5, type=float)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--num_train_samples", default=640000, type=int)
parser.add_argument("--num_test_samples", default=6400, type=int)

# Printing
parser.add_argument("--test_every", default=500, type=int, help="nr steps")
parser.add_argument("--print_every", default=100, type=int, help="nr steps")
parser.add_argument("--summary_every", default=10, type=int, help="nr steps")
parser.add_argument("--sample_every", default=100, type=int, help="nr steps")

# The (constant) states
parser.add_argument("--state_size", default=100, type=int)


# The CRFs
parser.add_argument("--crf_S_diag", default=True, type=bool)
parser.add_argument("--S_fn", default="none", type=str)
parser.add_argument("--crf_emb_size", default=100, type=int)
parser.add_argument("--crf_d", default=50, type=int)
parser.add_argument("--nr_s_layers", default=1, type=int)
parser.add_argument("--embedding_method", default="transformer", choices=["transformer", "simple", "onehot", "zeros"], type=str)
parser.add_argument("--crf_transitions", default="emb-contextual", choices=["emb-contextual", "single-matrix", "multi-matrix"], type=str)

parser.add_argument("--crf_pretrained_embedding", default=None, type=str)
'''
  Must be in non-binary gensim word2vec format
  Noote that the paper uses different GloVe embeddings for X and Y from
  factorizing the pairwise co-occurrences. To obtain such embeddings, use the original
  GloVe code and use it with WINDOW_SIZE=2 and BINARY=0 and set -symmetric 0 for cooccur
  In the resulting vectors file, add the number of words and dimensions in the first line
  to comply with the word2vec format (see the example encodings in the repository)
'''

# Randomness
# Remember that the data is random, so fix the seeds for train and test
parser.add_argument("--train_seed", default=1, type=int)
parser.add_argument("--test_seed", default=2000, type=int)
# There is some randomness in the initialization. We run across ten seeds
# and report for an average run. 
parser.add_argument("--model_seed", default=2, type=int)





flags = parser.parse_args()


models = {"iid" : IIDModel, "tf-crf" : TFCRF}



  
def _run_test(loss, session, handle):
  ''' Compute test loss '''
  losses = []
  try:
    while True:
      losses += [session.run(loss, feed_dict=handle)] 
  except OutOfRangeError:
    pass
  
  print("test loss is {:10.2f}".format(np.mean(losses)))
  
  
  
  
def main():
  tf.set_random_seed(flags.model_seed)
  # These two are not used directly by us but internally, tensorflow might
  np.random.seed(1001) 
  random.seed(1001)
  #########
  # Data  
  # The set for train and test must be different
  train_set = get_toy_dataset(flags.vocab_size, flags.max_sequence_length, seed=flags.train_seed).take(flags.num_train_samples).batch(flags.batch_size).shuffle(buffer_size=1000, seed=flags.train_seed + 1).repeat(flags.num_epochs).prefetch(10).make_initializable_iterator()
  test_set = get_toy_dataset(flags.vocab_size, flags.max_sequence_length, n_samples=flags.num_test_samples, seed=flags.test_seed + 1).batch(flags.batch_size).prefetch(10).make_initializable_iterator()

  # Joint iterator
  handle = tf.placeholder(tf.string, shape=[])
  data_iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)
  seqs = data_iterator.get_next()
  lengths = tf.fill([tf.shape(seqs)[0]], tf.shape(seqs)[1])
  
  #########
  # Model
  # Constant states
  # bs x L x state_size
  states = get_positional_states(seqs, flags.state_size, flags.vocab_size, flags.embedding_method)


  # Our embedding-based CRF
  if flags.model == "emb-crf":
    crf = ContextualEmbCRF(seqs, tf.cast(lengths, tf.float32), states, flags.crf_d, flags.crf_emb_size, flags.vocab_size, 
                 flags.nr_s_layers, flags.crf_transitions, flags.max_sequence_length, flags.S_fn, crf_S_diag=flags.crf_S_diag)
    ll = crf.ll
    
    if flags.crf_pretrained_embedding and flags.crf_transitions == "emb-contextual":
      id2tok = {i : str(i) for i in range(flags.vocab_size)}
      preload_factors_X, preload_factors_Y = load_asymmetric_glove_embedding(flags.vocab_size, flags.crf_emb_size, id2tok, (crf.X_T, crf.Y), flags.crf_pretrained_embedding, [])
  # Some baseline model
  else:
    Model = models[flags.model]
    crf = Model(flags.vocab_size, flags.max_sequence_length)
    crf.build(flags.state_size)
    ll = crf(seqs, lengths, states)
   
      
  # Use batch average as loss
  loss = -tf.reduce_mean(ll, axis=0)
  train_op, global_step = add_optimizer_graph(loss, flags.initial_learning_rate, flags.max_grad_norm)

  # Summaries
  summaries = [tf.summary.scalar("train_loss", loss)] + crf.summaries
  summary_writer = tf.summary.FileWriter(flags.model_dir, max_queue = 0, flush_secs = 2)
  summary_op = tf.summary.merge(summaries)
  

  with tf.Session() as session:
    # Initialization a nd loading of external embeddings
    session.run([tf.global_variables_initializer(), train_set.initializer])
    if flags.crf_pretrained_embedding and flags.model == "emb-crf" and flags.crf_transitions == "emb-contextual":
      session.run(preload_factors_X[0], preload_factors_X[1])
      session.run(preload_factors_Y[0], preload_factors_Y[1])

        
    handles = {"train" : {handle : session.run(train_set.string_handle())},
               "test"  : {handle : session.run(test_set.string_handle())}}

    try:
      # Training loop
      while True:
        # Run
        _, step, batch_loss = session.run([train_op, global_step, loss], feed_dict=handles["train"])
        # Print, summaries, test, sampling
        if step % flags.print_every == 1:
          print("step {} loss {:10.2f}".format(step, batch_loss))
        if step % flags.test_every == 1:
          session.run(test_set.initializer)
          _run_test(loss, session, handles["test"])
        if step % flags.summary_every == 0:
          summary_str = session.run(summary_op, feed_dict=handles["train"])
          summary_writer.add_summary(summary_str, step)
        if flags.model == "emb-crf" and flags.crf_transitions == "emb-contextual" and step % flags.sample_every == 0:
          sampled_seqs = session.run(crf.sampled_seqs, feed_dict=handles["train"])
          for seq in sampled_seqs[:4]:
            print(" ".join([str(idx) for idx in seq]))
          
    except OutOfRangeError:
      print("Training finished")


if __name__ == "__main__":   
  main()
  
  
  
  
  