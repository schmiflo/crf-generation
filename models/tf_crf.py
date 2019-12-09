import tensorflow as tf
from tensorflow.contrib.crf.python.ops.crf import crf_log_likelihood




class TFCRF(object):
  '''
    A CRF using the tensorflow API. Uses a single VxV pairwise potentials matrix
  '''
  def __init__(self, vocab_size, max_sequence_length):
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    self.summaries = []

  def build(self, d):
    self.pairwise = tf.get_variable("pairwise", [self.vocab_size, self.vocab_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


    self.softmax_w = tf.get_variable("softmax_w",  [self.vocab_size, d], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size], initializer=tf.zeros_initializer(tf.float32))

  

  def __call__(self, seqs, lengths, states):
    '''
      states: bs x L x d
    '''
    batch_size, L, d = tf.shape(states)[0], tf.shape(states)[1], tf.shape(states)[2]
    unary = tf.matmul(tf.reshape(states, [-1, d]), self.softmax_w, transpose_b=True) + self.softmax_b

    unary = tf.reshape(unary, [batch_size, L, self.vocab_size])
    unary.set_shape([None, self.max_sequence_length, self.vocab_size]) # TF akwardness
 
    
    ll, _ = crf_log_likelihood(unary, seqs, lengths, self.pairwise)

    return ll
