import tensorflow as tf


class IIDModel(object):
  '''
    A baseline model that uses an iid softmax in every timestep
  '''
  
  def __init__(self, vocab_size, max_sequence_length):
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    self.summaries = []

  def build(self, d):
    self.softmax_w = tf.get_variable("softmax_w",  [self.vocab_size, d], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size], initializer=tf.zeros_initializer(tf.float32))

  
  def __call__(self, seqs, lengths, states):
    '''
      states: bs x L x d
    '''
    batch_size, L, d = tf.shape(states)[0], tf.shape(states)[1], tf.shape(states)[2]
    # bs * L x vocab_size
    logits = tf.matmul(tf.reshape(states, [-1, d]), self.softmax_w, transpose_b=True) + self.softmax_b
    
    # bs x L
    ll = - tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, self.vocab_size]), labels=tf.reshape(seqs, [-1]))
    ll = tf.reshape(ll, [batch_size, -1])    
    # bs 
    ll = tf.reduce_sum(ll, axis=-1)
    
    return ll
