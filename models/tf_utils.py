import tensorflow as tf
import math
import numpy as np
from gensim import models


def get_positional_states(seqs, state_size, vocab_size, method):
  '''
    Positional encodings of various flavours
    
    seqs: bs x L
    state_size: python integer
    vocab_size: python integer
    method: transformer or simple.
            transformer uses cosine-based encodings
            simple uses randomly initialized traiable embeddings
            one-hot uses a one-hot enconding
            zeros uses simply zeros
    return: bs x L x state_size
  '''
  batch_size, length = tf.shape(seqs)[0], tf.shape(seqs)[1]
  positions = tf.tile(tf.reshape(tf.range(length), [1, -1]), [batch_size, 1])

  with tf.variable_scope("positional"):
    if method == "transformer":
      positional_embeddings = get_position_encoding(length, state_size)
    elif method == "simple":
      positional_embeddings = tf.get_variable("emb", [vocab_size, state_size], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    elif method == "onehot":
      assert(vocab_size <= state_size), "cannot do one-hot with %d dimensions and %d symbols" % (state_size, vocab_size)
      positional_embeddings = tf.one_hot(tf.range(vocab_size), state_size, dtype=tf.float32)
    elif method == "zeros":
      positional_embeddings = tf.zeros([vocab_size, state_size], dtype=tf.float32)
    else:
      raise ValueError("Unknown embedding method {}".format(method))
    
  # bs x L x d (nonautoregressive)
  states = tf.nn.embedding_lookup(positional_embeddings, positions)
  assert(states.get_shape()[-1].value == state_size), "tensorflow's get_position_encoding returned a strange shape"

  return states
  
  
'''
  from https://github.com/tensorflow/models/blob/master/official/transformer/model/model_utils.py 
'''
def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal


def tfprint(tensor, data=None, message="", p=None):
  if data is None:
    if message == "":
      data = tensor
    else:
      data = tf.constant(' ')

      
  if type(data) is not list:
    data = [data]
  if len(data) == 0:
    data = [tf.constant(' ')]
    
  if p is not None:
    r = tf.random_uniform([])
    pred = tf.less(r, p)
  else:
    pred = tf.constant(True)
  
  def do_print(): return tf.Print(tensor, data, message, summarize=10000000)
  def dont_print(): return tf.identity(tensor) 
  return tf.cond(pred, do_print, dont_print)


def reverse_and_repad(tensor, lengths, pad_symbol):
  '''
    tensor: shape at least two
    length: [bs] shaped tensor with length into dimension 1
    
    returns: tensor with axis 1 reversed so that the data is still 'ragged left'
             i.e. padding is on the right
  '''
  assert(len(tensor.get_shape()) >= 2)
  bs = tf.shape(tensor)[0]
  max_length = tf.shape(tensor)[1]
  mask = tf.sequence_mask(lengths, max_length, dtype=tf.int32)
  mask_ext = tf.concat([tf.reverse(mask, [1]), 1 - tf.reverse(mask, [1])], axis=1)
  
  if isinstance(pad_symbol, int):
    tensor_pad = tf.ones_like(tensor) * pad_symbol
  else:
    tensor_pad = tf.tile(pad_symbol, [bs, max_length, 1])
    
  tensor_ext = tf.concat([tf.reverse(tensor, [1]), tensor_pad], axis=1)
  
  # bs x max_length
  indices_ext = tf.reshape(tf.range(bs * 2 * max_length), [bs, 2 * max_length])
 
  indices_flat = tf.boolean_mask(indices_ext, mask_ext)
 
  if len(tensor.get_shape()) == 2:
    new_shape = [-1]
  else:
    new_shape = tf.concat([[-1], tf.shape(tensor_ext)[2:]], axis=0)
                          

  tensor_ext_flat = tf.reshape(tensor_ext, new_shape)
  

  slices = tf.nn.embedding_lookup(tensor_ext_flat, indices_flat)
  
  out = tf.reshape(slices, tf.shape(tensor))

  return out



def load_asymmetric_glove_embedding(vocab_size, factor_emb_size, id2tok, emb, path, special_tokens, name = ""):
  (X_T,Y) = emb
  # Asymmetrically trained Glove vectors so *not* have a right context. That means, contex vectors go into X
  # The context is the second vector of the concatenation
  print("Loading external embeddings from %s" % path)
  vector_model = models.KeyedVectors.load_word2vec_format(path, binary=False)
  external_embedding_X = np.zeros(shape=(vocab_size,  factor_emb_size))
  external_embedding_Y = np.zeros(shape=(vocab_size,  factor_emb_size))

  unk_vector = 0.5 * vector_model["<unk>"][1 + factor_emb_size:-1] + 0.5 * vector_model["<unk>"][:factor_emb_size]
  matches = 0
  for idx, tok in id2tok.items():
      if tok in vector_model.vocab:
          external_embedding_Y[idx] = vector_model[tok][:factor_emb_size]
          external_embedding_X[idx] = vector_model[tok][1 + factor_emb_size:-1]
          matches += 1
      else:
          print("%s not in embedding file" % tok)
          if tok in special_tokens:
            external_embedding_Y[idx] = external_embedding_X[idx] = np.random.uniform(low=-0.005, high=0.005, size=factor_emb_size)
          else:
            external_embedding_Y[idx] = unk_vector
            external_embedding_X[idx] = unk_vector
          
  print("%d words out of %d could be loaded" % (matches, vocab_size))
  
  pretrained_embeddings = tf.placeholder(tf.float32, [None, None])   
  preload_factors_X = X_T.assign(pretrained_embeddings), {pretrained_embeddings: external_embedding_X}
  preload_factors_Y = Y.assign(pretrained_embeddings), {pretrained_embeddings: external_embedding_Y}

  return preload_factors_X, preload_factors_Y




def get_fn(name):
  ''' activation helper for flags '''
  if name == "relu":
    return tf.nn.relu
  elif name == "none":
    return tf.identity
  elif name == "exp":
    return tf.exp
  


def add_optimizer_graph(loss, initial_learning_rate, max_grad_norm):
  '''
    Add an optimizer to a 1D loss tensor
  '''
  # Optimizer
  global_step = tf.get_variable("global_step", [], tf.int32, trainable=False)
  optimizer = tf.train.AdamOptimizer(initial_learning_rate)
  # Grads
  variables = tf.trainable_variables()
  grads = tf.gradients(loss, variables)
  
  for v,g in zip(variables, grads):
    print("{}\t {}, {}".format(v.name, "x".join([str(i) for i in v.shape]), "NOGRAD" if g is None else ""))
          
  # Clip
  grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

  train_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

  return train_op, global_step