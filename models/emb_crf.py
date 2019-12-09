import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from models.contextual_crf import  sample_from_crf, crf_log_likelihood
from models.transitions import EmbStateTransformation, SingleMatrixTransitions,\
  MultiMatrixTransitions
from models.tf_utils import get_fn


class ContextualEmbCRF(object):
  '''
    An implementation of the contextual embedding CRF of the paper
  '''
  
  def __init__(self, seqs, lengths, states, d, factor_emb_size, vocab_size, nr_s_layers, transitions_type, max_sequence_length, S_fn, crf_S_diag=False):
    '''
      Inputs:
        seqs: bs x L          tensor of integer indices
        lengths: bs           tensor of integer sequence lengths
        states: bs x L x d    tensor of d-dimensional states that the CRF conditions its predictions on
        
      Parameters:  
        factor_emb_size       the size of the embeddings used in the factorization (X and Y in the paper)
        d                     d the size of the diagonal matrix d. This determines the rank of the approximation
                              Note that if `factor_emb_size != d` then we use a FC layer to make the two compatible
        vocab_size            Vocabulary Size
        nr_s_layers           How many FCs to parametrize S?
        transitions_type      Either emb-contextual (used in the paper) or single-matrix or multi-matrix (for comparison)
                              Note that single-matrix is identical to what TFCRF in tf_crf.py does and can be used
                              to verify correctness of the algorithms in contextual_crf (to some degree)
        max_sequence_length   Maximum sequence length necessary for the multi-matrix baseline
        S_fn                  Final activation on the neurons of S: none, relu, exp
        crf_S_diag            Is S a diagonal matrix?
    '''
      
    batch_size, L = tf.shape(states)[0], tf.shape(states)[1]
    state_size = states.get_shape()[-1].value
    self.summaries = []
    assert(nr_s_layers > 0)
    
    ### Pairwise potentials
    if transitions_type == "emb-contextual":
      with tf.variable_scope("embedding"):
        self.X_T = tf.get_variable("X_T", [vocab_size, factor_emb_size], tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
        self.Y = tf.get_variable("Y", [vocab_size, factor_emb_size], tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01)) 
   
        X_T = self.X_T
        Y = self.Y
      
        if factor_emb_size != d:
          X_T = fully_connected(X_T, d, activation_fn=tf.identity) 
          Y = fully_connected(Y, d, activation_fn=tf.identity)
        else:
          X_T = tf.identity(X_T)
          Y = tf.identity(Y)
           
        Y = tf.transpose(Y)
     
      with tf.variable_scope("contextual"):
        # The states to condition on: Use neighbouring states to parametrize a potential
        features_first = tf.concat([tf.zeros([batch_size, 1, state_size]), states[:,:-1,:]] , axis=1)
        features_second = tf.concat([tf.zeros([batch_size, 1, state_size]), states[:,1:-1,:],tf.zeros([batch_size, 1, state_size])] , axis=1)
        prepared_states =  tf.concat([features_first, features_second], axis=2)

        prepared_state_size = prepared_states.get_shape()[-1].value
        with tf.variable_scope("S"):
          if crf_S_diag:
            # S is diagonal
            diagonal = tf.reshape(prepared_states, [-1, prepared_state_size])
            for i in range(nr_s_layers):
              with tf.variable_scope("layer-%d" % i):
                activation_fn = get_fn(S_fn) if i == nr_s_layers -1 else tf.nn.relu
                diagonal = fully_connected(diagonal, d, activation_fn=activation_fn) 
            
            S = tf.matrix_diag(diagonal)
          else:
            # Not diag
            S = tf.reshape(prepared_states, [-1, prepared_state_size])
            for i in range(nr_s_layers):
              with tf.variable_scope("layer-%d" % i):
                activation_fn = get_fn(S_fn) if i == nr_s_layers -1 else tf.nn.relu
                S = fully_connected(tf.reshape(prepared_states, [-1, prepared_state_size]), d * d) 
                
          S = tf.reshape(S, [batch_size, -1, d, d])
      
      # Plot S mean and std to investigate potentials
      mean, var = tf.nn.moments(tf.reshape(S, [-1, d*d]), axes=[1])
      self.summaries += [tf.summary.scalar("S_mean", tf.reduce_mean(mean, axis=0)), tf.summary.scalar("S_std", tf.reduce_mean(tf.sqrt(var), axis=0))]
      
      transitions = EmbStateTransformation(X_T, S, Y)
    elif transitions_type ==  "single-matrix":
      transitions = SingleMatrixTransitions(batch_size, L, vocab_size)
    elif transitions_type == "multi-matrix":
      transitions = MultiMatrixTransitions(batch_size, max_sequence_length, vocab_size)

     
    ### Unaries are standard logits
    self.softmax_w = tf.get_variable("softmax_w",  [vocab_size, state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.zeros_initializer(tf.float32))

    # bs * L x vocab_size
    unary = tf.matmul(tf.reshape(states, [-1, state_size]), self.softmax_w, transpose_b=True) + self.softmax_b
    unary = tf.reshape(unary, [batch_size, -1, vocab_size])

    # Loss
    self.ll =  crf_log_likelihood(unary, seqs, lengths, transitions)
    
    # Sampling
    if transitions_type == "emb-contextual":
      # Sampling only implemented for the method proposed in the paper
      _, self.sampled_seqs = sample_from_crf(unary, lengths, transitions)

    
    
    
      
      
      
      
      
      
      