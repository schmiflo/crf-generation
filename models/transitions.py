import tensorflow as tf
import abc

''' Transitions internally used by ContextualEmbCRF '''


class TransitionManager(object):
  __metaclass__ = abc.ABCMeta
  '''
    A TransitionManager implements the functionality to perform efficient log likelihood 
    computation and sampling in a CRF.
    Depending on the parametrization of the pairwise potentials, access has to be 
    implemented differently which we do in appropriate subclasses
  '''

  def __init__(self, num_tags, total_nr_parameters):
    '''
      num_tags refers to V (vocab_size)
      total_nr_parameters is the total number of parameters required to represent the pairwise
      potentials at a single timestep for a single batch entry. This is V*V for a native implementation
      and less for a factorized approach
    '''
    self.num_tags = num_tags
    self._total_nr_parameters = total_nr_parameters
    
  @abc.abstractmethod
  def pack_to_parameter_sequence(self):
    """back to something of shape [bs, L, d] where d depends on the type of transition"""

  @abc.abstractmethod
  def perform_transition(self, transition_parameters, state, from_left=True):
    """ perform a one step iteration of forward or backward computation 
        from_left allows to either perform multiplication from the left or form the right
    
        transition_parameters: bs x _total_nr_parameters
        state: bs x V
        from_left: boolean
        
        returns: bs x V
    """
  
  @abc.abstractmethod
  def get_pairwise_at_indices(self, start, end):
    """ Get the binary transition scores for a given batch of sequences
    
        start: bs x L
        end: bs x L
        
        returns bs x L
        
        Note: In practice L will be sequence length minus one as a sequence 
        with N elements has N-1 transitions
    """

  def save_matrix_times_vector(self, matrix, vector, from_left=True):    
    ''' matrix vector product in log space 
        matrix: bs x d x d
        vector: bs x d
        
        returns: bs x d
    '''
    
    summation_axis, expand_axis = (1, 2) if from_left else (2, 1)
    return self._save_matrix_times_vector(matrix, vector, expand_axis, summation_axis) 
  
  def _save_matrix_times_vector(self, matrix, vector, expand_axis, summation_axis):
    vector = tf.expand_dims(vector, expand_axis)
    log_products = matrix + vector
    return tf.reduce_logsumexp(log_products, [summation_axis])
    
    
class MultiMatrixTransitions(TransitionManager):
  '''
    A CRF transition implementation that uses a different VxV pairwise potential matrix at *every* position.
    Note that the memory cost is L*V*V which is infeasible for anything but a toy language
  '''
  def __init__(self, batch_size, max_sequence_length, vocab_size):
    super(MultiMatrixTransitions, self).__init__(vocab_size, vocab_size * vocab_size )

    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.sequence_length = max_sequence_length
    
    self.pairwise = tf.get_variable("pairwise", [max_sequence_length, self.vocab_size, self.vocab_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


  def pack_to_parameter_sequence(self):
    transition_params = tf.tile(tf.reshape(self.pairwise, [1,self.sequence_length,-1]), [self.batch_size, 1, 1]) 
    transition_params.set_shape([None, None, self._total_nr_parameters])

    return transition_params


  def perform_transition(self, transition_parameters, state, from_left=True):    
    batch_size = tf.shape(state)[0]
    pairwise = tf.reshape(transition_parameters, [batch_size, self.vocab_size, self.vocab_size])  
    transition_scores = self.save_matrix_times_vector(pairwise, state, from_left) 
      
    return transition_scores
  
  def get_pairwise_at_indices(self, start, end):
    assert(len(start.get_shape()) == 2)
    assert(len(end.get_shape()) == 2)
    batch_size, L = tf.shape(start)[0], tf.shape(start)[1]

    all_pairwise = self.pairwise[1:] # We never use the first transition
    # bs x L
    length_indices = tf.tile(tf.reshape(tf.range(L), [1,-1]), [batch_size, 1])
    indices = tf.stack([length_indices, start, end], axis=2)
    indices = tf.reshape(indices, [-1, 3])
      
    pariwise = tf.gather_nd(all_pairwise, indices)
    pairwise = tf.reshape(pariwise, tf.shape(start))
     
    return pairwise
    
    
    

class SingleMatrixTransitions(TransitionManager):
  '''
    A CRF transition implementation that uses a single static V x V matrix of pairwise potentials.
    Note that for a word-level task this will be infeasible memory wise for most vocabularies.
    
    If these transitions are used in ContextualEmbCRF, the model is identical
    to the tensorflow CRF implementation.
    
    This implementation is for testing correctness of the CRF algorithms below against the TF implementation.
    For compliance with EmbStateTransformation, this implementation will use memory-intense parameter
    replications internally. Use the TF CRF implementation if performance is important
  '''
  def __init__(self, batch_size, sequence_length, vocab_size):
    super(SingleMatrixTransitions, self).__init__(vocab_size, vocab_size * vocab_size )

    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.sequence_length = sequence_length
    
    self.pairwise = tf.get_variable("pairwise", [self.vocab_size, self.vocab_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

  def pack_to_parameter_sequence(self):
    # Replicate along time dimension. Note that this costs us space
    # Use the tensorflow CRF implementation instead if performance is required
    transition_params = tf.tile(tf.reshape(self.pairwise, [1,1,-1]), [self.batch_size, self.sequence_length, 1]) 
    transition_params.set_shape([None, None, self._total_nr_parameters])

    return transition_params


  def perform_transition(self, transition_parameters, state, from_left=True):
    batch_size = tf.shape(state)[0]
    pairwise = tf.reshape(transition_parameters, [batch_size, self.vocab_size, self.vocab_size])  
    transition_scores = self.save_matrix_times_vector(pairwise, state, from_left)
      
    return transition_scores
  
  def get_pairwise_at_indices(self, start, end):
    assert(len(start.get_shape()) == 2)
    assert(len(end.get_shape()) == 2)

    # bs * L x 2
    indices = tf.concat([tf.reshape(start, [-1,1]), tf.reshape(end, [-1,1])], axis=1)

    # bs x L
    pairwise = tf.gather_nd(self.pairwise, indices)
    pairwise = tf.reshape(pairwise, tf.shape(start))

    return pairwise
    



class EmbStateTransformation(TransitionManager):
  '''
    The CRF transition implementation presented in the paper that uses the factorization
      
        T = X^t * S * Y
        
    in Equation (9).
  '''
  def __init__(self, X_T, S, Y, lambda_tensor = tf.reshape(tf.constant(1.0), [1,1,1,1])):
    self.num_tags, self.d = Y.get_shape()[-1].value,  S.get_shape()[-1].value
    super(EmbStateTransformation, self).__init__(self.num_tags, self.d * self.d)

    self.lambda_tensor = lambda_tensor
    self._batch_size, self._sequence_length = tf.shape(S)[0], tf.shape(S)[1]
    
    # V x d
    self.X_T = X_T
    # d x V
    self.Y = Y
    # bs x L x d x d
    self.S = S

  def pack_to_parameter_sequence(self):
    transition_params = tf.reshape(self.S, [self._batch_size, self._sequence_length, -1]) 
    transition_params.set_shape([None, None, self._total_nr_parameters])

    return transition_params


  def perform_transition(self, transition_parameters, state, from_left=True):
    batch_size = tf.shape(state)[0]
    
    S = tf.reshape(transition_parameters, [batch_size, self.d, self.d])

    # Send the state vector through the matrices of the factorization
    # Either left to right or right to left.
    transition_scores = state
    if from_left:
      transition_scores = self.save_matrix_times_vector(tf.expand_dims(self.X_T, 0), transition_scores, from_left)
      transition_scores = self.save_matrix_times_vector(S, transition_scores, from_left)
      transition_scores = self.save_matrix_times_vector(tf.expand_dims(self.Y, 0), transition_scores, from_left)
    else:
      transition_scores = self.save_matrix_times_vector(tf.expand_dims(self.Y, 0), transition_scores, from_left)
      transition_scores = self.save_matrix_times_vector(S, transition_scores, from_left)
      transition_scores = self.save_matrix_times_vector(tf.expand_dims(self.X_T, 0), transition_scores, from_left)
      
    return transition_scores
  
  def get_pairwise_at_indices(self, start, end):
    assert(len(start.get_shape()) == 2)
    assert(len(end.get_shape()) == 2)

    batch_size = tf.shape(start)[0]
    L = tf.shape(start)[1]


    # bs * L x d
    start_vectors = tf.gather(self.X_T, tf.reshape(start, [-1]))
    # bs * L x d
    end_vectors = tf.gather(self.Y, tf.reshape(end, [-1]), axis=1)
    end_vectors = tf.transpose(end_vectors)
    
    # S matrices
    S = tf.reshape(self.S[:,1:,:,:], [-1, self.d, self.d])
    
    products = tf.reduce_logsumexp(tf.expand_dims(start_vectors,2) + S + tf.expand_dims(end_vectors, 1), axis=[1,2])
   
    return tf.reshape(products, [batch_size, L])     


  def get_pairwise_given_start(self, flat_pairwise, start):
    ''' Helper for implementing actual sampling '''
    assert(len(start.get_shape()) == 2)

    # bs x V
    start_vectors = tf.one_hot(tf.reshape(start, [-1]), self.num_tags)
    start_vectors = (1 - start_vectors) * -1000
    # bs x V
    emb_pairwise = self.perform_transition(flat_pairwise, start_vectors)
    return emb_pairwise
  
