
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.tf_utils import reverse_and_repad

'''
  Note: For compatibility with later versions, this file is written with
  the tf-internal style of imports and access
'''
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops

from tensorflow.contrib.crf.python.ops.crf import crf_unary_score # No changes to unaries, so it's ok to you the TFv ersion


'''
  This file implements the contextual pairwise potentials used in the paper.
  Potentials can either be parametrized directly or via the matrix factorization.
  
  NOTE: The implementation relies heavily on the tensorflow CRF implementation at
    tensorflow.contrib.crf.python.ops.crf
  Their implementation however assumes that there is a single pairwise potential
  identical at every time-step.
  
  Key components and considerations:
    - All algorithms rely on a TransitionManager object instead of a parameter matrix
      since an explicit representation of that matrix is of size V*V which we want to
      avoid through matrix factoriation.
      Whenever the tensorflow CRF would reference a `transition_params` tensor of size
      num_tags x num_tags, we use a `transitions` object that implicitly carries contextual
      factorized potentials
    - Forward and backward computations rely on iterated matrix-vector products which 
      need to be carried out in log space. We use `save_matrix_times_vector` for this.
    - Variable sequence length should be treated by using an <EOS> token to end generation.
      The sequence level tensors arguments can then be set to the maximum length
    - The tensorflow CRF implementation uses some special care to handle sequences of 
      length 0 and 1. This implementation does *not* and is untested for these cases.
      
      
  Shape Notation
    - bs = batch_size
    - V = number of discrete realizations (num_tags and vocab_size in the code)
    - L = sequence length
    
  Conventions
    - For a sequence of length N we parametrize N transition matrices but only need N-1.
      Wlog we drop the *first* (sliced out in all algorithms)
      
  Overview:
    - crf_log_likelihood to compute log likelihood
    - crf_sequence_score to compute the unnormalized sequence score
    - sample_from_crf to sample a sequence of symbols from the CRF
    - get_backwards_probabilities nad get_forwards_probabilities to obtain alpha and beta values
'''



       
def crf_binary_score(tag_indices, sequence_lengths, transitions):
  """Computes the binary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transitions: An object implementing transitions
  Returns:
    binary_scores: A [batch_size] vector of binary scores.
  """
  # Get shape information.
  num_transitions = array_ops.shape(tag_indices)[1] - 1

  # Truncate by one on each side of the sequence to get the start and end
  # indices of each transition.
  start_tag_indices = array_ops.slice(tag_indices, [0, 0], [-1, num_transitions])
  end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])

  binary_scores = transitions.get_pairwise_at_indices(start_tag_indices, end_tag_indices)

  masks = array_ops.sequence_mask(sequence_lengths,
                                  maxlen=array_ops.shape(tag_indices)[1],
                                  dtype=dtypes.float32)
  truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
  binary_scores =  math_ops.reduce_sum(binary_scores * truncated_masks, 1)
  return binary_scores





  

class CrfForwardRnnCell(rnn_cell.RNNCell):
  """Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transitions):
    """Initialize the CrfForwardRnnCell.

    Args:
      num_tags
    """
    self._transitions = transitions
   
  @property
  def state_size(self):
    return self._transitions.num_tags 

  @property
  def output_size(self):
    return self._transitions.num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags + num_tags^2] matrix of potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.

    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """

    unary, concatenated_parameters = array_ops.split(inputs, [self._transitions.num_tags, self._transitions._total_nr_parameters], axis=1)
 
    new_alphas = unary + self._transitions.perform_transition(concatenated_parameters, state)
 
 
    return new_alphas, new_alphas
  
  



class CrfBackwardsRnnCell(rnn_cell.RNNCell):
  """Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transitions):
    """Initialize the CrfForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transitions = transitions

  @property
  def state_size(self):
    return self._transitions.num_tags

  @property
  def output_size(self):
    return self._transitions.num_tags

  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.

    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """

    unary, concatenated_parameters = array_ops.split(inputs, [self._transitions.num_tags, self._transitions._total_nr_parameters], axis=1)
    new_betas = self._transitions.perform_transition(concatenated_parameters, unary + state, from_left=False)

    return new_betas, new_betas
  
  




def crf_log_norm(inputs, sequence_lengths, transitions):
  ''' The CRF normalizer 
      Note that this could be either computed using get_forwards_probabilities or
      get_backwards_probabilities. Hence, it can be used to verify correctness to 
      a certain degree.
  '''
  
  _, log_norm =  get_forwards_probabilities(inputs, sequence_lengths, transitions)

  return log_norm
  

  
def get_forwards_probabilities(inputs, sequence_lengths, transitions):
  '''
    CRF forward probabilities and log normalizer
    
    inputs: bs x L x V unaries
    sequence_length: bs
    transitions: An object implementing CRF transitions
    
    returns: bs x L and bs 
  '''

  
  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = array_ops.squeeze(first_input, [1])


  """Forward computation of alpha values."""
  unary = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])


  pairwise = transitions.pack_to_parameter_sequence()
  pairwise = pairwise[:,1:,:]

  rnn_inputs = array_ops.concat([unary, pairwise], axis=2)
    
  # Compute the alpha values in the forward algorithm in order to get the
  # partition function.
  forward_cell = CrfForwardRnnCell(transitions)
  # Sequence length is not allowed to be less than zero.
  sequence_lengths_less_one = math_ops.maximum(
      constant_op.constant(0, dtype=sequence_lengths.dtype),
      sequence_lengths - 1)
  all_alphas, alphas = rnn.dynamic_rnn(
      cell=forward_cell,
      inputs=rnn_inputs,
      sequence_length=sequence_lengths_less_one,
      initial_state=first_input,
      dtype=dtypes.float32)
  
  log_norm = math_ops.reduce_logsumexp(alphas, [1])
  # Mask `log_norm` of the sequences with length <= zero.
  log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                             array_ops.zeros_like(log_norm),
                             log_norm)

      
  return all_alphas, log_norm





  
def get_backwards_probabilities(inputs, sequence_lengths, transitions):
  '''
    CRF backwards probabilities and log normalizer
    
    inputs: bs x L x V unaries
    sequence_length: bs
    transitions: An object implementing CRF transitions
    
    returns: bs x L and bs 
  '''
  
  batch_size = array_ops.shape(inputs)[0]

  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = inputs[:,0,:]

  num_tags = transitions.num_tags

  pairwise = transitions.pack_to_parameter_sequence()
  rest_of_pairwise = pairwise[:,1:,:]
    
  
  rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])
  
  sequence_lengths_minus_one = math_ops.maximum(array_ops.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)
  
  # Compute the alpha values in the forward algorithm in order to get the
  # partition function.
  forward_cell = CrfBackwardsRnnCell(transitions)
  # Sequence length is not allowed to be less than zero.
  #
  
  concatenated_rest_of_input = array_ops.concat([rest_of_input, rest_of_pairwise], axis=2)
  reversed_concatenated_rest_of_input = reverse_and_repad(concatenated_rest_of_input, sequence_lengths_minus_one, 0)
  
  initial_state = array_ops.zeros([batch_size, num_tags], dtype=dtypes.float32)
  
  
  all_betas, betas = rnn.dynamic_rnn(
      cell=forward_cell,
      inputs=reversed_concatenated_rest_of_input,
      sequence_length=sequence_lengths_minus_one,
      initial_state=initial_state,
      dtype=dtypes.float32)
  log_norm = math_ops.reduce_logsumexp(first_input + betas, [1])
  # Mask `log_norm` of the sequences with length <= zero.
  log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0), array_ops.zeros_like(log_norm), log_norm)
  
  all_betas = reverse_and_repad(all_betas, sequence_lengths_minus_one, 0)



  return all_betas, log_norm








class ForwardSamplingCell(rnn_cell.RNNCell):
  '''
    A cell that implements the ancestral sampling discussed in Equations (6) - (7) 
    in the paper.
    
    The recurrent state consists of the following information:
      - The last bs x V beta vector 
      - The bs x 1 last symbol generated
      
    The inputs consist of 
      - The current unaries 
      - The current bs x V beta vector
      - The bs x _total_nr_parameters tensor that encodes the current transitions
      
      
  '''
  def __init__(self, transitions):
    self._transitions = transitions

  @property
  def state_size(self):
    return self._transitions.num_tags + 1

  @property
  def output_size(self):
    return self._transitions.num_tags + 3

  def __call__(self, inputs, state, scope=None):
    batch_size = array_ops.shape(inputs)[0]
    
    ### Unpack state
    # last_betas: bs x V
    # last_index : bs
    last_beta, last_index = array_ops.split(state, [self._transitions.num_tags, 1], axis=1)
    last_index = math_ops.cast(last_index, dtypes.int32)

    ### Unpack inputs
    # unary:   bs x V
    # last_beta bs x V
    shape = [self._transitions.num_tags,
             self._transitions.num_tags,
             self._transitions._total_nr_parameters]
    unary, beta, pairwise_flat = array_ops.split(inputs, shape, axis=1)
    
    ### Construct logits of this timestep according to (6) in the paper
    batch_indices = array_ops.reshape(math_ops.range(batch_size), [-1,1])

    pairwise = self._transitions.get_pairwise_given_start(pairwise_flat, last_index)
    # bs x V
    last_beta = array_ops.gather_nd(last_beta, array_ops.concat([batch_indices, last_index], axis=1))
    last_beta = array_ops.reshape(last_beta, [-1,1]) 
    logits = pairwise + unary + beta - last_beta # NOTE this is only valid for the index we are gathering from
    
    # bs x V
    log_probs = nn_ops.log_softmax(logits)
    # bs x 1
    entropy = - math_ops.reduce_sum(log_probs * math_ops.exp(log_probs), axis=1, keepdims=True)
    
    ### Sample the next symbol
    # bs x 1
    new_indices = random_ops.multinomial(logits, 1)
    new_indices = math_ops.to_int32(new_indices)
    
    ### Gather the logits of the new symbol to return the sequence probability
    gather_indices = array_ops.concat([batch_indices, array_ops.reshape(new_indices, [-1,1])], axis=1)
   
    # bs x 1
    output_logits = array_ops.gather_nd(logits, gather_indices)
    output_logits = array_ops.reshape(output_logits, [-1,1])
    
    ### Pack the new state
    new_state = array_ops.concat([beta, math_ops.to_float(new_indices)], axis=1)
 
    output = array_ops.concat([math_ops.to_float(new_indices), output_logits, entropy, logits], axis=1)
    
    return output, new_state 
  
  

 

def sample_from_crf(inputs, sequence_lengths, transitions):
  '''  
    Implements sampling from the contextual embedding CRF as presented in Equations (6) - (7) in the paper
    
    inputs: bs x V unary features
    sequence_lengths: bs
    transitions: An object to perfrom CRF transitions
  '''
  batch_size = array_ops.shape(inputs)[0]
  
  sequence_lengths_minus_one = math_ops.maximum(array_ops.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)

  num_tags = transitions.num_tags

  # betas: bs x L-1 x V
  # log_norm: bs 
  betas, log_norm = get_backwards_probabilities(inputs, sequence_lengths, transitions)
  # add 1 message for last time-step
  # bs x L x V
  betas = array_ops.concat([betas, array_ops.zeros([batch_size, 1, num_tags])], axis=1)

  # Pairwise
  pairwise = transitions.pack_to_parameter_sequence()
  rest_of_pairwise = pairwise[:,1:,:]
  # bs x V
  log_norm = array_ops.reshape(log_norm, [-1,1])
  initial_logits = inputs[:,0] + betas[:,0] - log_norm


  # bs x 1
  initial_indices = random_ops.multinomial(initial_logits, 1)
  initial_indices = array_ops.reshape(math_ops.to_int32(initial_indices), [-1, 1])
  
  batch_indices = array_ops.reshape(math_ops.range(batch_size), [-1,1])
  gather_indices = array_ops.concat([batch_indices, array_ops.reshape(initial_indices, [-1,1])], axis=1)
 
  # bs x V
  initial_log_probs = nn_ops.log_softmax(initial_logits)
  initial_entropy = - math_ops.reduce_sum(initial_log_probs * math_ops.exp(initial_log_probs), axis=1, keepdims=True)


  # bs x 1
  initial_output_logits = array_ops.gather_nd(initial_logits, gather_indices)
  initial_output_logits = array_ops.reshape(initial_output_logits, [-1,1])
  
  
  sampling_cell = ForwardSamplingCell(transitions)

  # We slice out the first transition parameters because it 
  # describes how one gets to the first state which is not relevant 
  # bs x num_transitions x num_tags*num_tags

  # The last sampling step does not have a beta anymore
  sampling_rnn_input = array_ops.concat([inputs[:,1:],
                                  betas[:,1:],
                                  rest_of_pairwise], axis=2) 
  sampling_rnn_input.set_shape([None, None, 2 * num_tags  + transitions._total_nr_parameters])
   
  sampling_initial_state = array_ops.concat([betas[:,0], math_ops.to_float(initial_indices)], axis=1)
  probs_and_samples, _ = rnn.dynamic_rnn(
        cell=sampling_cell,
        inputs=sampling_rnn_input,
        sequence_length=sequence_lengths_minus_one,
        initial_state=sampling_initial_state,
        dtype=dtypes.float32)
  samples, output_logits, entropy, output_sampling_logits = array_ops.split(probs_and_samples, [1,1,1, num_tags], axis=2)
  samples = array_ops.squeeze(samples, axis=2)
  # Concat with initial step
  samples = array_ops.concat([initial_indices, math_ops.to_int32(samples)], axis=1)
  output_logits = array_ops.squeeze(output_logits, axis=2)
  output_logits = array_ops.concat([initial_output_logits, output_logits], axis=1)
  
  entropy = array_ops.squeeze(entropy, axis=2)
  entropy = array_ops.concat([initial_entropy, entropy], axis=1)
  

    
  return output_logits, samples




  
  
  
  
  
  
  
  
  
  
  
  
################ COPIED OVER ######################
'''
  These functions are identical to the code in tensorflow.contrib.crf.python.ops.crf
  but call our implementation above
'''


def crf_sequence_score(inputs, tag_indices, sequence_lengths, transitions):
  """Computes the unnormalized score for a tag sequence.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the unnormalized score.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    sequence_scores: A [batch_size] vector of unnormalized sequence scores.
  """

  # Compute the scores of the given tag sequence.
  unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
  binary_scores = crf_binary_score(tag_indices, sequence_lengths, transitions)
  sequence_scores = unary_scores + binary_scores
  return sequence_scores



def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transitions):
  """Computes the log-likelihood of tag sequences in a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transitions: An object implementing the transitions
  Returns:
    log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
      each example, given the sequence of tag indices.
    transition_params: A [num_tags, num_tags] transition matrix. This is either
        provided by the caller or created in this function.
  """


  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths, transitions)
  log_norm = crf_log_norm(inputs, sequence_lengths, transitions)

  # Normalize the scores to get the log-likelihood per example.
  log_likelihood = sequence_scores - log_norm
  return log_likelihood



