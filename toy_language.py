import tensorflow as tf
import numpy as np
import itertools

'''
  A toy language consisting of numbers as tokens.
  
  Sequence length and vocabulary size can be choosen and are fixed.
  The first half of the sequence is non-decreasing, the second half is
  non-increasing. Sampling the next symbol consist of two steps
    - with propability p_rep simply repeat the last symbol
    - with probability (1-p_rep) choose a symbol u.a.r. from the allowed ones
'''


def _generate(n_symbols, max_length, seed):
  '''
    Generate a toy sample of length `max_length`  across a vocabulary of size `n_symbpls` 
  '''
  assert(max_length >= 2)
  assert(n_symbols >= max_length) 
  assert(max_length % 2 == 0) , "use an even max_length"
  symbols = list(range(n_symbols))
  V = len(symbols)
  
  random = np.random.RandomState(seed)
    
  def coin(p): 
    return random.rand() < p

  def multinom(p):
    assert(np.isclose(sum(p), 1.0))
    return random.choice(range(len(p)), p=p)

  def gen(l):
    ''' Generate a single increasing sequence '''   
    seq = []
    for i in range(l):
      ''' Repetition probability '''
      if i > 0 and coin(0.5 * (l - i) / l):
        # repeat
        s = seq[-1]
      else:
        # don't repear
        last_symbol = 0 if i == 0 else seq[-1]
        counts_sym = [V - j if j >= last_symbol else 0 for j in symbols]
        counts_sym = np.asarray([-10000.0 if c == 0 else c for c in counts_sym])
        counts_sym = np.exp(0.2 * counts_sym)
        p_sym = counts_sym / sum(counts_sym)
        s = multinom(p_sym)  
        
      seq += [s]
    
    return seq
  
  while True:
    seq_1, seq_2 = gen(max_length // 2), gen(max_length // 2)
    yield seq_1 + seq_2[::-1]
    
   
def get_toy_dataset(n_symbols, max_length, n_samples=None, seed=1001):
  '''
    Generate a toy language dataset of length `n_samples'
    Samples have a length of `max_length`  and use `n_symbols` many different symbols
  '''
  if n_samples is None:
    # Return an 'infinite' dataset
    generator = lambda: _generate(n_symbols, max_length, seed=seed)
  else:
    # Return a dataset of n_datapoints fixed datapoints
    test_data = list(itertools.islice(_generate(n_symbols, max_length, seed=seed), n_samples))
    def generator():
      for s in test_data:
        yield s
        
  return tf.data.Dataset.from_generator(lambda: generator(), (tf.int32), (tf.TensorShape([max_length])))

  


  
  
  