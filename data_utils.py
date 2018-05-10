import numpy as np

# -------------------------------------------------------------#
# Load data and proprocessing
# -------------------------------------------------------------#
data_URL = 'Dataset/shakespeare_train.txt'
with open(data_URL, 'r') as f:
    text = f.read()

# Characters' collection
vocab = set(text)

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [# of characters]
train_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
# -------------------------------------------------------------#



# -------------------------------------------------------------#
# Divide data into mini-batches
# -------------------------------------------------------------#
def get_batches(arr, n_seqs, n_steps):
    
    '''
    arr: data to be divided
    n_seqs: batch-size, # of input sequences
    n_steps: timestep, # of characters in a input sequences
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

# Function above define a generator, call next() to get one mini-batch
batch_size	= 10
num_steps	= 50
train_batches = get_batches(train_encode, batch_size, num_steps)
x, y = next(train_batches)

# Traverse whole batches
for x, y in get_batches(train_encode, batch_size, num_steps):
	'''
	training
	'''
# -------------------------------------------------------------#
