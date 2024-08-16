import numpy as np
np.random.seed(42)
names = np.load('names_aligned.npy')

np.random.shuffle(names)
import pdb;pdb.set_trace()
np.save('shuffled_names.npy', names)
