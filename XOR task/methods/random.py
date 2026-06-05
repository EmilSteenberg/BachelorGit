import numpy as np
from functions import set_seed

def run_RandomAL(x_train_tensor, L, SEED):
    # Randomly select L% of the training data to be labeled
    N = x_train_tensor.shape[0]
    labeled_indices = [0] * len(L)  # Initialize a list to store labeled indices for each percentage
    for i, l in enumerate(L):
        set_seed(SEED)
        N_L = int(l * N)
        all_indices = np.arange(N)
        np.random.shuffle(all_indices)
        labeled_indices[i] = all_indices[:N_L]
    
    print("Random done!")
    return labeled_indices  # List of len L, where each element is an array of labeled indices for that percentage