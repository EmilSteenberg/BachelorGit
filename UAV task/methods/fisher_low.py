from functions import set_seed

def pruning_fisher_low(F_i, learning_percentages, SEED):
    set_seed(SEED)
    N = F_i.shape[0]
    
    labeled_indices = [0] * len(learning_percentages)  # Initialize a list to store labeled indices for each percentage
    for i, l in enumerate(learning_percentages):
        N_L = int(l * N)
        _, sorted_indices = F_i.sort(descending=False)  # Sort indices by Fisher information in ascending order
        labeled_indices[i] = sorted_indices[:N_L]  # Select the indices with the lowest Fisher information
    
    print("Fisher low done!")
    return labeled_indices
