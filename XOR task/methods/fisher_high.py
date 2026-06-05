from functions import set_seed

def pruning_fisher_high(F_i, learning_percentages, SEED):
    set_seed(SEED)
    N = F_i.shape[0]
    
    labeled_indices = [0] * len(learning_percentages)  # Initialize a list to store labeled indices for each percentage
    for i, l in enumerate(learning_percentages):
        N_L = int(l * N)
        _, sorted_indices = F_i.sort(descending=True)  # Sort indices by Fisher information (descending)
        labeled_indices[i] = sorted_indices[:N_L]  # Select the indices with the highest Fisher information
    
    print("Fisher high done!")
    return labeled_indices
