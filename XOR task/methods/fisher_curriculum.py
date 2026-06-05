import torch
import datetime
import copy
import numpy as np
from math import ceil, floor

from functions import set_seed, NN, model_definition, train_one_epoch, validate_model, get_NN_input

def pruning_curriculum_fisher(X_train, t_train, X_val, t_val, data_loss,
                              F_i, SEED, learning_percentage, NN_config, n_prints, num_bins=200, epochs=10000):
    
    n_input = NN_config["n_input"]
    n_output = NN_config["n_output"]
    hidden_layers_size = NN_config["hidden_layers_size"]
    activation_fn = NN_config["activation_functions"][1]
    eta = NN_config["eta"]
    eta_min = NN_config["eta_min"]
    Weight_decay = NN_config["Weight_decay"]

    # Set the seed for reproducibility
    set_seed(SEED)

    # Setup model for curriculum learning
    model_curr = model_definition(NN, n_input, n_output, hidden_layers_size, activation_fn, p=0.0)
    opt_curr = torch.optim.AdamW(model_curr.parameters(), lr=eta, weight_decay=Weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_curr, T_max=epochs, eta_min=eta_min
    )

    Ntr = len(X_train)
    all_indices = torch.arange(Ntr)

    # Sort by actual Fisher score: low Fisher first = "easy" first, but the random subset is kept
    # sorted_indices = all_indices[torch.argsort(F_i[random_indices], descending=False)]

    # stage_epochs = 50
    # n_bins = epochs // stage_epochs
    n_bins = num_bins
    # stage_epochs = epochs // n_bins
    stage_epochs = (epochs//2) // n_bins

    # print(f'n_bins: {n_bins}')
    bin_size = Ntr // n_bins

    sorted, indices = torch.sort(F_i)
    fi_bins = torch.tensor_split(indices, n_bins)

    num_samples_for_percentage = Ntr * learning_percentage
    num_samples_per_bin = num_samples_for_percentage / n_bins

    n_lower = num_samples_per_bin % 1 * num_bins
    n_higher = num_bins - n_lower
    
    fi_bins_pruned = [] # List to hold the pruned bins for the current learning percentage

    for i, bin in enumerate(fi_bins):
        if i < n_higher:
            num_samples_to_keep = floor(len(bin) * learning_percentage)
            fi_bins_pruned.append(bin[-num_samples_to_keep:])
        else:
            num_samples_to_keep = ceil(len(bin) * learning_percentage)
            fi_bins_pruned.append(bin[-num_samples_to_keep:])


    bin_indices = torch.cat(fi_bins_pruned, dim=0)
    
    # Sort the indices of the pruned bins to get the order in which to add samples to the curriculum
    X_curr = X_train[bin_indices]
    t_curr = t_train[bin_indices]

    # Split into bins
    # stage_epochs = 50
    validation_curve_curr = []
    train_curve_curr = []
    min_val_loss = float('inf')
    best_model = None

    start = datetime.datetime.now()
    epoch = 1

    for i in range(n_bins):
        # included_samples = bin_indices[:(i+1)*bin_size]
        included_samples = torch.cat(fi_bins_pruned[:i+1], dim=0)
        X_curr_stage = X_train[included_samples]
        t_curr_stage = t_train[included_samples]


        for _ in range(stage_epochs):
            train_loss = train_one_epoch(model_curr, opt_curr, X_curr_stage, t_curr_stage, data_loss)
            
            train_curve_curr.append(train_loss.item())

            validation_loss = validate_model(model_curr, X_val, t_val, data_loss)
            validation_curve_curr.append(validation_loss.item())

            scheduler.step()

            if validation_curve_curr[-1] < min_val_loss:
                min_val_loss = validation_curve_curr[-1]
                best_model = copy.deepcopy(model_curr)
                best_epoch = epoch
            
            epoch += 1

            if epoch % (epochs // n_prints) == 0:
                print(f"Epoch {epoch}/{epochs}, Training Loss: {train_loss.item():.6f}, Validation Loss: {validation_loss.item():.6f}")
    
    for _ in range(epochs//2):
        train_loss = train_one_epoch(model_curr, opt_curr, X_curr, t_curr, data_loss)
        
        train_curve_curr.append(train_loss.item())

        validation_loss = validate_model(model_curr, X_val, t_val, data_loss)
        validation_curve_curr.append(validation_loss.item())

        scheduler.step()

        if validation_curve_curr[-1] < min_val_loss:
            min_val_loss = validation_curve_curr[-1]
            best_model = copy.deepcopy(model_curr)
            best_epoch = epoch
        
        epoch += 1

        if epoch % (epochs // n_prints) == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {train_loss.item():.6f}, Validation Loss: {validation_loss.item():.6f}")

    end = datetime.datetime.now()
    diff_time = (end - start).total_seconds()

    history = np.array(list(zip(range(1, epochs), train_curve_curr, validation_curve_curr)))
    print(f"Chosen model from epoch {best_epoch} with validation loss {min_val_loss:.7f}")

    return history, best_model, diff_time, bin_indices