import torch
import datetime
import copy
import numpy as np

from functions import set_seed, NN, model_definition, train_one_epoch, validate_model, get_NN_input

def pruning_curriculum_random(X_train, t_train, X_val, t_val, data_loss,
                              F_i, random_idx_for_lp, SEED, NN_config, n_prints, num_bins=200, epochs=10000):
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

    # Amount of samples in training set
    Ntr = len(X_train)

    # Select a random subset of the specified percentage of training samples
    random_indices = random_idx_for_lp

    # An array of all indices in the training set
    # all_indices = torch.arange(Ntr)

    # Sort by actual Fisher score: low Fisher first = "easy" first, but the random subset is kept
    sorted_indices = random_indices[torch.argsort(F_i[random_indices], descending=False)]
    X_curr = X_train[sorted_indices]
    t_curr = t_train[sorted_indices]

    # Split into bins
    # stage_epochs = epochs // num_bins
    stage_epochs = (epochs//2) // num_bins
    # stage_epochs = 50
    # num_bins = epochs // stage_epochs

    curriculum_positions = torch.arange(len(sorted_indices))
    bins = torch.chunk(curriculum_positions, num_bins)

    # Train the model on the curriculum subsets and evaluate on the test set, 
    # keeping track of accuracy over epochs
    validation_curve_curr = []
    train_curve_curr = []

    min_val_loss = float('inf')
    best_model = None
    start = datetime.datetime.now()

    epoch = 1
    print_counter = 0
    for stage in range(num_bins):
        # Get the indices for the current stage (cumulative bins up to the current stage)
        current_indices = torch.cat(bins[:stage + 1]).tolist()
        X_curr_stage = X_curr[current_indices]
        t_curr_stage = t_curr[current_indices]

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

    history = np.array(list(zip(range(1, epoch), train_curve_curr, validation_curve_curr)))
    print(f"Chosen model from epoch {best_epoch} with validation loss {min_val_loss:.7f}")

    return history, best_model, diff_time, random_indices