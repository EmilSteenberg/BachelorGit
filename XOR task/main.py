import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle
import matplotlib.cm as cm
import os
import datetime
from sklearn.cluster import DBSCAN
import copy

# ========== PART 1 ==========
### Every setting and parameter to be used in the experiments is defined in this part.
print("\n\n========== Setting up initial values and parameters for the experiments ==========")

## Initial values to be used in the experiments
n_samples = 500
noise = 0.00
corner_ratios = None                # None = Skewed, "Equal" = Equal distribution among corners

# Neural network architecture
n_input = 2
n_output = 1
hidden_layers_size = [64, 64, 64]
eta = 3e-3 
eta_min = eta*0.01
Weight_decay = 0.05
activation_functions = [nn.ReLU, nn.ReLU]   # Sampling, Training
p = 0.2                                     # Dropout probability for the MCAL method

NN_config = {
    "n_input": n_input,
    "n_output": n_output,
    "hidden_layers_size": hidden_layers_size,
    "activation_functions": activation_functions,
    "eta": eta,
    "eta_min": eta_min,
    "Weight_decay": Weight_decay,
    "p": p
}


# Sampling parameters
budget = [0.05, 0.01]             # Initial budget for the first query (MC), subsequent budget for each query after that
b = 40                           # Percentile  which should come from coreset in the MCpCAL method 
epochs = [20, 10]               # initial training epochs, subsequent training epochs after each query
use_OFE = False                  # Whether to use OFE in MC_URF method or not
data_loss = nn.BCEWithLogitsLoss()         # For the fisher methods
fisher_epochs = 50

learning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
# learning_percentages = [0.5]       # All learning percentages to be used in the experiments. 
SEEDS = [0]             # Seeds for random number generation to ensure reproducibility [0, 1, 2, 3, 4]
# 2, 19, 35, 41, 42


# Which methods to run
Run_Allsamples                  = True
Run_Random                      = True
Run_Fisher_low                  = True
Run_Fisher_high                 = True
Run_Fisher_Random_curriculum    = True
Run_Fisher_curriculum           = True
Run_MCAL                        = True
Run_MC_URF                      = False
Run_MCpCAL                      = True

# Training parameters
train_epochs = 1000              # How many epochs to train final models
n_prints = 2                    # Number of times to print training progress during training of final models.  
num_bins_for_curriculum = 10    # Number of bins to use for the curriculum learning methods



# Saving directiory setup
results_dir = "results"
file_name = "main_XOR"
activation_dir = "_".join([af.__name__.lower() for af in activation_functions])
corner_label = "skewed" if corner_ratios is None else "equal"
test_dir = f"N_{n_samples}_{corner_label}_noise_{noise}"

# Save or show results (For the .py file, we will only save results, not show them)
show_any                = False

save_loss               = True 
show_loss               = False and show_any

save_histories          = True
show_histories          = False and show_any

save_XOR_data_plots     = True

save_accuracy           = True

save_times              = True



# ========== PART 2 ==========
## Importing data
print("\n\n========== Loading and preprocessing data ==========")
from dataPreProcessing import *

data_dict = {}
for s, SEED in enumerate(SEEDS):
    data_dict[SEED] = {}
    set_seed(SEED)
    X_train, t_train, _ = make_XOR_data(n_samples, noise, seed=SEED, corner_ratios=corner_ratios)
    X_val, t_val, _ = make_XOR_data(1000, noise, seed=SEED, corner_ratios="Equal")
    X_test, t_test, _ = make_XOR_data(1000, noise, seed=(SEED+1), corner_ratios="Equal")
    X_test_skewed, t_test_skewed, _ = make_XOR_data(500, noise, seed=(SEED+2), corner_ratios=corner_ratios)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    X_test_skewed = torch.tensor(X_test_skewed, dtype=torch.float32)
    t_test_skewed = torch.tensor(t_test_skewed, dtype=torch.float32)

    data_dict[SEED]["X_train"] = X_train
    data_dict[SEED]["t_train"] = t_train
    data_dict[SEED]["X_val"] = X_val
    data_dict[SEED]["t_val"] = t_val
    data_dict[SEED]["X_test"] = X_test
    data_dict[SEED]["t_test"] = t_test
    data_dict[SEED]["X_test_skewed"] = X_test_skewed
    data_dict[SEED]["t_test_skewed"] = t_test_skewed

    print("Data loaded successfully.")
    # print(f"Shapes of loaded data: X_train_current: {X_train_current.shape}, X_val_current: {X_val_current.shape}, X_test: {X_test.shape}")


    # os.makedirs(f"{results_dir}/{file_name}/{activation_dir}/seed_{SEEDS[0]}", exist_ok=True)
    # plot_XOR_data(X_train, t_train, label="train", save_dir=f"{results_dir}/{file_name}/{activation_dir}/seed_{SEED}")
    # plot_XOR_data(X_val, t_val, label="val", save_dir=f"{results_dir}/{file_name}/{activation_dir}/seed_{SEED}")
    # plot_XOR_data(X_test, t_test, label="test", save_dir=f"{results_dir}/{file_name}/{activation_dir}/seed_{SEED}")


budget = [int(budget[0] * X_train.shape[0]), int(budget[1] * X_train.shape[0])]
b = int(budget[1] // (100/b))

# ========== PART 3 ==========
## Importing methods
from methods.random import *
from methods.fisher_low import *
from methods.fisher_high import *
from methods.mcdropout import *
from methods.mcdropout_CORESET import *
from methods.mcdropout_URFEAL import *
from functions import get_FI

## Sampling using methods and saving indices of selected samples and fisher scores for pruning methods
print("\n\n========== Running sampling methods and saving selected indices and fisher scores ==========")

indices_dict = {}
fisher_scores_dict = {}
time_dict = {}

for s, SEED in enumerate(SEEDS):
    X_train = data_dict[SEED]["X_train"]
    t_train = data_dict[SEED]["t_train"]
    X_val = data_dict[SEED]["X_val"]
    t_val = data_dict[SEED]["t_val"]

    print(f"\n\n===== SEED {SEED} =====")
    set_seed(SEED)
    indices_dict[SEED] = {}
    fisher_scores_dict[SEED] = {}
    time_dict[SEED] = {}
    time_dict[SEED]["Sampling"] = {}

    if Run_Allsamples:
        print("---All samples:---")
        All_time_start = datetime.datetime.now()
        All_samples_Lidx = np.arange(X_train.shape[0])
        All_samples_Lidx_list = [All_samples_Lidx]
        All_time_end = datetime.datetime.now()
        All_time = (All_time_end - All_time_start).total_seconds()

    if Run_Random:
        print("---Random:---")
        Random_Lidx_list_seed, Random_time_dict = run_RandomAL(X_train, learning_percentages, SEED)

    if Run_Fisher_low or Run_Fisher_high or (Run_Fisher_Random_curriculum and Run_Random) or Run_Fisher_curriculum:
        print("---Calculating fisher scores for pruning---")
        F_i, fisher_time = get_FI(X_train, t_train, data_loss, NN_config, SEED, epochs=fisher_epochs)
        fisher_scores_dict[SEED] = F_i.cpu().numpy()
        
        if Run_Fisher_low:
            print("---Fisher low:---")
            Fi_low_list = pruning_fisher_low(F_i, learning_percentages, SEED)

        if Run_Fisher_high:
            print("---Fisher high:---")
            Fi_high_list = pruning_fisher_high(F_i, learning_percentages, SEED)

    if Run_MCAL:
        print("---MC-Dropout:---")
        MCAL_Lidx_list_seed, _ , MCAL_time_dict = run_MCAL(X_train, t_train, X_val, t_val, data_loss, learning_percentages, budget, SEED, NN_config, epochs)

    if Run_MC_URF:
        print("---MC-Dropout + URFEAL:---")
        MC_URF_Lidx_list_seed, _, MC_URF_time_dict = run_MC_URF(X_train, t_train, X_val, t_val, data_loss, learning_percentages, budget, SEED, NN_config, epochs, radius=0.3, beta=5, use_OFE=use_OFE)
    
    if Run_MCpCAL:
        print("---MC-Dropout + Coreset:---")
        MCpCAL_Lidx_list_seed, _, MCpCAL_time_dict = run_MCpCAL(X_train, t_train, X_val, t_val, data_loss, learning_percentages, budget, b, SEED, NN_config, epochs=[20, 10])

    for i, lp in enumerate(learning_percentages):
        indices_dict[SEED][lp] = {}
        time_dict[SEED]["Sampling"][lp] = {}
        
        if Run_Allsamples:
            indices_dict[SEED][lp]["All samples"] = All_samples_Lidx_list[0]
            time_dict[SEED]["Sampling"][lp]["All samples"] = All_time

        if Run_Random:
            indices_dict[SEED][lp]["Random"] = Random_Lidx_list_seed[i]
            time_dict[SEED]["Sampling"][lp]["Random"] = Random_time_dict[i]

        if Run_Fisher_low or Run_Fisher_high or Run_Fisher_Random_curriculum or Run_Fisher_curriculum:
            time_dict[SEED]["Sampling"][lp]["Fisher"] = fisher_time

        if Run_Fisher_low:
            indices_dict[SEED][lp]["Fisher_low"] = Fi_low_list[i]

        if Run_Fisher_high:
            indices_dict[SEED][lp]["Fisher_high"] = Fi_high_list[i]

        if Run_MCAL:
            indices_dict[SEED][lp]["MCAL"] = MCAL_Lidx_list_seed[i]
            time_dict[SEED]["Sampling"][lp]["MCAL"] = MCAL_time_dict[lp]

        if Run_MC_URF:
            indices_dict[SEED][lp]["MC_URF"] = MC_URF_Lidx_list_seed[i]
            time_dict[SEED]["Sampling"][lp]["MC_URF"] = MC_URF_time_dict[lp]

        if Run_MCpCAL:
            indices_dict[SEED][lp]["MCpCAL"] = MCpCAL_Lidx_list_seed[i]
            time_dict[SEED]["Sampling"][lp]["MCpCAL"] = MCpCAL_time_dict[lp]


# ========= PART 4 ==========
## Training final models and saving results

from functions import train_for_N_epochs
from methods.fisher_curriculum import *
from methods.fisher_random_curriculum import *

models_dict = {}
history_dict = {}

print("\n\n========== Training final models for each method and saving results ==========")

for s, SEED in enumerate(SEEDS):
    X_train = data_dict[SEED]["X_train"]
    t_train = data_dict[SEED]["t_train"]
    X_val = data_dict[SEED]["X_val"]
    t_val = data_dict[SEED]["t_val"]

    seed_time_start = datetime.datetime.now()
    print(f"\n\n==================== SEED {SEED} ====================")

    models_dict[SEED] = {}
    history_dict[SEED] = {}
    time_dict[SEED]["Training"] = {}

    if Run_Allsamples:
        print("Training final model for all samples...")
        All_samples_Lidx = np.arange(X_train.shape[0])
        All_samples_history, All_samples_model, All_samples_time = train_for_N_epochs(X_train, t_train, X_val, t_val, All_samples_Lidx, data_loss, NN_config, idx=All_samples_Lidx, epochs=train_epochs, SEED=SEED, n_prints=n_prints)
        print("Done training final model for all samples: time taken = {:.2f} seconds".format(All_samples_time))

    for i, lp in enumerate(learning_percentages):
        print(f"\n---Training final models for L={int(lp*100)}%---")
        models_dict[SEED][lp] = {}
        history_dict[SEED][lp] = {}
        time_dict[SEED]["Training"][lp] = {}

        if Run_Allsamples:
            models_dict[SEED][lp]["All samples"] = All_samples_model
            history_dict[SEED][lp]["All samples"] = All_samples_history
            time_dict[SEED]["Training"][lp]["All samples"] = All_samples_time

        if Run_Random:
            print("Training Random model...")
            Random_history, Random_model, Random_time = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["Random"], data_loss, NN_config, idx=indices_dict[SEED][lp]["Random"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training Random model: time taken = {:.2f} seconds".format(Random_time))
            history_dict[SEED][lp]["Random"] = Random_history
            models_dict[SEED][lp]["Random"] = Random_model
            time_dict[SEED]["Training"][lp]["Random"] = Random_time


        if Run_Fisher_low:
            print("\nTraining Fisher low model...")
            history_dict[SEED][lp]["Fisher_low"], models_dict[SEED][lp]["Fisher_low"], time_dict[SEED]["Training"][lp]["Fisher_low"] = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["Fisher_low"], data_loss, NN_config, idx=indices_dict[SEED][lp]["Fisher_low"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)

            print("Done training Fisher low model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_low"]))

        if Run_Fisher_high:
            print("\nTraining Fisher high model...")
            history_dict[SEED][lp]["Fisher_high"], models_dict[SEED][lp]["Fisher_high"], time_dict[SEED]["Training"][lp]["Fisher_high"] = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["Fisher_high"], data_loss, NN_config, idx=indices_dict[SEED][lp]["Fisher_high"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training Fisher high model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_high"]))

        if Run_Fisher_Random_curriculum and Run_Random:
            print("\nTraining Fisher Random curriculum model...")
            history_dict[SEED][lp]["Fisher_Random_curriculum"], models_dict[SEED][lp]["Fisher_Random_curriculum"], time_dict[SEED]["Training"][lp]["Fisher_Random_curriculum"], indices_dict[SEED][lp]["Fisher_Random_curriculum"] = pruning_curriculum_random(
                                                                                                                                                                                                                            X_train, t_train, X_val, t_val, data_loss,
                                                                                                                                                                                                                            F_i,  
                                                                                                                                                                                                                            indices_dict[SEED][lp]["Random"],
                                                                                                                                                                                                                            SEED,
                                                                                                                                                                                                                            NN_config,
                                                                                                                                                                                                                            n_prints, 
                                                                                                                                                                                                                            num_bins_for_curriculum, 
                                                                                                                                                                                                                            train_epochs) 
            print("Done training Fisher Random curriculum model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_Random_curriculum"]))

        if Run_Fisher_curriculum:
            print("\nTraining Fisher curriculum model...")
            history_dict[SEED][lp]["Fisher_curriculum"], models_dict[SEED][lp]["Fisher_curriculum"], time_dict[SEED]["Training"][lp]["Fisher_curriculum"], indices_dict[SEED][lp]["Fisher_curriculum"] = pruning_curriculum_fisher(
                                                                                                                                                                                                        X_train, t_train, X_val, t_val, data_loss, 
                                                                                                                                                                                                        F_i,  
                                                                                                                                                                                                        SEED,
                                                                                                                                                                                                        lp,
                                                                                                                                                                                                        NN_config,
                                                                                                                                                                                                        n_prints, 
                                                                                                                                                                                                        num_bins_for_curriculum, 
                                                                                                                                                                                                        train_epochs) 
            print("Done training Fisher curriculum model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_curriculum"]))

        if Run_MCAL:
            print("\nTraining MC-Dropout model...")
            history_dict[SEED][lp]["MCAL"], models_dict[SEED][lp]["MCAL"], time_dict[SEED]["Training"][lp]["MCAL"] = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["MCAL"], data_loss, NN_config, idx=indices_dict[SEED][lp]["MCAL"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-Dropout model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MCAL"]))

        if Run_MC_URF:
            print("\nTraining MC-URF model...")
            history_dict[SEED][lp]["MC_URF"], models_dict[SEED][lp]["MC_URF"], time_dict[SEED]["Training"][lp]["MC_URF"] = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["MC_URF"], data_loss, NN_config, idx=indices_dict[SEED][lp]["MC_URF"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-URF model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MC_URF"]))

        if Run_MCpCAL:
            print("\nTraining MC-Dropout + Coreset model...")
            history_dict[SEED][lp]["MCpCAL"], models_dict[SEED][lp]["MCpCAL"], time_dict[SEED]["Training"][lp]["MCpCAL"] = train_for_N_epochs(X_train, t_train, X_val, t_val, indices_dict[SEED][lp]["MCpCAL"], data_loss, NN_config, idx=indices_dict[SEED][lp]["MCpCAL"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-Dropout + Coreset model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MCpCAL"]))
    seed_time_end = datetime.datetime.now()
    total_seed_time = (seed_time_end - seed_time_start).total_seconds()
    print(f"\nTotal time taken for SEED {SEED}: {total_seed_time:.2f} seconds")



# ========== PART 5 ==========
## Saving results and configuration
from test_utils import save_resultData
from plots import save_config

print("\n\n========== Saving results and configuration ==========")

for s, SEED in enumerate(SEEDS):
    seed_dir = f"seed_{SEED}"
    save_dir = os.path.join(results_dir, file_name, activation_dir, seed_dir, test_dir)

    config = {
        "SEED": SEED,
        "n_samples": n_samples,
        "noise": noise,
        "corner_ratios": "Skewed" if corner_ratios is None else corner_ratios,
        "learning_percentages": learning_percentages,
        "eta": eta,
        "eta_min": eta*0.01,
        "Weight_decay": Weight_decay,
        "hidden_layers_size": hidden_layers_size,
        "activation_functions": activation_functions,
        "n_input": n_input,
        "n_output": n_output,
        "p": p,
        "fisher_training_epochs": fisher_epochs,
        "budget": budget,
        "b": b,
        "epochs": epochs,
        "use_OFE": use_OFE,
        "train_epochs": train_epochs,
        "num_bins_for_curriculum": num_bins_for_curriculum,
        "save_dir": save_dir
    }

    os.makedirs(save_dir, exist_ok=True)
    save_config(save_dir, config)

    models = models_dict[SEED]
    histories = history_dict[SEED]
    times = time_dict[SEED]
    indices = indices_dict[SEED]
    fisher_scores = fisher_scores_dict[SEED]

    save_resultData(save_dir, SEED, models, histories, indices, fisher_scores, times, config)



# ========== PART 6 ==========
## Calculating validation losses, 
# one-step prediction errors, 
# plotting training/validation histories, 
# histograms of selected indices, 
# one-step prediction plots and errors, 
# and saving these results.

from plots import save_val_loss, plot_all_histories, plot_XOR_data, save_accuracies, mean_accuracies, save_timings, save_mean_timings

print("\n\n========== Plotting ==========")

accuracies_equal_dict = {}
accuracies_skewed_dict = {}
for s, SEED in enumerate(models_dict.keys()):
    X_train = data_dict[SEED]["X_train"]
    t_train = data_dict[SEED]["t_train"]
    X_val = data_dict[SEED]["X_val"]
    t_val = data_dict[SEED]["t_val"]
    X_test = data_dict[SEED]["X_test"]
    t_test = data_dict[SEED]["t_test"]
    X_test_skewed = data_dict[SEED]["X_test_skewed"]
    t_test_skewed = data_dict[SEED]["t_test_skewed"]

    print(f"\n\n==================== SEED {SEED} ====================")
    seed_dir = f"seed_{SEED}"
    save_dir = os.path.join(results_dir, file_name, activation_dir, seed_dir, test_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving results to: {save_dir}")


    # Validation losses
    if save_loss or show_loss:
        print("---Calculating validation losses for all models---")
        for i, lp in enumerate(models_dict[SEED].keys()):

            clear_file = False
            if i == 0:
                clear_file = True

            val_losses = {}
            for method in models_dict[SEED][lp].keys():
                model = models_dict[SEED][lp][method]
                val_loss = validate_model(model, X_val, t_val, data_loss)
                val_losses[method] = val_loss.item()

            if save_loss:
                save_val_loss(val_losses, save_dir, lp, epochs=train_epochs, NN_config=NN_config, clear_file=clear_file)

            if show_loss:
                print(f"\n---Validation Losses for L={int(lp*100)}%---")
                for method, loss in val_losses.items():
                    print(f"{method} validation loss: {loss:.10f}")

                val_loss_no_all = {method: loss for method, loss in val_losses.items() if method != "All samples"}
                print(f"\nMinimum validation loss: {min(val_loss_no_all.values()):.10f}, for model {min(val_loss_no_all, key=val_loss_no_all.get)}")
        
        print("Validation losses done.")


    # Training and validation histories
    if save_histories or show_histories:
        print("---Plotting training and validation histories---")
        for i, lp in enumerate(history_dict[SEED].keys()):
            
            all_names = [method for method in history_dict[SEED][lp].keys() if history_dict[SEED][lp][method] is not None]
            all_histories = [history_dict[SEED][lp][method] for method in history_dict[SEED][lp].keys() if history_dict[SEED][lp][method] is not None]

            plot_all_histories(all_histories, all_names, save_dir, SEED, lp, save_histories, show_histories)
        print("Histories done")
    
    # XOR data plots
    if save_XOR_data_plots:
        print("---Saving XOR data plots---")
        plot_XOR_data(X_train, t_train, label="train", save_dir=save_dir)
        plot_XOR_data(X_val, t_val, label="val", save_dir=save_dir)
        plot_XOR_data(X_test, t_test, label="test", save_dir=save_dir)
        print("XOR data plots done")

        for i, lp in enumerate(indices_dict[SEED].keys()):
            for method in indices_dict[SEED][lp].keys():
                if method == "All samples" or method == "Fisher_Random_curriculum":
                    continue
                plot_XOR_data(X_train[indices_dict[SEED][lp][method]], t_train[indices_dict[SEED][lp][method]], label=f"{method}_{int(lp*100)}%", save_dir=save_dir)
    
    
    if save_accuracy:
        accuracies_equal_dict[SEED] = {}
        accuracies_skewed_dict[SEED] = {}
        print("---Calculating and saving accuracies for all models---")
        for i, lp in enumerate(models_dict[SEED].keys()):
            clear_file = False
            if i == 0:
                clear_file = True

            models = models_dict[SEED][lp]
            accuracies_equal_dict[SEED][lp] = save_accuracies(models, X_test, t_test, save_dir, lp, NN_config, corner_ratios="equal", clear_file=clear_file)
            accuracies_skewed_dict[SEED][lp] = save_accuracies(models, X_test_skewed, t_test_skewed, save_dir, lp, NN_config, corner_ratios="skewed", clear_file=clear_file)
        
        print("Accuracies done")

if save_accuracy:
    save_dir = os.path.join(results_dir, file_name, activation_dir, test_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("\n---Calculating mean accuracies across seeds and saving to file---")
    mean_accuracies_equal = mean_accuracies(accuracies_equal_dict, save_dir, corner_ratios="equal")
    mean_accuracies_skewed = mean_accuracies(accuracies_skewed_dict, save_dir, corner_ratios="skewed")
    print("Mean accuracies done")

if save_times:
    save_dir = os.path.join(results_dir, file_name, activation_dir, test_dir)
    print("\n---Saving times to file---")
    save_timings(time_dict, save_dir, noise=noise)
    save_mean_timings(time_dict, save_dir, noise=noise)
    print("Times done")

    