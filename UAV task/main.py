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
print("========== Setting up initial values and parameters for the experiments ==========")
## Initial values to be used in the experiments

# Neural network architecture
n_input = 16
n_output = 12
hidden_layers_size = [64, 32, 16]
eta = 3e-3 
eta_min = eta*0.01
Weight_decay = 0.05
activation_functions = [nn.Tanh, nn.Tanh]   # Sampling, Training
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

# Saving directiory setup
results_dir = "results"
file_name = "main_UAV"
activation_dir = "_".join([af.__name__.lower() for af in activation_functions])


# Sampling parameters
# budget = [128, 128]             # Initial budget for the first query (MC), subsequent budget for each query after that
# b = budget[1] // 5              # Number of samples which should come from coreset in the MCpCAL method

budget = [0.05, 0.01]             # Budget as a percentage of the training data size, for the first query (MC), subsequent queries after that
b = 0.2                           # Percentage of samples which should come from coreset in the

epochs = [20, 10]               # initial training epochs, subsequent training epochs after each query
use_OFE = True                  # Whether to use OFE in MC_URF method or not
train_loss_fn = nn.MSELoss()    # For the fisher methods
fisher_epochs = 50                   # Number of epochs to train the fisher model for calculating fisher scores.
URF_radius = 0.3
URF_beta = 5

# learning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
learning_percentages = [0.1, 0.2]       # All learning percentages to be used in the experiments. 
SEEDS = [0, 1]                             # Seeds for random number generation to ensure reproducibility [0, 1, 2, 3, 4]


# Which methods to run
Run_Allsamples                  = True
Run_Random                      = True
Run_Fisher_low                  = True
Run_Fisher_high                 = False
Run_Fisher_Random_curriculum    = False
Run_Fisher_curriculum           = False
Run_MCAL                        = False
Run_MC_URF                      = True
Run_MCpCAL                      = True

# Training parameters
train_epochs = 400              # How many epochs to train final models
n_prints = 2                    # Number of times to print training progress during training of final models.  
num_bins_for_curriculum = 200   # Number of bins to use for the curriculum learning methods


# Save or show results (For the .py file, we will only save results, not show them)
show_any                = False

save_loss               = True 
show_loss               = False and show_any

save_OSP_mean_error     = True

save_histories          = True
show_histories          = False and show_any

save_histograms         = True
show_histograms         = False and show_any

save_OSP                = False       
show_OSP                = False and show_any

save_OSP_error          = True
show_OSP_error          = False and show_any

save_times              = True






# ========== PART 2 ==========
## Importing data
print("========== Loading and preprocessing data ==========")
from dataPreProcessing import *

X_data, U_data, data_mean, data_std, dt_mean = load_XU_data()
# print(f"Data shapes: X_data: {X_data.shape}, U_data: {U_data.shape}, data_mean: {data_mean.shape}, data_std: {data_std.shape}, dt_mean: {dt_mean}")


# Check if file "dataset/configured_data_new.pkl" exists. If it does, load the data from it. 
for s, SEED in enumerate(SEEDS):
    if not os.path.exists(f"dataset/configured_data_new_{SEED}.pkl"):
        print("File not found. Running TrainValTest_split to create the file.")
        TrainValTest_split(X_data, U_data, data_mean, data_std, dt_mean, SEED)

print("Data loaded successfully.")
# print(f"Shapes of loaded data: X_train_current: {X_train_current.shape}, X_val_current: {X_val_current.shape}, X_test: {X_test.shape}")




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
print("========== Running sampling methods and saving selected indices and fisher scores ==========")

indices_dict = {}
fisher_scores_dict = {}

time_dict = {}
for s, SEED in enumerate(SEEDS):
    print(f"\n\n===== SEED {SEED} =====")

    [X_train_current, X_train_next, X_train_current_norm, X_train_next_norm, 
     U_train_curr, U_train_next, U_train_curr_norm, U_train_next_norm, 
     X_val_current, X_val_next, X_val_current_norm, X_val_next_norm, 
     U_val_curr, U_val_next, U_val_curr_norm, U_val_next_norm, 
     X_test, X_test_next, U_test, U_test_next, 
     dt_mean, data_mean, data_std] = loadSeedData(SEED)
    
    budget_samples = [int(budget[0]*X_train_current.shape[0]), int(budget[1]*X_train_current.shape[0])]
    b_samples = int(b*budget_samples[1])

    # print(f"Budget in samples for SEED {SEED}: initial budget = {budget_samples[0]}, subsequent budget = {budget_samples[1]}, b (coreset samples in MCpCAL) = {b_samples}")

    set_seed(SEED)
    indices_dict[SEED] = {}
    fisher_scores_dict[SEED] = {}
    time_dict[SEED] = {}
    time_dict[SEED]["Sampling"] = {}

    if Run_Allsamples:
        print("---All samples:---")
        All_time_start = datetime.datetime.now()
        All_samples_Lidx = np.arange(X_train_current_norm.shape[0])
        All_samples_Lidx_list = [All_samples_Lidx]
        All_time_end = datetime.datetime.now()
        All_time_diff = (All_time_end - All_time_start).total_seconds()


    if Run_Random:
        print("---Random:---")
        Random_Lidx_list_seed, random_time_dict = run_RandomAL(X_train_current_norm, learning_percentages, SEED)


    if Run_Fisher_low or Run_Fisher_high or (Run_Fisher_Random_curriculum and Run_Random) or Run_Fisher_curriculum:
        print("---Calculating fisher scores for pruning---")
        F_i, fisher_time = get_FI(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                    dt_mean, fisher_epochs, NN_config, SEED)
        fisher_scores_dict[SEED] = F_i.cpu().numpy()
        
        if Run_Fisher_low:
            print("---Fisher low:---")
            Fi_low_list = pruning_fisher_low(F_i, learning_percentages, SEED)

        if Run_Fisher_high:
            print("---Fisher high:---")
            Fi_high_list = pruning_fisher_high(F_i, learning_percentages, SEED)

    if Run_MCAL:
        print("---MC-Dropout:---")
        MCAL_Lidx_list_seed, _, MCAL_time_dict = run_MCAL(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, 
                                                    X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr,
                                                    dt_mean, learning_percentages, budget_samples, SEED, NN_config, epochs)

    if Run_MC_URF:
        print("---MC-Dropout + URFEAL:---")
        MC_URF_Lidx_list_seed, _, MC_URF_time_dict = run_MC_URF(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                    X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr,
                                                    dt_mean, learning_percentages, budget_samples, SEED, NN_config, epochs, radius=URF_radius, beta=URF_beta, use_OFE=use_OFE)
    
    if Run_MCpCAL:
        print("---MC-Dropout + Coreset:---")
        MCpCAL_Lidx_list_seed, _, MCpCAL_time_dict = run_MCpCAL(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, 
                                                    X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr,
                                                    dt_mean, learning_percentages, budget_samples, b_samples, SEED, NN_config, epochs)

    for i, lp in enumerate(learning_percentages):
        indices_dict[SEED][lp] = {}
        time_dict[SEED]["Sampling"][lp] = {}
        
        if Run_Allsamples:
            indices_dict[SEED][lp]["All samples"] = All_samples_Lidx_list[0]
            time_dict[SEED]["Sampling"][lp]["All samples"] = All_time_diff

        if Run_Random:
            indices_dict[SEED][lp]["Random"] = Random_Lidx_list_seed[i]
            time_dict[SEED]["Sampling"][lp]["Random"] = random_time_dict[i]

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


# Print all times:
# print("\n\n========== Times taken for sampling methods ==========")
# for s, SEED in enumerate(time_dict.keys()):
#     print(f"\n--- SEED {SEED} ---")
#     for lp in time_dict[SEED]["Sampling"].keys():
#         print(f"\nL = {int(lp*100)}%:")
#         for method in time_dict[SEED]["Sampling"][lp].keys():
#             print(f"{method}: {time_dict[SEED]['Sampling'][lp][method]:.2f} seconds")

# ========= PART 4 ==========
## Training final models and saving results

from functions import train_for_N_epochs
from methods.fisher_curriculum import *
from methods.fisher_random_curriculum import *

models_dict = {}
history_dict = {}

print("========== Training final models for each method and saving results ==========")

for s, SEED in enumerate(SEEDS):
    seed_time_start = datetime.datetime.now()
    print(f"\n\n==================== SEED {SEED} ====================")

    [X_train_current, X_train_next, X_train_current_norm, X_train_next_norm, 
    U_train_curr, U_train_next, U_train_curr_norm, U_train_next_norm, 
    X_val_current, X_val_next, X_val_current_norm, X_val_next_norm, 
    U_val_curr, U_val_next, U_val_curr_norm, U_val_next_norm, 
    X_test, X_test_next, U_test, U_test_next, 
    dt_mean, data_mean, data_std] = loadSeedData(SEED)

    models_dict[SEED] = {}
    history_dict[SEED] = {}
    time_dict[SEED]["Training"] = {}

    if Run_Allsamples:
        print("Training final model for all samples...")
        All_samples_Lidx = np.arange(X_train_current_norm.shape[0])
        All_samples_history, All_samples_model, All_samples_time = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, 
                                                                dt_mean, NN_config, idx=All_samples_Lidx, epochs=train_epochs, SEED=SEED, n_prints=n_prints)
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
            Random_history, Random_model, Random_time = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, 
                                                                dt_mean, NN_config, idx=indices_dict[SEED][lp]["Random"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training Random model: time taken = {:.2f} seconds".format(Random_time))
            history_dict[SEED][lp]["Random"] = Random_history
            models_dict[SEED][lp]["Random"] = Random_model
            time_dict[SEED]["Training"][lp]["Random"] = Random_time


        if Run_Fisher_low:
            print("\nTraining Fisher low model...")
            history_dict[SEED][lp]["Fisher_low"], models_dict[SEED][lp]["Fisher_low"], time_dict[SEED]["Training"][lp]["Fisher_low"] = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, 
                                                                dt_mean, NN_config, idx=indices_dict[SEED][lp]["Fisher_low"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)

            print("Done training Fisher low model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_low"]))

        if Run_Fisher_high:
            print("\nTraining Fisher high model...")
            history_dict[SEED][lp]["Fisher_high"], models_dict[SEED][lp]["Fisher_high"], time_dict[SEED]["Training"][lp]["Fisher_high"] = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, 
                                                                dt_mean, NN_config, idx=indices_dict[SEED][lp]["Fisher_high"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training Fisher high model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["Fisher_high"]))

        if Run_Fisher_Random_curriculum:
            print("\nTraining Fisher Random curriculum model...")
            history_dict[SEED][lp]["Fisher_Random_curriculum"], models_dict[SEED][lp]["Fisher_Random_curriculum"], time_dict[SEED]["Training"][lp]["Fisher_Random_curriculum"], indices_dict[SEED][lp]["Fisher_Random_curriculum"] = pruning_curriculum_random(
                                                                                                                                                                                                                            X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                                                                                                                                                                            X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, dt_mean, 
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
                                                                                                                                                                                                        X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                                                                                                                                                        X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, dt_mean, 
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
            history_dict[SEED][lp]["MCAL"], models_dict[SEED][lp]["MCAL"], time_dict[SEED]["Training"][lp]["MCAL"] = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                            X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, 
                                                            dt_mean, NN_config, idx=indices_dict[SEED][lp]["MCAL"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-Dropout model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MCAL"]))

        if Run_MC_URF:
            print("\nTraining MC-URF model...")
            history_dict[SEED][lp]["MC_URF"], models_dict[SEED][lp]["MC_URF"], time_dict[SEED]["Training"][lp]["MC_URF"] = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                                            X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr,
                                                                            dt_mean, NN_config, idx=indices_dict[SEED][lp]["MC_URF"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-URF model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MC_URF"]))

        if Run_MCpCAL:
            print("\nTraining MC-Dropout + Coreset model...")
            history_dict[SEED][lp]["MCpCAL"], models_dict[SEED][lp]["MCpCAL"], time_dict[SEED]["Training"][lp]["MCpCAL"] = train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                                                            X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr,
                                                            dt_mean, NN_config, idx=indices_dict[SEED][lp]["MCpCAL"], epochs=train_epochs, SEED=SEED, n_prints=n_prints)
            print("Done training MC-Dropout + Coreset model: time taken = {:.2f} seconds".format(time_dict[SEED]["Training"][lp]["MCpCAL"]))
    seed_time_end = datetime.datetime.now()
    total_seed_time = (seed_time_end - seed_time_start).total_seconds()
    print(f"\nTotal time taken for SEED {SEED}: {total_seed_time:.2f} seconds")





# ========== PART 5 ==========
## Saving results and configuration
from test_utils import save_resultData
from plots import save_config

print("========== Saving results and configuration ==========")

for s, SEED in enumerate(SEEDS):
    seed_dir = f"seed_{SEED}"
    save_dir = os.path.join(results_dir, file_name, activation_dir, seed_dir)

    config = {
        "SEED": SEED,
        "learning_percentages": learning_percentages,
        "eta": eta,
        "eta_min": eta_min,
        "Weight_decay": Weight_decay,
        "hidden_layers_size": hidden_layers_size,
        "activation_functions": activation_functions,
        "n_input": n_input,
        "n_output": n_output,
        "p": p,
        "fisher_training_epochs": fisher_epochs,
        "budget": budget,
        "budget_samples": budget_samples,
        "b": b,
        "b_samples": b_samples,
        "epochs": epochs,
        "use_OFE": use_OFE,
        "train_epochs": train_epochs,
        "num_bins_for_curriculum": num_bins_for_curriculum,
        "save_dir": save_dir
    }

    seed_dir = f"seed_{SEED}"
    save_dir = os.path.join(results_dir, file_name, activation_dir, seed_dir)
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

from plots import save_val_loss, save_OSP_error_txt, plot_all_histories, plot_labeled_histograms, normalize_test_data
from plots import plot_one_step, mean_val_loss, mean_error, save_timings, plot_all_OSP, plot_set_histograms

feature_names = ["sin(roll)", "cos(roll)", "sin(pitch)", "cos(pitch)", "sin(yaw)", "cos(yaw)",
                 "vx_norm", "vy_norm", "vz_norm", "w_roll_norm", "w_pitch_norm", "w_yaw_norm",
                 "thrust_norm", "torque_roll_norm", "torque_pitch_norm", "torque_yaw_norm"]

print("========== Plotting ==========")
errors_dict = {}
val_losses_dict = {}

for s, SEED in enumerate(models_dict.keys()):
    [X_train_current, X_train_next, X_train_current_norm, X_train_next_norm, 
    U_train_curr, U_train_next, U_train_curr_norm, U_train_next_norm, 
    X_val_current, X_val_next, X_val_current_norm, X_val_next_norm, 
    U_val_curr, U_val_next, U_val_curr_norm, U_val_next_norm, 
    X_test, X_test_next, U_test, U_test_next, 
    dt_mean, data_mean, data_std] = loadSeedData(SEED)
    
    print(f"\n\n==================== SEED {SEED} ====================")
    seed_dir = f"seed_{SEED}"
    save_dir = os.path.join(results_dir, file_name, activation_dir, seed_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving results to: {save_dir}")

    errors_dict[SEED] = {}
    val_losses_dict[SEED] = {}

    # Validation losses
    if save_loss or show_loss:
        print("---Calculating validation losses for all models---")
        for i, lp in enumerate(models_dict[SEED].keys()):
            val_losses_dict[SEED][lp] = {}

            clear_file = False
            if i == 0:
                clear_file = True

            val_losses = {}
            for method in models_dict[SEED][lp].keys():
                model = models_dict[SEED][lp][method]
                val_loss = validate_model(model, X_val_current_norm, U_val_curr_norm, X_val_current, X_val_next, U_val_curr, dt_mean)
                val_losses[method] = val_loss.item()

            if save_loss:
                save_val_loss(val_losses, save_dir, lp, epochs=train_epochs, NN_config=NN_config, clear_file=clear_file)

            if show_loss:
                print(f"\n---Validation Losses for L={int(lp*100)}%---")
                for method, loss in val_losses.items():
                    print(f"{method} validation loss: {loss:.10f}")

                val_loss_no_all = {method: loss for method, loss in val_losses.items() if method != "All samples"}
                print(f"\nMinimum validation loss: {min(val_loss_no_all.values()):.10f}, for model {min(val_loss_no_all, key=val_loss_no_all.get)}")
            val_losses_dict[SEED][lp] = val_losses

        print("Validation losses done.")


    # One step prediction mean error, all = True for all 12 dimensions
    if save_OSP_mean_error:
        print("---Calculating one-step prediction mean errors---")
        for i, lp in enumerate(models_dict[SEED].keys()):
            errors_dict[SEED][lp] = {}
            clear_file = False
            if i == 0:
                clear_file = True

            models = models_dict[SEED][lp]

            errors_dict[SEED][lp] = save_OSP_error_txt(models, X_test, X_test_next, U_test, dt_mean, data_mean, data_std,
                               save_dir, lp, NN_config, clear_file, train_epochs, all=False)
        print("One-step prediction mean errors done.")
    

    # Training and validation histories
    if save_histories or show_histories:
        print("---Plotting training and validation histories---")
        for i, lp in enumerate(history_dict[SEED].keys()):
            
            all_names = [method for method in history_dict[SEED][lp].keys() if history_dict[SEED][lp][method] is not None]
            all_histories = [history_dict[SEED][lp][method] for method in history_dict[SEED][lp].keys() if history_dict[SEED][lp][method] is not None]

            plot_all_histories(all_histories, all_names, save_dir, SEED, lp, save_histories, show_histories)
        print("Histories done")
    

    # Histograms of the selected indices
    if save_histograms or show_histograms:
        print("---Plotting histograms of selected indices---")
        for i, lp in enumerate(indices_dict[SEED].keys()):
            for method in indices_dict[SEED][lp].keys():
                if method == "All samples" or method == "Fisher_Random_curriculum": # Fisher random curriculum uses the same indices as random for that lp
                    continue
                plot_labeled_histograms(indices_dict[SEED][lp][method], 
                                        X_train_current_norm, U_train_curr_norm, X_train_current,
                                        method, save_dir, lp, feature_names, bins=40, 
                                        save=save_histograms, show_plot=show_histograms) 
        print("Histograms done")
    
    print("---Plotting histograms of the whole training, validation and test sets---")
    plot_set_histograms(X_val_current_norm, U_val_curr_norm, X_val_current,
                        X_train_current_norm, U_train_curr_norm, X_train_current, 
                        "Validation_set", save_dir, feature_names, 
                        bins=40, save=True)
    

    X_test_current_norm, U_test_current_norm = normalize_test_data(X_test, U_test, data_mean, data_std)
    X_test_current = torch.tensor(X_test, dtype=torch.float32)

    plot_set_histograms(X_test_current_norm, U_test_current_norm, X_test_current,
                            X_train_current_norm, U_train_curr_norm, X_train_current, 
                            "Test_set", save_dir, feature_names, 
                            bins=40, save=True)
    

    plot_set_histograms(X_test_current_norm, U_test_current_norm, X_test_current,
                            X_val_current_norm, U_val_curr_norm, X_val_current, 
                            "Test_over_validation_set", save_dir, feature_names, 
                            bins=40, save=True)

    print("Histograms of whole sets done")


    # One step prediction plots and errors
    if save_OSP or show_OSP or save_OSP_error or show_OSP_error:
        print("---Plotting one-step predictions and errors---")
        for i, lp in enumerate(models_dict[SEED].keys()):
            models = models_dict[SEED][lp]
            plot_one_step(models, X_test, X_test_next, U_test, dt_mean, data_mean, data_std,
                          save_dir, lp,
                          save_OSP=save_OSP, show_plot_OSP=show_OSP, 
                          save_OSP_error=save_OSP_error, show_plot_OSP_error=show_OSP_error)
        print("One-step prediction plots and errors done.")
            
if save_loss:
    save_dir = os.path.join(results_dir, file_name, activation_dir)
    print(f"\nValidation losses saved to {save_dir}")
    mean_val_loss(val_losses_dict, save_dir)

if save_OSP_mean_error:
    save_dir = os.path.join(results_dir, file_name, activation_dir)
    print(f"\nOne-step prediction mean errors saved to {save_dir}")
    mean_error(errors_dict, save_dir)

if save_OSP_error:
    save_dir = os.path.join(results_dir, file_name, activation_dir)
    print("---Plotting one-step prediction errors across seeds---")
    plot_all_OSP(models_dict, save_dir)
    print(f"\nOne-step prediction errors saved to {save_dir}")

if save_times:
    save_dir = os.path.join(results_dir, file_name, activation_dir)
    save_timings(time_dict, save_dir)

