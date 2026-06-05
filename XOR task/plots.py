import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import matplotlib.cm as cm
import torch.nn as nn

from functions import get_NN_input

# ============================================================================ #
# Plot history

def plot_all_histories(histories, labels, save_dir, SEED, learning_percentage, save=False, show_plot=True):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    x_lim_max = max(history[:, 0].max() for history in histories) * 1.02
    x_lim_min = 0 - x_lim_max * 0.02

    y_lim_max = max(history[:, 1:3].max() for history in histories) * 1.1
    y_lim_min = min(history[:, 1:3].min() for history in histories) - y_lim_max * 0.1

    for i in range(2):
        ax[i].set_xlim(x_lim_min, x_lim_max)
        ax[i].set_ylim(y_lim_min, y_lim_max)
        
        for history, label in zip(histories, labels):
            ax[i].plot(history[:, 0], history[:, i+1], label=label)
        L = int(learning_percentage * 100)

        if i == 0:
            ax[i].set_title(f"Training Loss Histories for L={L}%, Seed={SEED}")
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel("Training Loss")
            ax[i].legend()
        else:
            ax[i].set_title(f"Validation Loss Histories for L={L}%, Seed={SEED}")
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel("Validation Loss")
            ax[i].legend()

    if save:
        history_dir = "histories"
        save_dir = os.path.join(save_dir, history_dir)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{L}%_val_loss_history.png"), dpi=300)

    if show_plot:
        plt.show()

    plt.close()


# ============================================================================ #
# Save final validation loss
def save_val_loss(val_losses, save_dir, learning_percentage, epochs, NN_config, clear_file=False):
    activation_functions = NN_config["activation_functions"]
    # Make directory if it doesn't exist /current_dir/results/file_name/seed/
    os.makedirs(save_dir, exist_ok=True)

    if clear_file:
        # Create an empty val_loss.txt file (or overwrite if it already exists)
        with open(os.path.join(save_dir, 'val_loss.txt'), 'w') as f:
            f.write('')  # Write an empty string to create/overwrite the file

    with open(os.path.join(save_dir, 'val_loss.txt'), 'a') as f:
        f.write(f'\nResults for {learning_percentage * 100:.0f}% Learning\n')
        f.write(f'Activation Functions: {", ".join([af.__name__ for af in activation_functions])}\n')
        f.write(f'Epochs: {epochs}\n\nValidation Loss:\n')

    min_val_loss = float('inf')
    min_val_loss_model = None
    for model_string, val_loss in val_losses.items():
        # print(f'{model_string}')
        if val_loss < min_val_loss and model_string != "All samples":
            min_val_loss = val_loss
            min_val_loss_model = model_string
            
        with open(os.path.join(save_dir, 'val_loss.txt'), 'a') as f:
            f.write(f'{model_string:<30}  {val_loss:.10f}\n')
    
    with open(os.path.join(save_dir, 'val_loss.txt'), 'a') as f:
        f.write(f'\nBest Validation Loss: {min_val_loss_model:<15} {min_val_loss:.10f}\n\n')


# ============================================================================ #
# Save config notes

def save_config(save_dir, config):
    with open(os.path.join(save_dir, f'config.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for key, value in config.items():
            f.write(f'{key}: {value}\n')



# ============================================================================ #
# Plot XOR data

def plot_XOR_data(X, t, label, save_dir):
    XOR_dir = "XOR_data_plots"
    dir = os.path.join(save_dir, XOR_dir)
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap='bwr', edgecolors='k')
    plt.title('XOR Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'XOR Data, {label}')
    plt.grid()
    
    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/XOR_{label}.png")
    plt.close()

# ============================================================================ #
# Calculate accuracy
def calculate_accuracy(model, X, t):
    model.eval()
    with torch.no_grad():
        logits = model(X)

        # Make sure logits and t have same shape
        logits = logits.view(-1)
        t = t.view(-1).float()

        preds = (logits >= 0).float()

        correct = (preds == t).sum().item()
        total = t.numel()

    return correct / total

def save_accuracies(models, X_test, t_test, save_dir, learning_percentage, NN_config, corner_ratios, clear_file=False):
    accuracy_label = f"accuracies_{corner_ratios}"

    accuracies = {}
    activation_functions = NN_config["activation_functions"]

    if clear_file:
        # Create an empty val_loss.txt file (or overwrite if it already exists)
        with open(os.path.join(save_dir, f'{accuracy_label}.txt'), 'w') as f:
            f.write('')  # Write an empty string to create/overwrite the file

    with open(os.path.join(save_dir, f'{accuracy_label}.txt'), 'a') as f:
        f.write(f'\nResults for {learning_percentage * 100:.0f}% Learning\n')
        f.write(f'Activation Functions: {", ".join([af.__name__ for af in activation_functions])}\n')

    max_accuracy = 0.0
    max_accuracy_model = None

    for method, model in models.items():
        accuracy = calculate_accuracy(model, X_test, t_test)
        # print(f'{model_string}')
        if accuracy > max_accuracy and method != "All samples":
            max_accuracy = accuracy
            max_accuracy_model = method

        accuracies[method] = accuracy

        with open(os.path.join(save_dir, f'{accuracy_label}.txt'), 'a') as f:
            f.write(f'{method:<30}  {accuracy:.10f}\n')

    with open(os.path.join(save_dir, f'{accuracy_label}.txt'), 'a') as f:
        f.write(f'\nBest Accuracy: {max_accuracy_model:<15} {max_accuracy:.10f}\n\n')
    
    return accuracies

def mean_accuracies(accuracies_dict, save_dir, corner_ratios):
    # For all learning percentages, calculate mean accuracy across seeds for each method and save to file
    mean_accuracies = {}

    SEEDS = [seed for seed in accuracies_dict.keys()]
    for lp in accuracies_dict[SEEDS[0]].keys():
        method_accuracies = {}
        for method in accuracies_dict[SEEDS[0]][lp].keys():
            accuracies = [accuracies_dict[seed][lp][method] for seed in SEEDS]
            mean_accuracy = np.mean(accuracies)
            method_accuracies[method] = mean_accuracy
        mean_accuracies[lp] = method_accuracies


    with open(os.path.join(save_dir, f'mean_accuracies_{corner_ratios}.txt'), 'w') as f:
        f.write(f'Mean Accuracies across seeds for {corner_ratios} corner ratios\n\n')
        for lp, method_accuracies in mean_accuracies.items():
            f.write(f'Learning Percentage: {lp * 100:.0f}%\n')

            max_accuracy = 0
            max_accuracy_method = None

            for method, mean_accuracy in method_accuracies.items():
                if mean_accuracy > max_accuracy and method != "All samples":
                    max_accuracy = mean_accuracy
                    max_accuracy_method = method
                f.write(f'{method:<30}  {mean_accuracy:.10f}\n')
            
            f.write(f'\nBest Mean Accuracy: {max_accuracy_method:<15} {max_accuracy:.10f}\n\n')
    return mean_accuracies