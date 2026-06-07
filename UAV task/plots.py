import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import matplotlib.cm as cm
import torch.nn as nn

from functions import get_NN_input
from dataPreProcessing import loadSeedData

## Plots
### Plot history


# ============================================================================ #
# plot_all_histories
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
# Labeled Histograms (Count)

feature_names = ["sin(roll)", "cos(roll)", "sin(pitch)", "cos(pitch)", "sin(yaw)", "cos(yaw)",
                 "vx_norm", "vy_norm", "vz_norm", "w_roll_norm", "w_pitch_norm", "w_yaw_norm",
                 "thrust_norm", "torque_roll_norm", "torque_pitch_norm", "torque_yaw_norm"]

def plot_labeled_histograms(labeled_indices, 
                            X_train_current_norm, U_train_curr_norm, X_train_current, 
                            label, save_dir, learning_percentage, feature_names=None, 
                            bins=40, save=False, show_plot=False):
    """
    Plot histograms of labeled samples across all feature dimensions,
    with the full training set shown in the background.
    """

    X = get_NN_input(X_train_current_norm, U_train_curr_norm, X_train_current) # Get the NN input features for all training samples (not just the labeled ones)

    labeled_indices = np.asarray(labeled_indices, dtype=int)
    labeled_indices = labeled_indices[(labeled_indices >= 0) & (labeled_indices < X.shape[0])]
    X_l = X[labeled_indices]

    n_features = X.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_features)]

    for i in range(n_features):
        # Background: all training samples
        axes[i].hist(
            X[:, i],
            bins=bins,
            alpha=0.50,
            color="gray",
            edgecolor="none",
            label="All samples" if i == 0 else None
        )

        # Foreground: labeled samples
        axes[i].hist(
            X_l[:, i],
            bins=bins,
            alpha=0.85,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            label="Labeled samples" if i == 0 else None
        )

        if feature_names is not None and i < len(feature_names):
            axes[i].set_title(feature_names[i])
        else:
            axes[i].set_title(f"x[{i}]")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    learning_percentage = int(learning_percentage * 100)
    plt.suptitle(f"Histogram of {learning_percentage}% Labeled Samples for All Input Dimensions ({label})", y=0.99)
    plt.tight_layout()

    if save:
        histogram_dir = "histograms"
        save_dir = os.path.join(save_dir, histogram_dir)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{label}_{learning_percentage}%_labeled_histograms.png"), dpi=300)

    if show_plot:
        plt.show()
    
    plt.close()



# ============================================================================ #
# Set Histograms (Density)

def plot_set_histograms(X_current_norm, U_curr_norm, X_current,
                            X_train_current_norm, U_train_curr_norm, X_train_current, 
                            label, save_dir, feature_names=None, 
                            bins=40, save=True):
    """
    Plot histograms of labeled samples across all feature dimensions,
    with the full training set shown in the background.
    """


    X = get_NN_input(X_train_current_norm, U_train_curr_norm, X_train_current) # Get the NN input features for all training samples (not just the labeled ones)

    X_l = get_NN_input(X_current_norm, U_curr_norm, X_current) # Get the NN input features for the labeled samples

    n_features = X.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_features)]

    for i in range(n_features):
        # Background: all training samples
        axes[i].hist(
            X[:, i],
            bins=bins,
            alpha=0.50,
            color="gray",
            edgecolor="none",
            density=True,
            label="All samples" if i == 0 else None
        )

        # Foreground: labeled samples
        axes[i].hist(
            X_l[:, i],
            bins=bins,
            alpha=0.85,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            density=True,
            label="Labeled samples" if i == 0 else None
        )

        if feature_names is not None and i < len(feature_names):
            axes[i].set_title(feature_names[i])
        else:
            axes[i].set_title(f"x[{i}]")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Histogram of {label} in all Input Dimensions ", y=0.99)
    plt.tight_layout()

    if save:
        histogram_dir = "histograms"
        save_dir = os.path.join(save_dir, histogram_dir)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{label}_histogram.png"), dpi=300)
    
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
# OLD Plot One-Step

# def normalize_test_data(X_test, U_test, mean=None, std=None):
def normalize_test_data(X_test, U_test, mean=None, std=None):
    """
    Normalize NN inputs using provided mean and std.
    """
    # Unpack mean and std
    X_test = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
    U_test = U_test.cpu().numpy() if isinstance(U_test, torch.Tensor) else U_test

    lin_pos_mean, lin_vel_mean, ang_vel_mean, lin_acc_mean, ang_acc_mean, controls_mean = mean[:3], mean[3:6], mean[6:9], mean[9:12], mean[12:15], mean[15:19]
    lin_pos_std, lin_vel_std, ang_vel_std, lin_acc_std, ang_acc_std, controls_std = std[:3], std[3:6], std[6:9], std[9:12], std[12:15], std[15:19]

    # Normalize current states
    linear_pos_curr_norm = (X_test[:, :3] - lin_pos_mean) / lin_pos_std
    linear_vel_curr_norm = (X_test[:, 6:9] - lin_vel_mean) / lin_vel_std
    angular_vel_curr_norm = (X_test[:, 9:12] - ang_vel_mean) / ang_vel_std
    # linear_acc_curr_norm = (X_test[:, 12:15] - lin_acc_mean) / lin_acc_std
    # angular_acc_curr_norm = (X_test[:, 15:18] - ang_acc_mean) / ang_acc_std
    controls_curr_norm = (U_test - controls_mean) / controls_std

    # Reconstruct normalized current state tensor
    X_test_norm = np.hstack((linear_pos_curr_norm, X_test[:, 3:6], linear_vel_curr_norm, angular_vel_curr_norm))
    U_test_norm = controls_curr_norm

    X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)
    U_test_norm = torch.tensor(U_test_norm, dtype=torch.float32)

    return X_test_norm, U_test_norm


def one_step_pred(model, X_test, X_test_next, U_test, dt, data_mean, data_std):  # For dt use dt_mean
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test_next = torch.tensor(X_test_next, dtype=torch.float32)
    U_test = torch.tensor(U_test, dtype=torch.float32)

    # X_test_current = X_test
    # U_test_current = U_test
    
    # X_test_next = X_test[1:]

    X_test_current = X_test
    U_test_current = U_test

    X_test_current_NN, U_test_current_NN = normalize_test_data(X_test_current, U_test_current, mean=data_mean, std=data_std)
    # X_test_current_NN, U_test_current_NN = normalize_test_data(X_test_current, U_test_current) # <-- Emil

    # 1a) Tale the real current states for the integration
    x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = X_test_current[:,0:1], X_test_current[:,1:2], X_test_current[:,2:3], X_test_current[:,3:4], X_test_current[:,4:5], X_test_current[:,5:6], X_test_current[:,6:7], X_test_current[:,7:8], X_test_current[:,8:9], X_test_current[:,9:10], X_test_current[:,10:11], X_test_current[:,11:12] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
    current = X_test_current[:, 0:12] # current state vector (without the time step dimension)

    # 1a) Prepare the states and control inputs necessary for the NN
    _,_,_, _, _, _, vx_n, vy_n, vz_n, w_roll_n, w_pitch_n, w_yaw_n, _, _, _, _, _, _ = X_test_current_NN[:,0:1], X_test_current_NN[:,1:2], X_test_current_NN[:,2:3], X_test_current_NN[:,3:4], X_test_current_NN[:,4:5], X_test_current_NN[:,5:6], X_test_current_NN[:,6:7], X_test_current_NN[:,7:8], X_test_current_NN[:,8:9], X_test_current_NN[:,9:10], X_test_current_NN[:,10:11], X_test_current_NN[:,11:12], X_test_current_NN[:,12:13], X_test_current_NN[:,13:14], X_test_current_NN[:,14:15], X_test_current_NN[:,15:16], X_test_current_NN[:,16:17], X_test_current_NN[:,17:18] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
    sin_roll,cos_roll = torch.sin(roll), torch.cos(roll)
    sin_pitch,cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_yaw,cos_yaw = torch.sin(yaw), torch.cos(yaw)
    thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n = U_test_current_NN[:,0:1], U_test_current_NN[:,1:2], U_test_current_NN[:,2:3], U_test_current_NN[:,3:4] # current control inputs

    # 1b) Prepare NN input tensor
    Z = torch.cat([sin_roll, cos_roll,
                   sin_pitch, cos_pitch,
                   sin_yaw, cos_yaw,
                   vx_n, vy_n, vz_n,
                   w_roll_n, w_pitch_n, w_yaw_n, thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n], dim=1) # dim =1 means concatenate along columns (we do it because we have 2D tensors: n_batch x features)

    model.eval() # Set the model to evaluation mode (disables dropout and other training-specific layers)
    
    # 1c) Forward pass
    X_pred = model(Z)
    
    vx_dot_pred = X_pred[:,0:1] 
    vy_dot_pred = X_pred[:,1:2]
    vz_dot_pred = X_pred[:,2:3]
    w_roll_dot_pred = X_pred[:,3:4]
    w_pitch_dot_pred = X_pred[:,4:5]
    w_yaw_dot_pred = X_pred[:,5:6]

    x_dot_pred = X_pred[:,6:7]
    y_dot_pred = X_pred[:,7:8]
    z_dot_pred = X_pred[:,8:9]
    roll_dot_pred = X_pred[:,9:10]
    pitch_dot_pred = X_pred[:,10:11]
    yaw_dot_pred = X_pred[:,11:12]

    # 1d) Find the predictions of the next positions and velocities integrating the predicted velocity increments
    x_pred = x + x_dot_pred * dt
    y_pred = y + y_dot_pred * dt
    z_pred = z + z_dot_pred * dt

    roll_pred = roll + roll_dot_pred * dt
    pitch_pred = pitch + pitch_dot_pred * dt
    yaw_pred = yaw + yaw_dot_pred * dt

    roll_pred = torch.atan2(torch.sin(roll_pred), torch.cos(roll_pred))
    pitch_pred = torch.atan2(torch.sin(pitch_pred), torch.cos(pitch_pred))
    yaw_pred = torch.atan2(torch.sin(yaw_pred), torch.cos(yaw_pred))

    vx_pred = vx + vx_dot_pred * dt
    vy_pred = vy + vy_dot_pred * dt
    vz_pred = vz + vz_dot_pred * dt
    w_roll_pred = w_roll + w_roll_dot_pred * dt
    w_pitch_pred = w_pitch + w_pitch_dot_pred * dt
    w_yaw_pred = w_yaw + w_yaw_dot_pred * dt

    predictions = torch.cat([x_pred, y_pred, z_pred, roll_pred, pitch_pred, yaw_pred, vx_pred, vy_pred, vz_pred, w_roll_pred, w_pitch_pred, w_yaw_pred], dim=1)


    # 2) take the next true states to compare it with the predicted ones
    x_true_next = X_test_next[:,0:1]
    y_true_next = X_test_next[:,1:2]
    z_true_next = X_test_next[:,2:3]
    roll_true_next = X_test_next[:,3:4]
    pitch_true_next = X_test_next[:,4:5]
    yaw_true_next = X_test_next[:,5:6]
    roll_true_next = torch.atan2(torch.sin(roll_true_next), torch.cos(roll_true_next))
    pitch_true_next = torch.atan2(torch.sin(pitch_true_next), torch.cos(pitch_true_next))
    yaw_true_next = torch.atan2(torch.sin(yaw_true_next), torch.cos(yaw_true_next))
    
    vx_true_next = X_test_next[:,6:7]
    vy_true_next = X_test_next[:,7:8]
    vz_true_next = X_test_next[:,8:9]
    w_roll_true_next = X_test_next[:,9:10]
    w_pitch_true_next = X_test_next[:,10:11]      
    w_yaw_true_next = X_test_next[:,11:12]

    targets = torch.cat([x_true_next, y_true_next, z_true_next, roll_true_next, pitch_true_next, yaw_true_next, vx_true_next, vy_true_next, vz_true_next, w_roll_true_next, w_pitch_true_next, w_yaw_true_next], dim=1) 

    # 3) Compute MSE loss between predicted and true next states
    x_pred = x_pred.detach().numpy()
    y_pred = y_pred.detach().numpy()
    z_pred = z_pred.detach().numpy()
    roll_pred = roll_pred.detach().numpy()
    pitch_pred = pitch_pred.detach().numpy()
    yaw_pred = yaw_pred.detach().numpy()
    vx_pred = vx_pred.detach().numpy()
    vy_pred = vy_pred.detach().numpy()
    vz_pred = vz_pred.detach().numpy()
    w_roll_pred = w_roll_pred.detach().numpy()
    w_pitch_pred = w_pitch_pred.detach().numpy()
    w_yaw_pred = w_yaw_pred.detach().numpy()

    x_true_next = x_true_next.detach().numpy()
    y_true_next = y_true_next.detach().numpy()
    z_true_next = z_true_next.detach().numpy()
    roll_true_next = roll_true_next.detach().numpy()
    pitch_true_next = pitch_true_next.detach().numpy()
    yaw_true_next = yaw_true_next.detach().numpy()
    vx_true_next = vx_true_next.detach().numpy()
    vy_true_next = vy_true_next.detach().numpy()
    vz_true_next = vz_true_next.detach().numpy()
    w_roll_true_next = w_roll_true_next.detach().numpy()
    w_pitch_true_next = w_pitch_true_next.detach().numpy()
    w_yaw_true_next = w_yaw_true_next.detach().numpy()

    x_loss = x_pred - x_true_next
    y_loss = y_pred - y_true_next
    z_loss = z_pred - z_true_next
    roll_loss = roll_pred - roll_true_next
    pitch_loss = pitch_pred - pitch_true_next
    yaw_loss = yaw_pred - yaw_true_next
    vx_loss = vx_pred - vx_true_next
    vy_loss = vy_pred - vy_true_next
    vz_loss = vz_pred - vz_true_next
    w_roll_loss = w_roll_pred - w_roll_true_next
    w_pitch_loss = w_pitch_pred - w_pitch_true_next
    w_yaw_loss = w_yaw_pred - w_yaw_true_next

    error = np.hstack([x_loss, y_loss, z_loss, roll_loss, pitch_loss, yaw_loss, vx_loss, vy_loss, vz_loss, w_roll_loss, w_pitch_loss, w_yaw_loss])
    
    return current, predictions, targets, error


# def plot_one_step(model, X_test, U_test, dt, test_index, learning_percentage, save, show_plot, filename):
def plot_one_step(models, X_test, X_test_next, U_test, dt, data_mean, data_std, save_dir, learning_percentage, save_OSP, show_plot_OSP, save_OSP_error, show_plot_OSP_error):
    OSP_dir = "one_step_plots"
    save_dir = os.path.join(save_dir, OSP_dir)

    valid_models = {name: model for name, model in models.items() if model is not None}
    n_models = len(valid_models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    n_rows_error = 4
    n_cols_error = 3

    title_coords = [
    "x", "y", "z",
    "roll", "pitch", "yaw", 
    "vx", "vy", "vz",
    "w_roll", "w_pitch", "w_yaw"]

    cmap = cm.get_cmap("tab20", len(title_coords))
    # if save_OSP or show_plot_OSP:
    #     fig = plt.figure(figsize=(8 * n_cols, 7 * n_rows), constrained_layout=True)
        
        # colors = {
        #     "Current State": "blue",
        #     "Predicted Next State": "orange",
        #     "True Next State": "green"
        # }

    colors = ["blue", "orange", "green"]

    current_list = [0] * n_models
    predictions_list = [0] * n_models
    targets_list = [0] * n_models

    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')

    for plot_idx, (model_string, model) in enumerate(valid_models.items()):
        if model_string == "Random":
            label = "RAN"
        elif model_string == "All samples":
            label = "ALL"
        else:
            label = model_string

        current, predictions, targets, errors = one_step_pred(model, X_test, X_test_next, U_test, dt, data_mean, data_std)
        current_np = current.cpu().detach().numpy()
        predictions_np = predictions.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()
        
        current_list[plot_idx] = current_np
        predictions_list[plot_idx] = predictions_np
        targets_list[plot_idx] = targets_np

        if save_OSP_error or show_plot_OSP_error:
            fig_error = plt.figure(figsize=(4 * n_cols_error, 3 * n_rows_error), constrained_layout=True)
        
            if learning_percentage == 0.2 or learning_percentage == 0.5:
                if learning_percentage == 0.5 and label == "ALL":
                    continue
                    
                y_max = max(errors[:, i].max() for i in range(errors.shape[1]))
                y_min = min(errors[:, i].min() for i in range(errors.shape[1]))
                y_limit = max(abs(y_max), abs(y_min))

                mean_value = np.abs(errors).mean(axis=0)
                mean_mean_value = mean_value.mean()

                for i in range(n_rows_error * n_cols_error):
                    mean_error = mean_value[i]
                    ax = fig_error.add_subplot(n_rows_error, n_cols_error, i + 1)
                    ax.bar(range(len(errors)), errors[:, i], color=cmap(i), alpha=1)
                    ax.set_xlabel("Sample Index")
                    ax.set_ylabel("Absolute Error")
                    ax.set_title(f"Error for Feature {title_coords[i]}")
                    ax.grid(alpha=0.3)
                    ax.text(0.05, 0.90,
                    f"Mean absolute error: {mean_error:.5f}",
                    transform=ax.transAxes, fontsize=10, color='black', fontweight='bold')
                    ax.set_ylim(-y_limit * 1.1, y_limit * 1.1)  # Set y-axis limit to be slightly above the max error for better visualization
                
                if label != "ALL":
                    fig_error.suptitle(f"Test error for {model_string} for learning percentage {learning_percentage*100:.0f}%, Mean: {mean_mean_value:.5f}", fontsize=14)
                else:
                    fig_error.suptitle(f"Test error  for {model_string}, Mean: {mean_mean_value:.5f}", fontsize=14)
                    

                if save_OSP_error:
                    os.makedirs(save_dir, exist_ok=True)
                    if label != "ALL":
                        plt.savefig(os.path.join(save_dir, f"{label}_{learning_percentage*100:.0f}%OSP_error.png"), dpi=300)

                    else:
                        plt.savefig(os.path.join(save_dir, f"{label}_OSP_error.png"), dpi=300)

                if show_plot_OSP_error:
                    plt.show()

                # plt.show()
                plt.close(fig_error)



        # Look through all the current, predictions and targets to find the min and max values for x, y and z to set the same limits for all plots
        for arr in [current_np, predictions_np, targets_np]:
            min_x = min(min_x, arr[0, 0])
            max_x = max(max_x, arr[0, 0])
            min_y = min(min_y, arr[0, 1])
            max_y = max(max_y, arr[0, 1])
            min_z = min(min_z, arr[0, 2])
            max_z = max(max_z, arr[0, 2])

    if save_OSP or show_plot_OSP:
        fig = plt.figure(figsize=(8 * n_cols, 7 * n_rows), constrained_layout=True)
        for plot_idx, (model_string, model) in enumerate(valid_models.items()):

            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection='3d')
            error = np.linalg.norm(predictions_list[plot_idx][0, 0:3] - targets_list[plot_idx][0, 0:3])

            ax.set_title(model_string, fontweight='bold', y=-0.05)

            ax.scatter3D(current_list[plot_idx][0, 0], current_list[plot_idx][0, 1], current_list[plot_idx][0, 2], label="Current State", color=colors[0], s=100)
            ax.scatter3D(predictions_list[plot_idx][0, 0], predictions_list[plot_idx][0, 1], predictions_list[plot_idx][0, 2], label="Predicted Next State", color=colors[1], s=100)
            ax.scatter3D(targets_list[plot_idx][0, 0], targets_list[plot_idx][0, 1], targets_list[plot_idx][0, 2], label="True Next State", color=colors[2], s=100)

            ax.plot([current_list[plot_idx][0, 0:1], predictions_list[plot_idx][0, 0:1]], [current_list[plot_idx][0, 1:2], predictions_list[plot_idx][0, 1:2]], [current_list[plot_idx][0, 2:3], predictions_list[plot_idx][0, 2:3]], color=colors[1], label="Predicted Next State", marker="o")
            ax.plot([current_list[plot_idx][0, 0:1], targets_list[plot_idx][0, 0:1]], [current_list[plot_idx][0, 1:2], targets_list[plot_idx][0, 1:2]], [current_list[plot_idx][0, 2:3], targets_list[plot_idx][0, 2:3]], color=colors[2], label="True Next State", marker="o")

            ax.text2D(0.05, 0.025,
                    f"Abs final error: {error:.5f} m",
                    transform=ax.transAxes, fontsize=10, color='black', fontweight='bold')
                
            ax.set_xlabel('X Position [m]')
            ax.set_ylabel('Y Position [m]')
            ax.set_zlabel('Z Position [m]')

            ax.set_xlim(min_x - 0.1*abs(max_x - min_x), max_x + 0.1*abs(max_x - min_x))
            ax.set_ylim(min_y - 0.1*abs(max_y - min_y), max_y + 0.1*abs(max_y - min_y))
            ax.set_zlim(min_z - 0.1*abs(max_z - min_z), max_z + 0.1*abs(max_z - min_z))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=10, bbox_to_anchor=(0.5, 0.95), ncol=3)
        fig.legend(handles, labels, loc=10, bbox_to_anchor=(0.5, 0.62), ncol=3)
        fig.legend(handles, labels, loc=10, bbox_to_anchor=(0.5, 0.29), ncol=3)

        fig.suptitle(f"One-Step Prediction in 3D Space at {learning_percentage*100:.0f}% Learning", y=0.99, fontweight='bold')

        if save_OSP:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{learning_percentage*100:.0f}%OSP.png"), dpi=300)
        
        if show_plot_OSP:
            plt.show()
        
        plt.close()



# ============================================================================ #
# NEW OSP plots

def OSP_plot(method, errors, learning_percentage, save_dir):
    OSP_dir = "one_step_plots"
    save_dir = os.path.join(save_dir, OSP_dir)

    n_rows_error = 4
    n_cols_error = 3

    title_coords = [
    "x", "y", "z",
    "roll", "pitch", "yaw", 
    "vx", "vy", "vz",
    "w_roll", "w_pitch", "w_yaw"]

    cmap = cm.get_cmap("tab20", len(title_coords))

    if method == "Random":
        label = "RAN"
    elif method == "All samples":
        label = "ALL"
    else:
        label = method

    fig_error = plt.figure(figsize=(4 * n_cols_error, 3 * n_rows_error), constrained_layout=True)

    y_max = max(errors[:, i].max() for i in range(errors.shape[1]))
    y_min = min(errors[:, i].min() for i in range(errors.shape[1]))
    y_limit = max(abs(y_max), abs(y_min))

    mean_value = errors.mean(axis=0)
    mean_mean_value = mean_value.mean()

    for i in range(n_rows_error * n_cols_error):
        mean_error = mean_value[i]
        ax = fig_error.add_subplot(n_rows_error, n_cols_error, i + 1)
        ax.bar(range(len(errors)), errors[:, i], color=cmap(i), alpha=1)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Error for Feature {title_coords[i]}")
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.90,
        f"Mean absolute error: {mean_error:.5f}",
        transform=ax.transAxes, fontsize=10, color='black', fontweight='bold')
        ax.set_ylim(-y_limit * 0.1, y_limit * 1.1)  # Set y-axis limit to be slightly above the max error for better visualization
    
    if label != "ALL":
        fig_error.suptitle(f"Test error for {label} for learning percentage {learning_percentage*100:.0f}%, Mean: {mean_mean_value:.7f}", fontsize=14)
    else:
        fig_error.suptitle(f"Test error  for {label}, Mean: {mean_mean_value:.5f}", fontsize=14)
        

    os.makedirs(save_dir, exist_ok=True)
    if label != "ALL":
        plt.savefig(os.path.join(save_dir, f"{label}_{learning_percentage*100:.0f}%OSP_error.png"), dpi=300)

    else:
        plt.savefig(os.path.join(save_dir, f"{label}_OSP_error.png"), dpi=300)

    # plt.show()
    plt.close(fig_error)


def plot_all_OSP(models_dict, save_dir):
    SEEDS = [seed for seed in models_dict.keys()]
    method_errors = {}
    method_mean_errors = {}
    for lp in models_dict[SEEDS[0]].keys():
        method_mean_errors[lp] = {}
        method_errors[lp] = {}
        for method in models_dict[SEEDS[0]][lp].keys():
            method_errors[lp][method] = {}
            models = {seed: models_dict[seed][lp][method] for seed in SEEDS if method in models_dict[seed][lp]}
            for seed, model in models.items():
                [_, _, _, _, 
                _, _, _, _, 
                _, _, _, _, 
                _, _, _, _, 
                X_test, X_test_next, U_test, _, 
                dt_mean, data_mean, data_std] = loadSeedData(seed)
                _, _, _, errors = one_step_pred(model, X_test, X_test_next, U_test, dt_mean, data_mean, data_std)
                method_errors[lp][method][seed] = errors
        
        for method in method_errors[lp].keys():
            all_errors = np.array([method_errors[lp][method][seed] for seed in method_errors[lp][method].keys()])
            mean_errors = np.abs(all_errors).mean(axis=0)
            method_mean_errors[lp][method] = mean_errors
    
    for lp, method in method_mean_errors.items():
        for method_name, mean_errors in method.items():
            OSP_plot(method_name, mean_errors, lp, save_dir)




# ============================================================================ #
# Save final OSP error

def save_OSP_error_txt(models, X_test, X_test_next, U_test, dt, data_mean, data_std, save_dir, learning_percentage, NN_config, clear_file, epochs, all=False):
    if clear_file:
        # Create an empty test_loss.txt file (or overwrite if it already exists)
        with open(os.path.join(save_dir, f'OSP_MAE.txt'), 'w') as f:
            f.write('')  # Write an empty string to create/overwrite the file

    with open(os.path.join(save_dir, f'OSP_MAE.txt'), 'a') as f:
        f.write(f'Results for {learning_percentage * 100:.0f}% Learning\n')
        f.write(f'Activation Functions: {", ".join([af.__name__ for af in NN_config["activation_functions"]])}\n')
        f.write(f'Epochs: {epochs}\n\n')

    # for test_index in range(len(X_test)):
    # X_test_single = X_test[test_index]
    # U_test_single = U_test[test_index]

    valid_models = {name: model for name, model in models.items() if model is not None}
    n_models = len(valid_models)

    title_coords = [
    "x", "y", "z",
    "roll", "pitch", "yaw", 
    "vx", "vy", "vz",
    "w_roll", "w_pitch", "w_yaw"]

    with open(os.path.join(save_dir, f'OSP_MAE.txt'), 'a') as f:
        f.write(f"One-Step Prediction MAE for test set:\n")
    mean_mean_errors_dict = {}
    min_mean_mean_error = float('inf')
    for model_string, model in valid_models.items():

        _, _, _, errors = one_step_pred(model, X_test, X_test_next, U_test, dt, data_mean, data_std)

        mean_errors = abs(errors).mean(axis=0)
        mean_mean_error = mean_errors.mean()
        mean_mean_errors_dict[model_string] = mean_mean_error

        if mean_mean_error < min_mean_mean_error and model_string != "All samples":
            min_mean_mean_error = mean_mean_error
            best_model_string = model_string

        with open(os.path.join(save_dir, f'OSP_MAE.txt'), 'a') as f:
            f.write(f"{model_string:<34} - Total MAE: {mean_mean_error:.7f}\n")
            if all:
                for i in range(len(title_coords)):
                    f.write(f"MAE in {title_coords[i]:<10}: {mean_errors[i]:.7f}\n")
    
    with open(os.path.join(save_dir, f'OSP_MAE.txt'), 'a') as f:
        f.write(f"\nBest model for test set: {best_model_string:<17}: {min_mean_mean_error:.7f}\n")
        f.write('\n\n')

    return mean_mean_errors_dict









# ============================================================================ #
# Save config notes

def save_config(save_dir, config):
    with open(os.path.join(save_dir, f'config.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for key, value in config.items():
            f.write(f'{key}: {value}\n')








# ============================================================================ #
# Mean error 

def mean_error(errors_dict, save_dir):
    mean_errors_dict = {}
    SEEDS = [seed for seed in errors_dict.keys()]
    for lp in errors_dict[SEEDS[0]].keys():
        method_errors = {}
        for method in errors_dict[SEEDS[0]][lp].keys():
            errors = [errors_dict[seed][lp][method] for seed in SEEDS if method in errors_dict[seed][lp]]
            mean_error = np.mean(errors)
            method_errors[method] = mean_error

        mean_errors_dict[lp] = method_errors
    
    with open(os.path.join(save_dir, f'mean_errors.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for lp, method_errors in mean_errors_dict.items():
            f.write(f'Learning percentage: {lp*100:.0f}%\n')

            min_error = float('inf')
            best_method = None              

            for method, mean_error in method_errors.items():
                f.write(f'{method:<30} - Mean MAE: {mean_error:.7f}\n')
                if mean_error < min_error and method != "All samples":
                    min_error = mean_error
                    best_method = method

            f.write(f'Best method: {best_method} with MAE: {min_error:.7f}\n')
            f.write('\n')



# ============================================================================ #
# Standard deviation of error
def std_error(errors_dict, save_dir):
    std_errors_dict = {}
    SEEDS = [seed for seed in errors_dict.keys()]
    for lp in errors_dict[SEEDS[0]].keys():
        method_errors = {}
        for method in errors_dict[SEEDS[0]][lp].keys():
            errors = [errors_dict[seed][lp][method] for seed in SEEDS if method in errors_dict[seed][lp]]
            std_error = np.std(errors, ddof=1)
            method_errors[method] = std_error

        std_errors_dict[lp] = method_errors
    
    with open(os.path.join(save_dir, f'std_errors.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for lp, method_errors in std_errors_dict.items():
            f.write(f'Learning percentage: {lp*100:.0f}%\n')

            max_std_error = float('-inf')
            min_std_error = float('inf')
            worst_method = None            
            best_method = None  

            for method, std_error in method_errors.items():
                f.write(f'{method:<30} - Std of Error: {std_error:.10f}\n')
                if std_error > max_std_error and method != "All samples":
                    max_std_error = std_error
                    worst_method = method
                
                if std_error < min_std_error and method != "All samples":
                    min_std_error = std_error
                    best_method = method

            f.write(f'\nWorst method: {worst_method} with Std of Error: {max_std_error:.10f}\n')
            f.write(f'Best method: {best_method} with Std of Error: {min_std_error:.10f}\n')
            f.write('\n\n')


# ============================================================================ #
# Mean validation loss

def mean_val_loss(val_losses_dict, save_dir):
    mean_val_losses_dict = {}
    SEEDS = [seed for seed in val_losses_dict.keys()]
    for lp in val_losses_dict[SEEDS[0]].keys():
        method_val_losses = {}
        for method in val_losses_dict[SEEDS[0]][lp].keys():
            val_losses = [val_losses_dict[seed][lp][method] for seed in SEEDS if method in val_losses_dict[seed][lp]]
            mean_val_loss = np.mean(val_losses)
            method_val_losses[method] = mean_val_loss

        mean_val_losses_dict[lp] = method_val_losses
    
    with open(os.path.join(save_dir, f'mean_val_losses.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for lp, method_val_losses in mean_val_losses_dict.items():
            f.write(f'Learning percentage: {lp*100:.0f}%\n')

            min_val_loss = float('inf')
            best_method = None              

            for method, mean_val_loss in method_val_losses.items():
                f.write(f'{method:<30} - Mean Validation Loss: {mean_val_loss:.10f}\n')
                if mean_val_loss < min_val_loss and method != "All samples":
                    min_val_loss = mean_val_loss
                    best_method = method

            f.write(f'Best method: {best_method} with Mean Validation Loss: {min_val_loss:.10f}\n')
            f.write('\n')

# ============================================================================ #
# Standard deviation of validation loss
def std_val_loss(val_losses_dict, save_dir):
    std_val_losses_dict = {}
    SEEDS = [seed for seed in val_losses_dict.keys()]
    for lp in val_losses_dict[SEEDS[0]].keys():
        method_val_losses = {}
        for method in val_losses_dict[SEEDS[0]][lp].keys():
            val_losses = [val_losses_dict[seed][lp][method] for seed in SEEDS if method in val_losses_dict[seed][lp]]
            std_val_loss = np.std(val_losses, ddof=1)
            method_val_losses[method] = std_val_loss

        std_val_losses_dict[lp] = method_val_losses
    
    with open(os.path.join(save_dir, f'std_val_losses.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for lp, method_val_losses in std_val_losses_dict.items():
            f.write(f'Learning percentage: {lp*100:.0f}%\n')

            max_std_val_loss = float('-inf')
            min_std_val_loss = float('inf')
            worst_method = None            
            best_method = None  

            for method, std_val_loss in method_val_losses.items():
                f.write(f'{method:<30} - Std of Validation Loss: {std_val_loss:.10f}\n')
                if std_val_loss > max_std_val_loss and method != "All samples":
                    max_std_val_loss = std_val_loss
                    worst_method = method
                
                if std_val_loss < min_std_val_loss and method != "All samples":
                    min_std_val_loss = std_val_loss
                    best_method = method

            f.write(f'\nWorst method: {worst_method} with Std of Validation Loss: {max_std_val_loss:.10f}\n')
            f.write(f'Best method: {best_method} with Std of Validation Loss: {min_std_val_loss:.10f}\n')
            f.write('\n\n')


# ============================================================================ #
# Save timings

def save_timings(timings_dict, save_dir):
    # timings_dict is a dictionary of the form {seed: {TrainingOrSampling: {learning_percentage: {method: timing}}}}
    with open(os.path.join(save_dir, f'timings.txt'), 'w') as f:
        f.write('')  # Write an empty string to create/overwrite the file

        for seed, timing_types in timings_dict.items():
            f.write(f'Seed: {seed}\n')
            for timing_type, learning_percentages in timing_types.items():
                f.write(f'  {timing_type}:\n')
                for lp, method_timings in learning_percentages.items():
                    f.write(f'    Learning percentage: {lp*100:.0f}%\n')
                    for method, timing in method_timings.items():
                        f.write(f'      {method:<30} - Timing: {timing:.3f} seconds\n')
            f.write('\n')