import torch
import numpy as np
import torch.nn as nn
import datetime
from sklearn.cluster import DBSCAN
from torch.func import functional_call, grad, vmap


## Set seed function
# Set the seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NN(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers_size, activation_fn=nn.Softplus, p=0.2):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layers_size = hidden_layers_size

        # Build network
        layers = []
        layers.append(nn.Linear(n_input, hidden_layers_size[0]))
        layers.append(activation_fn())
        layers.append(nn.Dropout(p))
        #layers.append(nn.LayerNorm(hidden_layer_size[0]))

        for i in range (len(hidden_layers_size) -1):
            layers.append(nn.Linear(hidden_layers_size[i], hidden_layers_size[i+1]))
            layers.append(activation_fn())
            layers.append(nn.Dropout(p))
            # if not (i+1)%2:
            #     layers.append(nn.LayerNorm(hidden_layer_size[i+1]))

        self.feature_extractor = nn.Sequential(*layers)

        # layers.append(nn.Linear(hidden_layers_size[-1], n_output))
        # self.network = nn.Sequential(*layers)
        self.network = nn.Linear(hidden_layers_size[-1], n_output)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, Z, return_features=False): #Z is the input tensor that contains both the current state and the control input, concatenated together. The forward method processes this input through the network to produce a prediction of the next state. The output of the network is interpreted as a delta (change) from the current state, which is then added to the current state to get the next state prediction.
        """
        Z: Input tensor [batch_size, n_input] = [state, control]
        returns: next_state_prediction [batch_size, n_output]
        """
        features = self.feature_extractor(Z)

        #current_state = Z[:,:self.n_output] # Extract current state from input tensor (first n_output elements of Z calculated in the previous time step)
        delta = self.network(features) # Neural network output
        next_state_prediction = delta # The NN directly predicts the next state, without adding the current state (so it is not a delta prediction but a direct prediction of the next state)
        #next_state_prediction = current_state + delta

        if return_features:
            return next_state_prediction, features

        return next_state_prediction
    
def model_definition(NN, n_input, n_output, hidden_layers_size, activation_fn, p):
    model = NN(n_input=n_input, n_output=n_output, hidden_layers_size=hidden_layers_size, activation_fn=activation_fn, p=p)
    return model
## Get input tensor
def get_NN_input(X_curr_NN, U_curr_NN, X_curr):

    # 1a) Tale the real current states for the integration
    x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = X_curr[:,0:1], X_curr[:,1:2], X_curr[:,2:3], X_curr[:,3:4], X_curr[:,4:5], X_curr[:,5:6], X_curr[:,6:7], X_curr[:,7:8], X_curr[:,8:9], X_curr[:,9:10], X_curr[:,10:11], X_curr[:,11:12] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)

    # 1a) Prepare the states and control inputs necessary for the NN
    _,_,_, _, _, _, vx_n, vy_n, vz_n, w_roll_n, w_pitch_n, w_yaw_n, _, _, _, _, _, _ = X_curr_NN[:,0:1], X_curr_NN[:,1:2], X_curr_NN[:,2:3], X_curr_NN[:,3:4], X_curr_NN[:,4:5], X_curr_NN[:,5:6], X_curr_NN[:,6:7], X_curr_NN[:,7:8], X_curr_NN[:,8:9], X_curr_NN[:,9:10], X_curr_NN[:,10:11], X_curr_NN[:,11:12], X_curr_NN[:,12:13], X_curr_NN[:,13:14], X_curr_NN[:,14:15], X_curr_NN[:,15:16], X_curr_NN[:,16:17], X_curr_NN[:,17:18] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
    sin_roll,cos_roll = torch.sin(roll), torch.cos(roll)
    sin_pitch,cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_yaw,cos_yaw = torch.sin(yaw), torch.cos(yaw)
    thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n = U_curr_NN[:,0:1], U_curr_NN[:,1:2], U_curr_NN[:,2:3], U_curr_NN[:,3:4] # current control inputs

    # 1b) Prepare NN input tensor
    Z = torch.cat([sin_roll, cos_roll,
                   sin_pitch, cos_pitch,
                   sin_yaw, cos_yaw,
                   vx_n, vy_n, vz_n,
                   w_roll_n, w_pitch_n, w_yaw_n, thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n], dim=1) # dim =1 means concatenate along columns (we do it because we have 2D tensors: n_batch x features)
    
    return Z

## Loss function
def data_loss(model, X_curr_NN, U_curr_NN, X_curr, X_next, dt, channel_weights=None):
    _mse_loss = nn.MSELoss()

    # 1a) Tale the real current states for the integration
    x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = X_curr[:,0:1], X_curr[:,1:2], X_curr[:,2:3], X_curr[:,3:4], X_curr[:,4:5], X_curr[:,5:6], X_curr[:,6:7], X_curr[:,7:8], X_curr[:,8:9], X_curr[:,9:10], X_curr[:,10:11], X_curr[:,11:12] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)

    # 1a) Prepare the states and control inputs necessary for the NN
    _,_,_, _, _, _, vx_n, vy_n, vz_n, w_roll_n, w_pitch_n, w_yaw_n, _, _, _, _, _, _ = X_curr_NN[:,0:1], X_curr_NN[:,1:2], X_curr_NN[:,2:3], X_curr_NN[:,3:4], X_curr_NN[:,4:5], X_curr_NN[:,5:6], X_curr_NN[:,6:7], X_curr_NN[:,7:8], X_curr_NN[:,8:9], X_curr_NN[:,9:10], X_curr_NN[:,10:11], X_curr_NN[:,11:12], X_curr_NN[:,12:13], X_curr_NN[:,13:14], X_curr_NN[:,14:15], X_curr_NN[:,15:16], X_curr_NN[:,16:17], X_curr_NN[:,17:18] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
    sin_roll,cos_roll = torch.sin(roll), torch.cos(roll)
    sin_pitch,cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_yaw,cos_yaw = torch.sin(yaw), torch.cos(yaw)
    thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n = U_curr_NN[:,0:1], U_curr_NN[:,1:2], U_curr_NN[:,2:3], U_curr_NN[:,3:4] # current control inputs

    # 1b) Prepare NN input tensor
    Z = torch.cat([sin_roll, cos_roll,
                   sin_pitch, cos_pitch,
                   sin_yaw, cos_yaw,
                   vx_n, vy_n, vz_n,
                   w_roll_n, w_pitch_n, w_yaw_n, thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n], dim=1) # dim =1 means concatenate along columns (we do it because we have 2D tensors: n_batch x features)


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

    # 2) take the next true states to compare it with the predicted ones
    x_true_next = X_next[:,0:1]
    y_true_next = X_next[:,1:2]
    z_true_next = X_next[:,2:3]
    roll_true_next = X_next[:,3:4]
    pitch_true_next = X_next[:,4:5]
    yaw_true_next = X_next[:,5:6]
    roll_true_next = torch.atan2(torch.sin(roll_true_next), torch.cos(roll_true_next))
    pitch_true_next = torch.atan2(torch.sin(pitch_true_next), torch.cos(pitch_true_next))
    yaw_true_next = torch.atan2(torch.sin(yaw_true_next), torch.cos(yaw_true_next))
    
    vx_true_next = X_next[:,6:7]
    vy_true_next = X_next[:,7:8]
    vz_true_next = X_next[:,8:9]
    w_roll_true_next = X_next[:,9:10]
    w_pitch_true_next = X_next[:,10:11]      
    w_yaw_true_next = X_next[:,11:12]

    # 3) Compute MSE loss between predicted and true next states
    x_loss = _mse_loss(x_pred, x_true_next)
    y_loss = _mse_loss(y_pred, y_true_next)
    z_loss = _mse_loss(z_pred, z_true_next)
    roll_loss = _mse_loss(roll_pred, roll_true_next)
    pitch_loss = _mse_loss(pitch_pred, pitch_true_next)
    yaw_loss = _mse_loss(yaw_pred, yaw_true_next)
    vx_loss = _mse_loss(vx_pred, vx_true_next)
    vy_loss = _mse_loss(vy_pred, vy_true_next)
    vz_loss = _mse_loss(vz_pred, vz_true_next)
    w_roll_loss = _mse_loss(w_roll_pred, w_roll_true_next)
    w_pitch_loss = _mse_loss(w_pitch_pred, w_pitch_true_next)
    w_yaw_loss = _mse_loss(w_yaw_pred, w_yaw_true_next)
    
    # Apply per-channel weights (default: all ones)
    if channel_weights is None:
        cw = torch.ones(12, device=X_curr.device)
    else:
        cw = channel_weights.to(X_curr.device)

    total_loss = (cw[0]  * x_loss       + cw[1]  * y_loss     + cw[2]  * z_loss +
                  cw[3]  * roll_loss    + cw[4]  * pitch_loss + cw[5]  * yaw_loss +
                  cw[6]  * vx_loss     + cw[7]  * vy_loss    + cw[8]  * vz_loss +
                  cw[9]  * w_roll_loss + cw[10] * w_pitch_loss + cw[11] * w_yaw_loss)

    return total_loss
## Validate/Evaluate model
def validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, dt,
                  mean=None, std=None, channel_weights=None):
    """
    Validate the model using the validation data without using gradients.
    Returns GPU tensors — caller should only call .item() when needed for printing/logging.
    """
    model.eval()

    with torch.no_grad():
        loss_data = data_loss(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, dt, channel_weights=None)

        return loss_data
## Train model
import copy

def train_one_epoch(
    model, 
    optimizer, 
    X_curr_NN, U_curr_NN, X_curr, X_next, U_train_current, dt,
    mean=None, std=None, channel_weights=None):

    model.train()
    # Reset the gradients
    optimizer.zero_grad() 
    
    loss_data = data_loss(model, X_curr_NN, U_curr_NN, X_curr, X_next, dt, channel_weights=None)
    
    # Backward propagate the gradients
    loss_data.backward() 
    
    # Perform an update step using the optimizer
    optimizer.step()  

    return loss_data

def train_for_N_epochs(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr,
                       X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                       dt_mean, NN_config, idx, SEED, epochs, n_prints=1):
    start_time = datetime.datetime.now()

    n_input = NN_config["n_input"]
    n_output = NN_config["n_output"]
    hidden_layers_size = NN_config["hidden_layers_size"]
    activation_functions = NN_config["activation_functions"]
    eta = NN_config["eta"]
    eta_min = NN_config["eta_min"]
    Weight_decay = NN_config["Weight_decay"]

    set_seed(SEED)
    model = model_definition(NN, n_input, n_output, hidden_layers_size, activation_functions[1], p=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta, weight_decay=Weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )

    # Data for training the model
    X_train_current_NN = X_train_current_norm[idx]
    U_train_curr_NN = U_train_curr_norm[idx]
    X_train_current = X_train_current[idx]
    X_train_next = X_train_next[idx]
    U_train_curr = U_train_curr[idx]
    dt_train_NN = dt_mean

    train_loss = []
    val_loss = []
    min_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        train_loss.append(train_one_epoch(model, optimizer, X_train_current_NN, U_train_curr_NN, X_train_current, X_train_next, U_train_curr, dt_train_NN).item())
        val_loss.append(validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, dt_mean).item())   

        scheduler.step()  # Update the learning rate according to the scheduler

        if val_loss[-1] < min_val_loss:
            min_val_loss = val_loss[-1]
            best_model = copy.deepcopy(model)  # Save the state dict of the best model (use deepcopy to ensure we get a separate copy of the state dict)
            best_epoch = epoch + 1  # Store the epoch number of the best model (add 1 to convert from 0-indexed to 1-indexed)

        if (epoch+1) % (epochs//n_prints) == 0: # or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss[-1]:.7f}, Val Loss: {val_loss[-1]:.7f}")
    
    end_time = datetime.datetime.now()
    diff_time = (end_time - start_time).total_seconds()
    print(f"Chosen model from epoch {best_epoch} with validation loss {min_val_loss:.7f}")
    
    return np.array(list(zip(range(1, epochs+1), train_loss, val_loss))), best_model, diff_time
## MC-Dropout
# Enable dropout during inference for MC Dropout
def enable_dropout_only(model):
    # Set dropout layers to train mode, but keep the rest of the model in eval mode
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

@torch.no_grad() # We don't need gradients for inference
def mc_dropout_predict(model, NN_input, T=30, return_features=False):
    # Eval mode disables dropout, so we need to enable it manually
    model.eval()
    enable_dropout_only(model)

    # Run T stochastic forward passes through the model to get T predictions
    preds = []
    
    if return_features:
        features = []
        for _ in range(T):
            y, z = model(NN_input, return_features=return_features)
            preds.append(y)  # Each pred is [batch, out_dim]
            features.append(z)  # Each feature is [batch, feature_dim]

        features = torch.stack(features, dim=0)  # [T, batch, feature_dim]

    else:
        for _ in range(T):
            y = model(NN_input, return_features=return_features)
            preds.append(y)  # Each pred is [batch, out_dim]

    preds = torch.stack(preds, dim=0)  # [T, batch, out_dim]

    # Calculate mean and variance across the T predictions for each data point
    mean = preds.mean(dim=0)
    var  = preds.var(dim=0, unbiased=False)

    if return_features:
        mean_feat = features.mean(dim=0)       # [B, feat_dim]
        return mean, var, mean_feat
    
    return mean, var

# Convert variance to an uncertainty score (e.g., by taking the mean across output dimensions)
def uncertainty_score_from_var(var):
    # var: [batch, out_dim] -> score pr datapunkt [batch]
    # std = torch.sqrt(var + 1e-8)
    # return std.mean(dim=1)
    return var.sum(dim=1)
## URFEAL functions
### Clusterer
def clusterer(unlabeled_features, radius, beta):
    feats_np = unlabeled_features.cpu().numpy()
    
    clustering = DBSCAN(eps=radius, min_samples=beta).fit(feats_np)

    labels = clustering.labels_  # Cluster labels for each data point
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters (excluding noise)

    return labels, n_clusters
### Remove redundant features (Currently unused)

def remove_redundant_features(features, u_score, radius, B, beta):
    sorted_u_score, sorted_indices = torch.sort(u_score, descending=True)
    # Take point with hiigest score. Remove points within a certain radius in feature space
    # Select the point with the highest uncertainty score, 
    # remove points within a certain radius in feature space, 
    # then select the next point with the highest score among the remaining points, 
    # and so on until we have selected the desired number of points or there are no more points left.
    
    N = len(sorted_indices)
    B_temp = B

    if B > 10:
        B_temp = max(min(B // 5, N // 5), beta)
    
    counter = 0

    marked_set = []

    if N <= beta and B > beta:
        marked_set = sorted_indices.tolist()
        counter = N
    else:
        while len(sorted_indices) > 0 and B_temp > 0:
            c = sorted_indices[0]  # Index of the point with the highest score
            # Remove points within the specified radius in feature space
            dist = (features[c] - features[sorted_indices]).pow(2).sum(1).sqrt()  # Compute distance from point c to all other points
            keep_indices = torch.where(dist > radius)[0]  # Keep points that are outside the radius
            sorted_indices = sorted_indices[keep_indices]  # Update the sorted indices to keep only the
            marked_set.append(c.item())  # Add the selected point to the marked set
            # B -= 1  # Decrease the budget by one for each selected point
            B_temp -= 1
            counter += 1


    B = B - counter  # Update the remaining budget after selecting points
    print("Selected points:", counter)

    return marked_set, B
### OFE for Regression
@torch.no_grad()
def run_OFE_regression(
    model,
    X_train_current_norm,
    U_train_curr_norm,
    X_train_current,
    X_train_next,
    dt_mean,
    labeled_idx,
    K=None,
    alpha=None,
    threshold_sigma=3.0,
    min_K=5
):
    """
    Regression-adapted OFE.

    Finds labeled samples that resemble other samples in feature space,
    but have targets/dynamics that deviate significantly from their neighbors.

    Returns:
        outlier_idx: global indices in the training set that are suspicious
        outlier_scores: score for all labeled samples
    """

    model.eval()

    # 1) Get NN input for labeled samples
    Z_l = get_NN_input(
        X_train_current_norm[labeled_idx],
        U_train_curr_norm[labeled_idx],
        X_train_current[labeled_idx]
    )   # [n_labeled, n_input] = [N_l, 16]

    # 2) Extract backbone features
    _, features_l = model(Z_l, return_features=True)

    # 3) Target-delta for regression/dynamic deviation
    delta = (X_train_next[labeled_idx] - X_train_current[labeled_idx]).clone()

    delta[:, 3:6] = torch.atan2(torch.sin(delta[:, 3:6]), torch.cos(delta[:, 3:6])) # Wrap angle differences to [-pi, pi]
    delta = delta / dt_mean 

    Y_l = torch.cat([
        delta[:, 6:7],    # vx_dot
        delta[:, 7:8],    # vy_dot
        delta[:, 8:9],    # vz_dot
        delta[:, 9:10],   # w_roll_dot
        delta[:, 10:11],  # w_pitch_dot
        delta[:, 11:12],  # w_yaw_dot
        delta[:, 0:1],    # x_dot
        delta[:, 1:2],    # y_dot
        delta[:, 2:3],    # z_dot
        delta[:, 3:4],    # roll_dot
        delta[:, 4:5],    # pitch_dot
        delta[:, 5:6],    # yaw_dot
    ], dim=1)   # [n_labeled, target_dim] = [N_l, 12]

    # Normalize targets for better distance calculations (we want to treat all target dimensions equally)
    Y_l = (Y_l - Y_l.mean(dim=0, keepdim=True)) / (Y_l.std(dim=0, keepdim=True, unbiased=False) + 1e-8)

    n_labeled = len(labeled_idx)

    if n_labeled <= min_K + 1:
        return torch.empty(0, dtype=torch.long), torch.zeros(n_labeled)

    # 4) K according to the paper's idea:
    # K = num(DT) * alpha
    # But for this setup it's more stable to use a limited K.
    if K is None:
        if alpha is None:
            # Practical default: about sqrt(N), but at least min_K
            K = max(min_K, int(np.sqrt(n_labeled)))
        else:
            K = max(min_K, int(n_labeled * alpha))

    K = min(K, n_labeled - 1)

    # 5) Distance between labeled features
    d_feat = torch.cdist(features_l, features_l)    # [n_labeled, n_labeled] pairwise distances in feature space

    # Remove self as nearest neighbor
    d_feat.fill_diagonal_(float("inf"))

    # 6) Find K nearest neighbors in feature space
    knn_idx = torch.topk(d_feat, k=K, largest=False).indices    # [n_labeled, K] indices of K nearest neighbors for each labeled sample

    # 7) Compute edge-core target as the mean of neighbors' targets
    neighbor_targets = Y_l[knn_idx]             # [n_labeled, K, target_dim]
    edge_target = neighbor_targets.mean(dim=1)  # [n_labeled, target_dim] mean target of K neighbors for each labeled sample

    # 8) Outlier score: distance between own target and neighbor target
    outlier_scores = torch.norm(Y_l - edge_target, dim=1)

    # 9) 3-sigma threshold
    mean_score = outlier_scores.mean()
    std_score = outlier_scores.std(unbiased=False)

    threshold = mean_score + threshold_sigma * std_score

    outlier_mask = outlier_scores > threshold

    outlier_idx = labeled_idx[outlier_mask]

    return outlier_idx, outlier_scores
## Fisher functions
train_loss_fn = nn.MSELoss()

# def fisher_information_one_step(
#     model,
#     X_curr_NN,
#     U_curr_NN,
#     X_curr,
#     X_next,
#     dt
# ):
    
#     model.eval()
#     fisher_scores = []

#     x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = X_curr[:,0:1], X_curr[:,1:2], X_curr[:,2:3], X_curr[:,3:4], X_curr[:,4:5], X_curr[:,5:6], X_curr[:,6:7], X_curr[:,7:8], X_curr[:,8:9], X_curr[:,9:10], X_curr[:,10:11], X_curr[:,11:12] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)

#     # 1a) Prepare the states and control inputs necessary for the NN
#     _,_,_, _, _, _, vx_n, vy_n, vz_n, w_roll_n, w_pitch_n, w_yaw_n, _, _, _, _, _, _ = X_curr_NN[:,0:1], X_curr_NN[:,1:2], X_curr_NN[:,2:3], X_curr_NN[:,3:4], X_curr_NN[:,4:5], X_curr_NN[:,5:6], X_curr_NN[:,6:7], X_curr_NN[:,7:8], X_curr_NN[:,8:9], X_curr_NN[:,9:10], X_curr_NN[:,10:11], X_curr_NN[:,11:12], X_curr_NN[:,12:13], X_curr_NN[:,13:14], X_curr_NN[:,14:15], X_curr_NN[:,15:16], X_curr_NN[:,16:17], X_curr_NN[:,17:18] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
#     sin_roll,cos_roll = torch.sin(roll), torch.cos(roll)
#     sin_pitch,cos_pitch = torch.sin(pitch), torch.cos(pitch)
#     sin_yaw,cos_yaw = torch.sin(yaw), torch.cos(yaw)
#     thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n = U_curr_NN[:,0:1], U_curr_NN[:,1:2], U_curr_NN[:,2:3], U_curr_NN[:,3:4] # current control inputs

#     # 1b) Prepare NN input tensor
#     Z = torch.cat([sin_roll, cos_roll,
#                    sin_pitch, cos_pitch,
#                    sin_yaw, cos_yaw,
#                    vx_n, vy_n, vz_n,
#                    w_roll_n, w_pitch_n, w_yaw_n, thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n], dim=1) # dim =1 means concatenate along columns (we do it because we have 2D tensors: n_batch x features)

#     for q, x_curr, t in zip(Z, X_curr, X_next):
#         q.unsqueeze_(0)  # add batch dimension to the input (shape becomes [1, n_input])
#         pred = model(q)                  # shape [12]
        
#         vx_dot_pred = pred[0:1] 
#         vy_dot_pred = pred[1:2]
#         vz_dot_pred = pred[2:3]
#         w_roll_dot_pred = pred[3:4]
#         w_pitch_dot_pred = pred[4:5]
#         w_yaw_dot_pred = pred[5:6]

#         x_dot_pred = pred[6:7]
#         y_dot_pred = pred[7:8]
#         z_dot_pred = pred[8:9]
#         roll_dot_pred = pred[9:10]
#         pitch_dot_pred = pred[10:11]
#         yaw_dot_pred = pred[11:12]


#         x_pred = x_curr[0:1] + x_dot_pred * dt
#         y_pred = x_curr[1:2] + y_dot_pred * dt
#         z_pred = x_curr[2:3] + z_dot_pred * dt

#         roll_pred = x_curr[3:4] + roll_dot_pred * dt
#         pitch_pred = x_curr[4:5] + pitch_dot_pred * dt
#         yaw_pred = x_curr[5:6] + yaw_dot_pred * dt

#         roll_pred = torch.atan2(torch.sin(roll_pred), torch.cos(roll_pred))
#         pitch_pred = torch.atan2(torch.sin(pitch_pred), torch.cos(pitch_pred))
#         yaw_pred = torch.atan2(torch.sin(yaw_pred), torch.cos(yaw_pred))

#         vx_pred = x_curr[6:7] + vx_dot_pred * dt
#         vy_pred = x_curr[7:8] + vy_dot_pred * dt
#         vz_pred = x_curr[8:9] + vz_dot_pred * dt
#         w_roll_pred = x_curr[9:10] + w_roll_dot_pred * dt
#         w_pitch_pred = x_curr[10:11] + w_pitch_dot_pred * dt
#         w_yaw_pred = x_curr[11:12] + w_yaw_dot_pred * dt

#         model_pred = torch.cat([x_pred, y_pred, z_pred, roll_pred, pitch_pred, yaw_pred, vx_pred, vy_pred, vz_pred, w_roll_pred, w_pitch_pred, w_yaw_pred]) # shape [12]
#         model_pred = model_pred.view_as(pred)  # ensure model_pred has the same shape as pred (which is [12])

#         x_true_next = t[0:1]
#         y_true_next = t[1:2]
#         z_true_next = t[2:3]
#         roll_true_next = t[3:4]
#         pitch_true_next = t[4:5]
#         yaw_true_next = t[5:6]
#         roll_true_next = torch.atan2(torch.sin(roll_true_next), torch.cos(roll_true_next))
#         pitch_true_next = torch.atan2(torch.sin(pitch_true_next), torch.cos(pitch_true_next))
#         yaw_true_next = torch.atan2(torch.sin(yaw_true_next), torch.cos(yaw_true_next))
        
#         vx_true_next = t[6:7]
#         vy_true_next = t[7:8]
#         vz_true_next = t[8:9]
#         w_roll_true_next = t[9:10]
#         w_pitch_true_next = t[10:11]      
#         w_yaw_true_next = t[11:12]

#         target = torch.cat([x_true_next, y_true_next, z_true_next, roll_true_next, pitch_true_next, yaw_true_next, vx_true_next, vy_true_next, vz_true_next, w_roll_true_next, w_pitch_true_next, w_yaw_true_next]) # shape [12]
#         target = target.view_as(model_pred)  # ensure target has the same shape as model_pred

#         loss = train_loss_fn(model_pred, target)  # sum the loss over the output dimensions
#         model.zero_grad()  # zero the gradients before backward pass
#         loss.backward()
    
#         F_i = torch.tensor(0.0, device=pred.device)
#         score = 0.0
#         for param in model.parameters():
#             if param.grad is not None:
#                 score = param.grad.detach().pow(2)  # square the gradients
#             F_i += score.sum()  # sum over all parameters to get the Fisher information
    
#         fisher_scores.append(F_i)
#     return torch.stack(fisher_scores)  

# def get_FI(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, dt_mean, NN_config, seed):
#     # Calculate fisher scores

#     n_input = NN_config["n_input"]
#     n_output = NN_config["n_output"]
#     hidden_layers_size = NN_config["hidden_layers_size"]
#     activation_functions = NN_config["activation_functions"]
#     eta = NN_config["eta"]
#     Weight_decay = NN_config["Weight_decay"]

#     set_seed(seed)
#     model_fi = model_definition(NN, n_input, n_output, hidden_layers_size, activation_functions[0], p=0.0)
#     # model_fi = nn_model(input_size, output_size, hidden_size)
#     opt_fi = torch.optim.AdamW(model_fi.parameters(), lr=eta, weight_decay=Weight_decay)
#     for epoch in range(50):
#         train_one_epoch(model_fi, opt_fi, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, dt_mean)

#     F_i = fisher_information_one_step(model_fi, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, dt_mean) 
#     print("Fisher Scores calculated.!")
#     return F_i
    


def fisher_per_sample(
    model,
    X_curr_NN,
    U_curr_NN,
    X_curr,
    X_next,
    dt
):
    
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    x, y, z, roll, pitch, yaw, vx, vy, vz, w_roll, w_pitch, w_yaw = X_curr[:,0:1], X_curr[:,1:2], X_curr[:,2:3], X_curr[:,3:4], X_curr[:,4:5], X_curr[:,5:6], X_curr[:,6:7], X_curr[:,7:8], X_curr[:,8:9], X_curr[:,9:10], X_curr[:,10:11], X_curr[:,11:12] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)

    # 1a) Prepare the states and control inputs necessary for the NN
    _,_,_, _, _, _, vx_n, vy_n, vz_n, w_roll_n, w_pitch_n, w_yaw_n, _, _, _, _, _, _ = X_curr_NN[:,0:1], X_curr_NN[:,1:2], X_curr_NN[:,2:3], X_curr_NN[:,3:4], X_curr_NN[:,4:5], X_curr_NN[:,5:6], X_curr_NN[:,6:7], X_curr_NN[:,7:8], X_curr_NN[:,8:9], X_curr_NN[:,9:10], X_curr_NN[:,10:11], X_curr_NN[:,11:12], X_curr_NN[:,12:13], X_curr_NN[:,13:14], X_curr_NN[:,14:15], X_curr_NN[:,15:16], X_curr_NN[:,16:17], X_curr_NN[:,17:18] # current state vector components (X[:,:-1,0:1] means all batches, all sequences except the last one, and only the first element of the state)
    sin_roll,cos_roll = torch.sin(roll), torch.cos(roll)
    sin_pitch,cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_yaw,cos_yaw = torch.sin(yaw), torch.cos(yaw)
    thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n = U_curr_NN[:,0:1], U_curr_NN[:,1:2], U_curr_NN[:,2:3], U_curr_NN[:,3:4] # current control inputs

    # 1b) Prepare NN input tensor
    Z = torch.cat([sin_roll, cos_roll,
                   sin_pitch, cos_pitch,
                   sin_yaw, cos_yaw,
                   vx_n, vy_n, vz_n,
                   w_roll_n, w_pitch_n, w_yaw_n, thrust_n, torque_roll_n, torque_pitch_n, torque_yaw_n], dim=1) # dim =1 means concatenate along columns (we do it because we have 2D tensors: n_batch x features)
    
    
    def loss(params, buffers, z, x_curr, x_next):
        pred = functional_call(model, (params, buffers), (z.unsqueeze(0),)).squeeze(0)                  # shape [12]
        
        vx_dot_pred = pred[0:1] 
        vy_dot_pred = pred[1:2]
        vz_dot_pred = pred[2:3]
        w_roll_dot_pred = pred[3:4]
        w_pitch_dot_pred = pred[4:5]
        w_yaw_dot_pred = pred[5:6]

        x_dot_pred = pred[6:7]
        y_dot_pred = pred[7:8]
        z_dot_pred = pred[8:9]
        roll_dot_pred = pred[9:10]
        pitch_dot_pred = pred[10:11]
        yaw_dot_pred = pred[11:12]


        x_pred = x_curr[0:1] + x_dot_pred * dt
        y_pred = x_curr[1:2] + y_dot_pred * dt
        z_pred = x_curr[2:3] + z_dot_pred * dt

        roll_pred = x_curr[3:4] + roll_dot_pred * dt
        pitch_pred = x_curr[4:5] + pitch_dot_pred * dt
        yaw_pred = x_curr[5:6] + yaw_dot_pred * dt

        roll_pred = torch.atan2(torch.sin(roll_pred), torch.cos(roll_pred))
        pitch_pred = torch.atan2(torch.sin(pitch_pred), torch.cos(pitch_pred))
        yaw_pred = torch.atan2(torch.sin(yaw_pred), torch.cos(yaw_pred))

        vx_pred = x_curr[6:7] + vx_dot_pred * dt
        vy_pred = x_curr[7:8] + vy_dot_pred * dt
        vz_pred = x_curr[8:9] + vz_dot_pred * dt
        w_roll_pred = x_curr[9:10] + w_roll_dot_pred * dt
        w_pitch_pred = x_curr[10:11] + w_pitch_dot_pred * dt
        w_yaw_pred = x_curr[11:12] + w_yaw_dot_pred * dt

        model_pred = torch.cat([x_pred, y_pred, z_pred, roll_pred, pitch_pred, yaw_pred, vx_pred, vy_pred, vz_pred, w_roll_pred, w_pitch_pred, w_yaw_pred]) # shape [12]
        model_pred = model_pred.view_as(pred)  # ensure model_pred has the same shape as pred (which is [12])

        x_true_next = x_next[0:1]
        y_true_next = x_next[1:2]
        z_true_next = x_next[2:3]
        roll_true_next = x_next[3:4]
        pitch_true_next = x_next[4:5]
        yaw_true_next = x_next[5:6]
        roll_true_next = torch.atan2(torch.sin(roll_true_next), torch.cos(roll_true_next))
        pitch_true_next = torch.atan2(torch.sin(pitch_true_next), torch.cos(pitch_true_next))
        yaw_true_next = torch.atan2(torch.sin(yaw_true_next), torch.cos(yaw_true_next))
        
        vx_true_next = x_next[6:7]
        vy_true_next = x_next[7:8]
        vz_true_next = x_next[8:9]
        w_roll_true_next = x_next[9:10]
        w_pitch_true_next = x_next[10:11]      
        w_yaw_true_next = x_next[11:12]

        target = torch.cat([x_true_next, y_true_next, z_true_next, roll_true_next, pitch_true_next, yaw_true_next, vx_true_next, vy_true_next, vz_true_next, w_roll_true_next, w_pitch_true_next, w_yaw_true_next]) # shape [12]
        target = target.view_as(model_pred)  # ensure target has the same shape as model_pred

        loss = train_loss_fn(model_pred, target)  # sum the loss over the output dimensions
        return loss.sum()
    
    grad_fn = grad(loss)

    grads = vmap(grad_fn, in_dims=(None, None, 0, 0, 0))(params, buffers, Z, X_curr, X_next)

    fisher = None
    for g in grads.values():
        g2 = g.pow(2).flatten(1).sum(dim=1)
        fisher = g2 if fisher is None else fisher + g2

    return fisher

def train_sampling_one_epoch(model, optimizer, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, dt_mean, fi_epochs=None, epoch=None, fisher=False):
    model.train()

    optimizer.zero_grad() 

    # Feed the data through the model
    # output = model(X_train) 

    # Compute the negative log-likelihood loss
    loss_data = data_loss(model, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, dt_mean, channel_weights=None) 

    if fisher:
        fi_b = fisher_per_sample(model, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, dt_mean).detach().cpu()

        fi_epochs[epoch] = fi_b
        
    # Backward propagate the gradients
    loss_data.backward() 

    # Perform an update step using the optimizer
    optimizer.step() 


    return fi_epochs


def get_FI(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, dt_mean, epochs, NN_config, seed):
    # Calculate fisher scores
    start_time = datetime.datetime.now()

    n_input = NN_config["n_input"]
    n_output = NN_config["n_output"]
    hidden_layers_size = NN_config["hidden_layers_size"]
    activation_functions = NN_config["activation_functions"]
    eta = NN_config["eta"]
    Weight_decay = NN_config["Weight_decay"]

    set_seed(seed)
    model_fi = model_definition(NN, n_input, n_output, hidden_layers_size, activation_functions[0], p=0.0)
    opt_fi = torch.optim.AdamW(model_fi.parameters(), lr=eta, weight_decay=Weight_decay)

    fi_epochs=torch.zeros(epochs, X_train_current_norm.shape[0])
    for epoch in range(epochs):
        fi_epochs = train_sampling_one_epoch(model_fi, opt_fi, X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, dt_mean, fi_epochs=fi_epochs, epoch=epoch, fisher=True)
    
    F_auc = torch.trapezoid(fi_epochs, dx=1.0, dim=0)  # Integrate over the iterations to get an overall FI score for each data point
    
    end_time = datetime.datetime.now()
    diff_time = (end_time - start_time).total_seconds()
    print("Fisher Scores calculated.!")
    return F_auc, diff_time
    