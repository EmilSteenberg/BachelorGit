import torch
import datetime

from functions import set_seed, NN, model_definition, train_one_epoch, validate_model, get_NN_input, mc_dropout_predict, uncertainty_score_from_var

def run_MCAL(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, 
             X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr,
             dt_mean,
             L, budget, SEED, NN_config, epochs=[20, 10]):

    time_dict = {}  # Dictionary to store time taken for each percentage
    start_time = datetime.datetime.now()

    lp_index = 0

    n_input = NN_config["n_input"]
    n_output = NN_config["n_output"]
    hidden_layers_size = NN_config["hidden_layers_size"]
    activation_fn = NN_config["activation_functions"][0]  # Use the first activation function for sampling
    eta = NN_config["eta"]
    Weight_decay = NN_config["Weight_decay"]
    p = NN_config["p"]

    # Initialize the MCAL data
    X_MCAL_current_norm = X_train_current_norm
    U_MCAL_curr_norm = U_train_curr_norm
    X_MCAL_current = X_train_current
    X_MCAL_next = X_train_next
    U_MCAL_curr = U_train_curr

    # Fresh model ONCE at the beginning
    set_seed(SEED)
    model = model_definition(NN, n_input, n_output, hidden_layers_size, activation_fn, p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta, weight_decay=Weight_decay)

    # Amount of data to start with
    N = X_MCAL_current_norm.shape[0]
    # n0 = int(0.10 * N)
    n0 = max(budget[0], 8)                 # Ensure we start with at least one batch worth of data for training

    perm = torch.randperm(N)
    labeled_idx = perm[:n0]
    unlabeled_idx = perm[n0:]

    history = []

    for epoch in range(epochs[0]):
        train_loss = train_one_epoch(model, optimizer, 
                                    X_MCAL_current_norm[labeled_idx], U_MCAL_curr_norm[labeled_idx], X_MCAL_current[labeled_idx], X_MCAL_next[labeled_idx], U_MCAL_curr[labeled_idx], 
                                    dt_mean)
        if epoch == 0:
            val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                    dt_mean)
            history.append([len(labeled_idx), train_loss.item(), val_loss.item()])
    
    val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                    dt_mean)
    history.append([len(labeled_idx), train_loss.item(), val_loss.item()])

    target_count = int(max(L) * N)
    targets = [int(l * N) for l in L]

    labeled_idx_list = []  # To store labeled indices at each target count for later analysis
    unlabeled_idx_list = []  # To store unlabeled indices at each target count for later analysis

    print_counter = 0
    
    while len(labeled_idx) < target_count:
        # Query phase
        # X_u = x_train_tensor[unlabeled_idx]
        Z_u = get_NN_input(X_MCAL_current_norm[unlabeled_idx], U_MCAL_curr_norm[unlabeled_idx], X_MCAL_current[unlabeled_idx])

        # MC Dropout selection
        K = min(budget[1], target_count - len(labeled_idx))
        mean_u, var_u = mc_dropout_predict(model, Z_u, T=30)
        u_score = uncertainty_score_from_var(var_u)

        top_rel = torch.topk(u_score, k=K).indices
        # print("u_score min/mean/max:",
                        # u_score.min().item(), u_score.mean().item(), u_score.max().item())

        query_idx = unlabeled_idx[top_rel]

        labeled_idx = torch.cat([labeled_idx, query_idx])
        labeled_idx = torch.unique(labeled_idx) 

        mask = torch.ones(N, dtype=torch.bool)
        mask[labeled_idx] = False
        unlabeled_idx = torch.arange(N)[mask]

        # Retrain on expanded labeled set
        # X_u = x_train_tensor[unlabeled_idx]
        # X_l = x_train_tensor[labeled_idx]
        # Y_l = t_train_tensor[labeled_idx]

        for epoch in range(epochs[1]):
            train_loss = train_one_epoch(model, optimizer, 
                                        X_MCAL_current_norm[labeled_idx], U_MCAL_curr_norm[labeled_idx], X_MCAL_current[labeled_idx], X_MCAL_next[labeled_idx], U_MCAL_curr[labeled_idx], 
                                        dt_mean)
   
    
        val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                        dt_mean)

        history.append([len(labeled_idx), train_loss, val_loss])

        # Every time 1000 new data points are labeled, print the current status
        if len(labeled_idx) // 1000 > print_counter:
            print(f"MC-Dropout:\t\tLabeled {len(labeled_idx)}/{target_count}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print_counter += 1
        
        # Once we reach each target count in L, store the labeled indices for that percentage for later analysis
        if len(labeled_idx) >= targets[0]:
            labeled_idx_list.append(labeled_idx.clone().cpu().numpy())  # Store a copy of the current labeled indices
            unlabeled_idx_list.append(unlabeled_idx.clone().cpu().numpy())  # Store a copy of the current unlabeled indices
            targets.remove(targets[0])
            time_end = datetime.datetime.now()
            time_dict[L[lp_index]] = (time_end - start_time).total_seconds()
            lp_index += 1
    
    print(f"MC-Dropout:\t\tLabeled {len(labeled_idx)}/{target_count}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("MC-Dropout done!")
    return labeled_idx_list, unlabeled_idx_list, time_dict