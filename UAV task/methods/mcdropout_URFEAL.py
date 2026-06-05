import torch
import datetime

from functions import set_seed, NN, model_definition, train_one_epoch, validate_model, get_NN_input, mc_dropout_predict, uncertainty_score_from_var, clusterer, run_OFE_regression

def run_MC_URF(X_train_current_norm, U_train_curr_norm, X_train_current, X_train_next, U_train_curr, 
             X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr,
             dt_mean,
             L, budget, SEED, NN_input, epochs=[20, 10], radius=0.2, beta=10, use_OFE=False):
    
    time_dict = {}  # Dictionary to store time taken for each percentage
    start_time = datetime.datetime.now()

    lp_index = 0

    n_input = NN_input["n_input"]
    n_output = NN_input["n_output"]
    hidden_layers_size = NN_input["hidden_layers_size"]
    activation_fn = NN_input["activation_functions"][0]  # Use the first activation function for sampling
    eta = NN_input["eta"]
    Weight_decay = NN_input["Weight_decay"]
    p = NN_input["p"]

    # Initialize the MC + URFEAL data (same as MCAL, but we keep the original non-normalized data for the URF part) 
    X_MC_URF_current_norm = X_train_current_norm
    U_MC_URF_curr_norm = U_train_curr_norm
    X_MC_URF_current = X_train_current
    X_MC_URF_next = X_train_next
    U_MC_URF_curr = U_train_curr

    # Fresh model ONCE at the beginning
    set_seed(SEED)
    model = model_definition(NN, n_input, n_output, hidden_layers_size, activation_fn, p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta, weight_decay=Weight_decay)

    # Amount of data to start with
    N = X_MC_URF_current_norm.shape[0]
    # n0 = int(0.05 * N) # Normally would be budget[0], but paper says 5%
    n0 = max(budget[0], 8)  # Ensure we start with at least one batch worth of data for training

    # Split the data into an initial labeled set and an unlabeled set
    perm = torch.randperm(N)
    labeled_idx = perm[:n0]
    unlabeled_idx = perm[n0:]

    history = []

    # Initial training on the small labeled set
    for epoch in range(epochs[0]):
        train_loss = train_one_epoch(model, optimizer, 
                                    X_MC_URF_current_norm[labeled_idx], U_MC_URF_curr_norm[labeled_idx], X_MC_URF_current[labeled_idx], X_MC_URF_next[labeled_idx], U_MC_URF_curr[labeled_idx], 
                                    dt_mean)
        if epoch == 0:
            val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                    dt_mean)
            history.append([len(labeled_idx), train_loss.item(), val_loss.item()])
    
    val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                    dt_mean)
    history.append([len(labeled_idx), train_loss.item(), val_loss.item()])

    # Define the target counts for analysis
    target_count = int(max(L) * N)
    targets = [int(l * N) for l in L]

    labeled_idx_list = []  # To store labeled indices at each target count for later analysis
    unlabeled_idx_list = []  # To store unlabeled indices at each target count for later analysis

    print_counter = 0

    cluster_radius = radius
    elim_radius = radius
    
    # Active learning loop. As long as we haven't labeled enough data to reach the largest percentage in L, keep going
    while len(labeled_idx) < target_count:
        
        Z_u = get_NN_input(X_MC_URF_current_norm[unlabeled_idx], U_MC_URF_curr_norm[unlabeled_idx], X_MC_URF_current[unlabeled_idx])

        # MC Dropout selection to calculate the uncertainty scores and the features
        _, var_u, features = mc_dropout_predict(model, Z_u, T=30, return_features=True)
        u_score = uncertainty_score_from_var(var_u)

        labels, n_clusters = clusterer(features, radius=cluster_radius, beta=beta)

        labels = torch.tensor(labels, dtype=torch.long)
        # print(f"labels shape: {labels.shape}")

        valid_labels_mask = labels >= 0
        valid_labels = labels[valid_labels_mask]
        # print(f"valid_labels shape: {valid_labels.shape}")

        remaining_budget = target_count - len(labeled_idx)
        B = min(budget[1], remaining_budget)
  

        available = torch.ones(len(unlabeled_idx), dtype=torch.bool)
        marked_set = []
        # Trying to check clusters, for which has the highest uncertainty score
        # Then I select the most uncertain point in that cluster and remove all points within the elimination radius
        # Then I repeat until I have marked B points or there are no more clusters to check
        while len(marked_set) < B:
            best_cluster = None
            best_score = -float("inf")

            for cluster in torch.unique(labels[(labels >= 0) & available]):
                cluster_indices = torch.where((labels == cluster) & available)[0]

                if len(cluster_indices) < beta:
                    continue

                cluster_scores = u_score[cluster_indices]
                max_score = cluster_scores.max().item()

                if max_score > best_score:
                    best_score = max_score
                    best_cluster = cluster

            if best_cluster is None:
                print("No more valid clusters to select from. Filling rest with MC-dropout fallback.")
                break

            cluster_indices = torch.where((labels == best_cluster) & available)[0]
            cluster_scores = u_score[cluster_indices]

            best_local = torch.argmax(cluster_scores)
            core_rel = cluster_indices[best_local]

            marked_set.append(core_rel.item())

            dist = torch.norm(features[core_rel] - features[cluster_indices], dim=1)
            keep_mask = dist > elim_radius

            available[cluster_indices[~keep_mask]] = False

        # Fallback for when best_cluster is None or 
        # when we haven't marked enough points after going through all clusters
        if len(marked_set) < B:
            missing = B - len(marked_set)

            all_rel = torch.arange(len(unlabeled_idx))
            selected_mask = torch.zeros(len(unlabeled_idx), dtype=torch.bool)

            if len(marked_set) > 0:
                selected_mask[torch.tensor(marked_set, dtype=torch.long)] = True

            # First fallback: choose among non-selected AND non-redundant samples
            fallback_rel = all_rel[(~selected_mask) & available]

            # Second fallback: if not enough, allow redundant/noise samples too
            if len(fallback_rel) < missing:
                fallback_rel = all_rel[~selected_mask]

            missing = min(missing, len(fallback_rel))

            if missing > 0:
                fallback_scores = u_score[fallback_rel]
                fallback_top = torch.topk(fallback_scores, k=missing).indices
                marked_set.extend(fallback_rel[fallback_top].tolist())

        
        query_rel = torch.tensor(marked_set, dtype=torch.long)      # Relative indices in the unlabeled set
        query_idx = unlabeled_idx[query_rel]                        # Global indices in the whole dataset

        labeled_idx = torch.cat([labeled_idx, query_idx])
        labeled_idx = torch.unique(labeled_idx) 

        mask = torch.ones(N, dtype=torch.bool)
        mask[labeled_idx] = False
        unlabeled_idx = torch.arange(N)[mask]

        for epoch in range(epochs[1]):
            train_loss = train_one_epoch(model, optimizer, 
                                        X_MC_URF_current_norm[labeled_idx], U_MC_URF_curr_norm[labeled_idx], X_MC_URF_current[labeled_idx], X_MC_URF_next[labeled_idx], U_MC_URF_curr[labeled_idx], 
                                        dt_mean)
            
        if use_OFE:
            outlier_idx, outlier_score = run_OFE_regression(
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
            )
            print(f"OFE found {len(outlier_idx)} outliers.")
   
    
        val_loss = validate_model(model, X_val_current_NN, U_val_curr_NN, X_val_current, X_val_next, U_val_curr, 
                                        dt_mean)

        history.append([len(labeled_idx), train_loss.item(), val_loss.item()])

        # Every time 1000 new data points are labeled, print the current status
        if len(labeled_idx) // 1000 > print_counter:
            print(f"MC-Dropout + URFEAL:\t\tLabeled {len(labeled_idx)}/{target_count}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print_counter += 1
        
        # Once we reach each target count in L, store the labeled indices for that percentage for later analysis
        if len(labeled_idx) >= targets[0]:
            labeled_idx_list.append(labeled_idx.clone().cpu().numpy())  # Store a copy of the current labeled indices
            unlabeled_idx_list.append(unlabeled_idx.clone().cpu().numpy())  # Store a copy of the current unlabeled indices
            targets.remove(targets[0])
            time_end = datetime.datetime.now()
            time_dict[L[lp_index]] = (time_end - start_time).total_seconds()
            lp_index += 1
    
    print(f"MC-Dropout + URFEAL:\t\tLabeled {len(labeled_idx)}/{target_count}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("MC-Dropout + URFEAL done!")
    return labeled_idx_list, unlabeled_idx_list, time_dict