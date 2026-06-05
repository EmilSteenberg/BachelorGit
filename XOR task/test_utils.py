import os
import pickle
import torch.nn as nn

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


def model_to_state(obj):
    """
    Konverterer model/list/nested list/None til state_dict-struktur.
    """
    if obj is None:
        return None
    
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    
    if isinstance(obj, list):
        return [model_to_state(item) for item in obj]
    
    raise TypeError(f"Unsupported type in models_dict: {type(obj)}")


def save_resultData(save_dir, SEED,
                    models_dict,
                    histories_dict,
                    indices_dict,
                    fisher_scores_dict,
                    timings,
                    model_config):
    
    os.makedirs(save_dir, exist_ok=True)

    # New format is models[learning_percentage][method] = model

    models_state = {}
    for i, lp in enumerate(models_dict.keys()):
        models_state[lp] = {}
        for method, model in models_dict[lp].items():
            if model is not None:
                # print(f"Converting model for learning percentage {lp} and method {method} to state dict...")
                models_state[lp][method] = model_to_state(model)
            else:
                # print(f"No model to convert for learning percentage {lp} and method {method} (model is None)")
                pass

    # models_state = {
    #     key: model_to_state(value) for key, value in models_dict.items()
    # }

    # 🔹 2. Saml ALT i én dict
    resultData = {
        # Seed
        "SEED": SEED,
        
        # Models
        "models": models_state,

        # Histories
        "histories": histories_dict,

        # Indices
        "indices": indices_dict,

        # Fisher scores
        "fisher_scores": fisher_scores_dict,

        # Timings
        "timings": timings,

        # Model setup
        "model_config": model_config,

    }

    # 🔹 3. Gem fil
    filepath = os.path.join(save_dir, "resultData.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(resultData, f)

    print(f"Saved resultData for SEED {SEED} at: {filepath}")

def load_resultData(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        
    SEED = data["SEED"]
    models_state = data["models"]
    histories = data["histories"]
    indices = data["indices"]
    fisher_scores = data["fisher_scores"]
    timings = data["timings"]
    config = data["model_config"]
    
    return SEED, models_state, histories, indices, fisher_scores, timings, config

def load_model_from_state(state_dict, config):
    if state_dict is None:
        return None
    

    model = model_definition(
        NN,
        config["n_input"],
        config["n_output"],
        config["hidden_layers_size"],
        config["activation_functions"][1],
        config["p"]
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model
