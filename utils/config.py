import os
import time
import threading

def initialize_parameters(settings, training_approach):
    # Don't override data_root if it's already set in config
    if "data_root" not in settings:
        settings["data_root"] = "Data"
    
    # Create descriptive directory names
    dataset_name = settings.get('name_dataset', 'unknown')
    n_clients = settings.get('number_of_clients_per_node', 1) * settings.get('number_of_nodes', 1)
    n_rounds = settings.get('n_rounds', 10)
    
    # Determine FL approach
    # Determine configuration suffix
    config_parts = []
    if settings.get('clustering', False):
        config_parts.append("smpc")
    if settings.get('diff_privacy', False):
        config_parts.append("dp")
    
    config_suffix = "_".join(config_parts) if config_parts else "classic"
    
    # Create informative directory name with key parameters
    result_dir = f"results/{dataset_name}_{config_suffix}_c{n_clients}_r{n_rounds}/"
    
    settings["roc_path"] = None  # "roc"
    settings["matrix_path"] = None  # "matrix"
    settings["save_results"] = result_dir
    # Legacy save_model removed - ModelManager handles all model saving
    # Legacy client models directory - now handled by ModelManager

    # clients
    training_barrier = threading.Barrier(settings['number_of_clients_per_node'])

    if training_approach.lower() == "cfl":
        settings["n_clients"] = settings["number_of_clients_per_node"] * settings["number_of_nodes"]
        [settings.pop(key, None) for key in ["number_of_clients_per_node",
                                             "number_of_nodes",
                                             "min_number_of_clients_in_cluster",
                                             "k",
                                             "m",
                                             "secret_sharing"]]

    elif training_approach.lower() == "bfl":
        if settings["m"] < settings["k"]:
            raise ValueError(
                "the number of parts used to reconstruct the secret must be greater than the threshold (k)")
        print("Number of Nodes: ", settings["number_of_nodes"],
              "\tNumber of Clients per Node: ", settings["number_of_clients_per_node"],
              "\tNumber of Clients per Cluster: ", settings["min_number_of_clients_in_cluster"], "\n")

    os.makedirs(settings["save_results"], exist_ok=True)
    # Legacy save_model and client_models directory creation removed - ModelManager handles directory structure
    return training_barrier, None