import os
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
    fl_approach = "smpc" if settings.get('clustering', False) else "classic"
    
    # Create structured directory name
    result_dir = f"results/{dataset_name}_{fl_approach}_{n_clients}clients_{n_rounds}rounds/"
    
    settings["roc_path"] = None  # "roc"
    settings["matrix_path"] = None  # "matrix"
    settings["save_results"] = result_dir
    settings["save_model"] = f"models/{dataset_name}_{fl_approach}_{n_clients}clients_{n_rounds}rounds/"
    settings["save_client_models"] = f"{settings['save_results']}client_models/"

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
    os.makedirs(settings["save_model"], exist_ok=True)
    os.makedirs(settings["save_client_models"], exist_ok=True)
    return training_barrier, None