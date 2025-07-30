import os
import threading

def initialize_parameters(settings, training_approach):
    # Don't override data_root if it's already set in config
    if "data_root" not in settings:
        settings["data_root"] = "Data"
    settings["roc_path"] = None  # "roc"
    settings["matrix_path"] = None  # "matrix"
    settings["save_results"] = f"results/{training_approach}/"
    settings["save_model"] = f"models/{training_approach}/"
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