import logging
import threading
import os
import time
import json
import sys
import argparse

from federated import create_nodes, create_clients, cluster_generation, train_client
from utils import initialize_parameters
from data import load_dataset
from config import settings

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils.system_metrics import MetricsTracker
from utils.model_manager import ModelManager

import warnings
warnings.filterwarnings("ignore")

client_weights = []

if __name__ == "__main__":
    # Parse command-line arguments for experiment configuration
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name (overrides config.py)')
    parser.add_argument('--rounds', type=int, default=None, help='Number of rounds (overrides config.py)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config.py)')
    parser.add_argument('--clustering', type=str, default=None, choices=['true', 'false'], help='Enable clustering')
    parser.add_argument('--smpc', type=str, default=None, choices=['none', 'additif', 'shamir'], help='SMPC type')
    parser.add_argument('--pruning', type=str, default=None, choices=['true', 'false'], help='Enable gradient pruning')
    parser.add_argument('--save-gradients', type=str, default=None, choices=['true', 'false'], help='Enable gradient saving')
    parser.add_argument('--save-gradients-rounds', type=str, default=None, help='Comma-separated rounds to save (e.g., "1,2,3")')
    args = parser.parse_args()

    # Override settings with command-line arguments if provided
    if args.experiment:
        settings['save_results'] = f'results/{args.experiment}/'
    if args.rounds:
        settings['n_rounds'] = args.rounds
    if args.epochs:
        settings['n_epochs'] = args.epochs
    if args.clustering:
        settings['clustering'] = (args.clustering == 'true')
    if args.smpc:
        settings['type_ss'] = args.smpc
    if args.pruning:
        settings['gradient_pruning']['enabled'] = (args.pruning == 'true')
    if args.save_gradients:
        settings['save_gradients'] = (args.save_gradients == 'true')
    if args.save_gradients_rounds:
        # Parse comma-separated rounds: "1,2,3" -> [1, 2, 3]
        settings['save_gradients_rounds'] = [int(r.strip()) for r in args.save_gradients_rounds.split(',')]

    logging.basicConfig(level=logging.DEBUG)
    training_barrier, length = initialize_parameters(settings, 'CFL')

    # Create metrics tracker at the very beginning
    metrics_tracker = MetricsTracker(settings['save_results'])
    metrics_tracker.start_tracking()
    metrics_tracker.measure_global_power("start")

    # Create model manager for structured model saving
    experiment_name = os.path.basename(settings['save_results'].rstrip('/'))
    model_manager = ModelManager(experiment_name)
    
    # Initial setup phase
    metrics_tracker.measure_power(0, "initial_setup_start")

    json_dict = {
        'settings': {**settings, "length": length}
    }
    with open(settings['save_results'] + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(settings['save_results'] + "output.txt", "w") as f:
        f.write("")

    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(
        resize=length,
        name_dataset=settings['name_dataset'],
        data_root=settings['data_root'],
        number_of_clients_per_node=settings.get('number_of_clients_per_node', 6),
        number_of_nodes=settings.get('number_of_nodes', 1),
        max_samples_per_client=settings.get('max_samples_per_client', None)
    )
    
    # Create server root dataset for FLTrust if needed
    server_root_data = None
    if settings.get('aggregation', {}).get('method') == 'fltrust':
        from data.loaders import load_data_from_path, create_server_root_dataset
        
        # Load the full training dataset
        dataset_train, _ = load_data_from_path(
            resize=length, 
            name_dataset=settings['name_dataset'],
            data_root=settings['data_root']
        )
        
        # Create root dataset
        root_size = settings.get('aggregation', {}).get('fltrust_root_size', 100)
        server_root_data = create_server_root_dataset(dataset_train, root_size=root_size)
        print(f"FLTrust: Created server root dataset with {len(server_root_data[0])} samples")


    # Create the server entity
    metrics_tracker.measure_power(0, "server_creation_start")
    server = create_nodes(
        node_test_sets, 1, 
        save_results=settings['save_results'],
        check_usefulness=settings['check_usefulness'],
        coef_useful=settings['coef_useful'],
        tolerance_ceil=settings['tolerance_ceil'],
        dp=settings['diff_privacy'], 
        model_choice=settings['arch'],
        model_manager=model_manager,
        server_root_data=server_root_data,
        batch_size=settings['batch_size'],
        classes=list_classes, 
        choice_loss=settings['choice_loss'],
        choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'],
        save_figure=None, 
        matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'],
        pretrained=settings['pretrained'],
        # Legacy save_model removed
        input_size=(128, 128) if 'ffhq' in settings['name_dataset'].lower() else (32, 32)
    )[0]
    metrics_tracker.measure_power(0, "server_creation_complete")

    # Create clients
    metrics_tracker.measure_power(0, "clients_creation_start")
    node_clients = create_clients(
        client_train_sets, 
        client_test_sets, 
        0,  # node number
        settings['n_clients'],
        save_results=settings['save_results'],
        metrics_tracker=metrics_tracker,
        model_manager=model_manager,
        dp=settings['diff_privacy'], 
        model_choice=settings['arch'],
        batch_size=settings['batch_size'],
        epochs=settings['n_epochs'], 
        classes=list_classes,
        learning_rate=settings['lr'],
        choice_loss=settings['choice_loss'],
        choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'],
        step_size=settings['step_size'],
        gamma=settings['gamma'],
        save_figure=None,
        matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'],
        patience=settings['patience'],
        pretrained=settings['pretrained'],
        # Legacy save_model removed
        input_size=(128, 128) if 'ffhq' in settings['name_dataset'].lower() else (32, 32)
    )
    metrics_tracker.measure_power(0, "clients_creation_complete")

    # Setup connections
    metrics_tracker.measure_power(0, "connections_setup_start")
    for client_id, client in node_clients.items():
        client.update_node_connection(server.id, server.port)

    for client_j in node_clients.values():
        server.add_client(c_id=client_j.id, client_address=("localhost", client_j.port))
    metrics_tracker.measure_power(0, "connections_setup_complete")

    print("done with the connections")

    # Start servers
    metrics_tracker.measure_power(0, "servers_start")
    threading.Thread(target=server.start_server).start()
    for client in node_clients.values():
        threading.Thread(target=client.start_server).start()

    server.create_first_global_model()
    metrics_tracker.measure_power(0, "initial_setup_complete")
    
    time.sleep(10)

    # Training rounds
    for round_i in range(settings['n_rounds']):
        current_fl_round = round_i + 1
        print(f"### ROUND {current_fl_round} ###")
        
        if settings.get('clustering', False):
            print(f"\nROUND {current_fl_round}: Clustering is ON. Generating clusters...")
            metrics_tracker.measure_power(current_fl_round, "cluster_generation_start")
            cluster_generation([server], [node_clients], 
                              settings.get('min_number_of_clients_in_cluster', 3), 
                              1)
            metrics_tracker.measure_power(current_fl_round, "cluster_generation_complete")
            # Log cluster information
            with open(settings['save_results'] + "output.txt", "a") as f:
                f.write(f"\nROUND {current_fl_round}: Clustering is ON. Clusters formed:\n")
                for i, cluster in enumerate(server.clusters):
                    f.write(f"  Node {server.id} - Cluster {i+1}: {cluster}\n")
                    for client_id_in_cluster in cluster:
                        client_obj = node_clients.get(client_id_in_cluster)
                        if client_obj:
                            f.write(f"    Client {client_id_in_cluster} connections: {list(client_obj.connections.keys())}\n")

        with open(settings['save_results'] + "output.txt", "a") as f:
            f.write(f"### ROUND {current_fl_round} ###\n")

        # Training phase
        print(f"\nROUND {current_fl_round}: Node 1 : Starting client training phase...\n")
        metrics_tracker.measure_power(current_fl_round, "node_1_training_start")
        
        # Update client metadata for gradient saving
        for client in node_clients.values():
            if hasattr(client, '_save_gradients'):
                # Update metadata in saved gradient data
                client.dataset_name = settings['name_dataset']
                client.model_architecture = settings['arch']
                client.num_classes = len(list_classes)
                client.single_batch_training = settings.get('single_batch_training', False)
        
        threads = []
        for client in node_clients.values():
            t = threading.Thread(target=train_client, args=(client, metrics_tracker, current_fl_round))
            t.start()
            threads.append(t)
    
        for t in threads:
            t.join()
        metrics_tracker.measure_power(current_fl_round, "node_1_training_complete")

        # Wait for all clients to finish and messages to be received
        wait_time_for_node = settings['ts'] 
        print(f"\nROUND {current_fl_round}: All client training threads joined. Waiting {wait_time_for_node}s for node to receive messages...")
        time.sleep(wait_time_for_node)

        # Aggregation phase
        print(f"\nROUND {current_fl_round}: Starting aggregation phase on node...")
        metrics_tracker.measure_power(current_fl_round, "aggregation_start")
        server.create_global_model(None, current_fl_round)  # Pass current_fl_round as index
        metrics_tracker.measure_power(current_fl_round, "aggregation_complete")

        print("\nBroadcasting updated global model to clients...")
        time.sleep(settings['ts'] * 2)  # Wait for model distribution

    # Final operations
    metrics_tracker.measure_global_power("complete")
    metrics_tracker.save_metrics()
    print("\nTraining completed. Exiting program...")

    # Force exit to kill all daemon threads (server/client threads)
    # This is necessary because server.start_server() creates long-running threads
    print("✓ Shutting down all threads...")
    os._exit(0)  # Force immediate exit, killing all threads
