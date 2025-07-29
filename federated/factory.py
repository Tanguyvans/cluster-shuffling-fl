import os
from .client import Client
from .server import Node
from config import settings


def create_nodes(test_sets, number_of_nodes, save_results, check_usefulness, coef_useful, tolerance_ceil, **kwargs):
    list_nodes = []
    for i in range(number_of_nodes):
        list_nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=8000 + i,  # Changed base port for nodes
                test=test_sets[i],
                save_results=save_results,
                check_usefulness=check_usefulness,
                coef_useful=coef_useful,
                tolerance_ceil=tolerance_ceil,
                **kwargs
            )
        )
    return list_nodes


def create_clients(train_sets, test_sets, node, number_of_clients, save_results, metrics_tracker, **kwargs):
    dict_clients = {}
    for num_client in range(number_of_clients):
        dataset_index = node * number_of_clients + num_client
        client_id = f"c{node}_{num_client + 1}"
        
        # Create individual client model save path in .npz format
        client_model_path = os.path.join(settings['save_client_models'], f"{client_id}_best_model.npz")
        
        client_kwargs = {
            'id': client_id,
            'host': "127.0.0.1",
            'port': 9000 + (node * number_of_clients) + num_client,  # Changed port calculation
            'train': train_sets[dataset_index],
            'test': test_sets[dataset_index],
            'save_results': save_results,
            **kwargs
        }
        
        # Override save_model with client-specific .npz path
        client_kwargs['save_model'] = client_model_path
        
        # Only add secret sharing parameters if they exist in settings
        if 'secret_sharing' in settings:
            client_kwargs.update({
                'type_ss': settings['secret_sharing'],
                'threshold': settings['k'],
                'm': settings['m']
            })

        # Add DP parameters from settings
        if settings.get('diff_privacy', False):
            client_kwargs.update({
                'epsilon': settings.get('dp_epsilon', 1.0),
                'delta': settings.get('dp_delta', 1e-5),
                'max_grad_norm': settings.get('dp_max_grad_norm', 1.2),
                'noise_multiplier': settings.get('dp_noise_multiplier', 1.0)
            })
        
        # Create the client
        client = Client(**client_kwargs)
        # Set metrics_tracker after client creation
        client.metrics_tracker = metrics_tracker
        
        dict_clients[client_id] = client

    return dict_clients


def cluster_generation(nodes, clients, min_number_of_clients_in_cluster, number_of_nodes):
    """Generate new clusters for each node"""
    if min_number_of_clients_in_cluster is None:
        min_number_of_clients_in_cluster = 3  # Valeur par défaut
        
    for num_node in range(number_of_nodes):
        # Reset all client connections
        for client in clients[num_node].values():
            client.reset_connections()
        
        # Generate new clusters
        nodes[num_node].generate_clusters(min_number_of_clients_in_cluster)
        
        # Set up connections within each cluster
        for cluster in nodes[num_node].clusters:
            for client_id_1 in cluster:
                if client_id_1 in ['tot', 'count']:
                    continue
                for client_id_2 in cluster:
                    if client_id_2 in ['tot', 'count'] or client_id_1 == client_id_2:
                        continue
                    clients[num_node][client_id_1].add_connection(
                        client_id_2, 
                        clients[num_node][client_id_2].port
                    )