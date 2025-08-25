"""
Centralized model management for federated learning experiments.
Handles all model I/O operations with consistent structure and metadata.
"""

import os
import torch
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union

from .model_paths import ModelPaths
from .model_metadata import ModelMetadata, MetadataFactory


class ModelManager:
    """Centralized manager for all federated learning model artifacts"""
    
    def __init__(self, experiment_name: str, base_results_dir: str = "results"):
        """
        Initialize ModelManager
        
        Args:
            experiment_name: Name of the experiment (e.g., 'cifar10_smpc_c6_r10')
            base_results_dir: Base directory for all results
        """
        self.experiment_name = experiment_name
        self.base_dir = os.path.join(base_results_dir, experiment_name)
        self.paths = ModelPaths(self.base_dir)
        
        # Create base directory structure
        self.paths.ensure_directories()
        
    def save_client_model(self,
                         client_id: str,
                         round_num: int, 
                         model_state: Dict[str, torch.Tensor],
                         training_metrics: Dict[str, float],
                         experiment_config: Dict[str, Any],
                         gradients: Optional[List[torch.Tensor]] = None,
                         gradient_metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Save client model with standardized structure
        
        Args:
            client_id: Unique client identifier
            round_num: Training round number
            model_state: PyTorch model state dict
            training_metrics: Training performance metrics
            experiment_config: Experiment configuration
            gradients: Optional gradients for attack evaluation
            gradient_metadata: Optional gradient metadata (batch info, etc.)
            
        Returns:
            Dict with paths to saved files
        """
        # Ensure round directory exists
        self.paths.ensure_directories(round_num)
        
        # Convert model state to CPU tensors
        cpu_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in model_state.items()}
        
        # Prepare model data
        model_data = {
            'round': round_num,
            'client_id': client_id,
            'model_state': cpu_model_state,
            'timestamp': time.time(),
            'experiment_name': self.experiment_name
        }
        
        # Save model
        model_path = self.paths.get_client_model_path(round_num, client_id)
        torch.save(model_data, model_path)
        
        # Prepare privacy config from experiment config
        privacy_config = {
            'differential_privacy': experiment_config.get('diff_privacy', False),
            'dp_epsilon': experiment_config.get('epsilon', None),
            'dp_delta': experiment_config.get('delta', None),
            'dp_noise_multiplier': experiment_config.get('noise_multiplier', None),
            'smpc_enabled': experiment_config.get('clustering', False),
            'smpc_scheme': experiment_config.get('type_ss', None),
            'smpc_threshold': experiment_config.get('threshold', None),
            'clustering_enabled': experiment_config.get('clustering', False)
        }
        
        # Create and save metadata
        metadata = MetadataFactory.create_client_metadata(
            client_id=client_id,
            round_num=round_num,
            experiment_name=self.experiment_name,
            architecture=experiment_config.get('arch', 'unknown'),
            dataset=experiment_config.get('name_dataset', 'unknown'),
            num_classes=experiment_config.get('num_classes', 10),
            train_loss=training_metrics.get('train_loss', 0.0),
            train_accuracy=training_metrics.get('train_acc', 0.0),
            val_loss=training_metrics.get('val_loss', None),
            val_accuracy=training_metrics.get('val_acc', None),
            test_loss=training_metrics.get('test_loss', None),
            test_accuracy=training_metrics.get('test_acc', None),
            epochs_trained=experiment_config.get('n_epochs', 0),
            learning_rate=experiment_config.get('lr', None),
            data_size=training_metrics.get('len_train', 0),
            privacy_config=privacy_config
        )
        
        metadata_path = self.paths.get_client_metadata_path(round_num, client_id)
        metadata.save(metadata_path)
        
        saved_paths = {
            'model': model_path,
            'metadata': metadata_path
        }
        
        # Save gradients if provided
        if gradients is not None:
            gradients_data = {
                'round': round_num,
                'client_id': client_id,
                'gradients': [g.cpu() for g in gradients],
                'timestamp': time.time(),
                'experiment_name': self.experiment_name
            }
            
            if gradient_metadata:
                gradients_data.update(gradient_metadata)
            
            gradients_path = self.paths.get_client_gradients_path(round_num, client_id)
            torch.save(gradients_data, gradients_path)
            saved_paths['gradients'] = gradients_path
            
        return saved_paths
    
    def save_global_model(self,
                         node_id: str,
                         round_num: int,
                         model_state: Dict[str, torch.Tensor],
                         contributors: List[str],
                         experiment_config: Dict[str, Any],
                         aggregation_method: str = "fedavg",
                         test_metrics: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Save global aggregated model
        
        Args:
            node_id: Server/node identifier
            round_num: Training round number
            model_state: Aggregated model state dict
            contributors: List of contributing client IDs
            experiment_config: Experiment configuration
            aggregation_method: Aggregation method used
            test_metrics: Optional test performance metrics
            
        Returns:
            Dict with paths to saved files
        """
        # Ensure directory exists
        self.paths.ensure_directories(round_num)
        
        # Convert model state to CPU tensors
        cpu_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in model_state.items()}
        
        # Prepare model data
        model_data = {
            'round': round_num,
            'node_id': node_id,
            'model_state': cpu_model_state,
            'contributors': contributors,
            'aggregation_method': aggregation_method,
            'timestamp': time.time(),
            'experiment_name': self.experiment_name
        }
        
        # Save model
        model_path = self.paths.get_global_model_path(round_num, node_id)
        torch.save(model_data, model_path)
        
        # Prepare privacy config
        privacy_config = {
            'differential_privacy': experiment_config.get('diff_privacy', False),
            'smpc_enabled': experiment_config.get('clustering', False),
            'clustering_enabled': experiment_config.get('clustering', False)
        }
        
        # Create and save metadata
        metadata = MetadataFactory.create_global_metadata(
            node_id=node_id,
            round_num=round_num,
            experiment_name=self.experiment_name,
            architecture=experiment_config.get('arch', 'unknown'),
            dataset=experiment_config.get('name_dataset', 'unknown'),
            num_classes=experiment_config.get('num_classes', 10),
            contributors=contributors,
            aggregation_method=aggregation_method,
            test_loss=test_metrics.get('test_loss', None) if test_metrics else None,
            test_accuracy=test_metrics.get('test_acc', None) if test_metrics else None,
            privacy_config=privacy_config
        )
        
        metadata_path = self.paths.get_global_metadata_path(round_num, node_id)
        metadata.save(metadata_path)
        
        return {
            'model': model_path,
            'metadata': metadata_path
        }
    
    def save_cluster_model(self,
                          client_id: str,
                          cluster_id: int,
                          round_num: int,
                          model_state: Union[List[np.ndarray], Dict[str, torch.Tensor]],
                          cluster_participants: List[str],
                          experiment_config: Dict[str, Any]) -> str:
        """
        Save cluster aggregated model
        
        Args:
            client_id: ID of client representing this cluster
            cluster_id: Cluster identifier
            round_num: Training round number
            model_state: Cluster aggregated model state
            cluster_participants: List of clients in this cluster
            experiment_config: Experiment configuration
            
        Returns:
            Path to saved model file
        """
        # Ensure directory exists
        self.paths.ensure_directories(round_num)
        
        # Convert model state to consistent format
        if isinstance(model_state, list):
            # Convert from list of numpy arrays to state dict
            cpu_model_state = {}
            for i, param in enumerate(model_state):
                if isinstance(param, np.ndarray):
                    cpu_model_state[f"param_{i}"] = torch.from_numpy(param)
                else:
                    cpu_model_state[f"param_{i}"] = param
        else:
            # Already a state dict
            cpu_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                              for k, v in model_state.items()}
        
        # Prepare model data
        model_data = {
            'round': round_num,
            'client_id': client_id,
            'cluster_id': cluster_id,
            'model_state': cpu_model_state,
            'cluster_participants': cluster_participants,
            'aggregation_method': 'smpc_sum',
            'timestamp': time.time(),
            'experiment_name': self.experiment_name
        }
        
        # Save model
        model_path = self.paths.get_cluster_model_path(round_num, cluster_id, client_id)
        torch.save(model_data, model_path)
        
        # Create and save metadata  
        privacy_config = {
            'smpc_enabled': True,
            'smpc_scheme': experiment_config.get('type_ss', 'additif'),
            'smmp_threshold': experiment_config.get('threshold', 3),
            'clustering_enabled': True
        }
        
        metadata = MetadataFactory.create_cluster_metadata(
            client_id=client_id,
            cluster_id=cluster_id,
            round_num=round_num,
            experiment_name=self.experiment_name,
            architecture=experiment_config.get('arch', 'unknown'),
            dataset=experiment_config.get('name_dataset', 'unknown'),
            num_classes=experiment_config.get('num_classes', 10),
            cluster_participants=cluster_participants,
            privacy_config=privacy_config
        )
        
        # Save metadata alongside model (not separate metadata file for clusters)
        model_data['metadata'] = metadata.to_dict()
        torch.save(model_data, model_path)
        
        return model_path
    
    def save_smpc_fragments(self,
                           client_id: str,
                           round_num: int,
                           shares: List[Any],
                           experiment_config: Dict[str, Any],
                           share_metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Save SMPC fragments/shares
        
        Args:
            client_id: Client that generated the shares
            round_num: Training round number
            shares: List of secret shares
            experiment_config: Experiment configuration
            share_metadata: Optional additional metadata
            
        Returns:
            Dict with paths to saved files
        """
        # Ensure directory exists
        self.paths.ensure_directories(round_num)
        shares_dir = self.paths.get_client_shares_dir(round_num, client_id)
        os.makedirs(shares_dir, exist_ok=True)
        
        share_paths = []
        share_ids = []
        
        # Save individual shares
        for i, share_data in enumerate(shares):
            share_path = self.paths.get_fragment_share_path(round_num, client_id, i)
            
            # Convert share data to consistent format
            if isinstance(share_data, list):
                # List of tensors/arrays
                share_dict = {}
                for j, param in enumerate(share_data):
                    if isinstance(param, np.ndarray):
                        share_dict[f"param_{j}"] = torch.from_numpy(param)
                    elif isinstance(param, torch.Tensor):
                        share_dict[f"param_{j}"] = param.cpu()
                    else:
                        share_dict[f"param_{j}"] = param
            else:
                share_dict = {"share_data": share_data}
            
            share_file_data = {
                'round': round_num,
                'client_id': client_id,
                'share_index': i,
                'share_data': share_dict,
                'timestamp': time.time(),
                'experiment_name': self.experiment_name
            }
            
            if share_metadata:
                share_file_data.update(share_metadata)
                
            torch.save(share_file_data, share_path)
            share_paths.append(share_path)
            share_ids.append(i)
        
        # Create and save fragment metadata
        metadata = MetadataFactory.create_fragment_metadata(
            client_id=client_id,
            round_num=round_num,
            experiment_name=self.experiment_name,
            smpc_scheme=experiment_config.get('type_ss', 'additif'),
            num_shares=len(shares),
            threshold=experiment_config.get('threshold', 3),
            share_ids=share_ids
        )
        
        metadata_path = self.paths.get_fragment_metadata_path(round_num, client_id)
        metadata.save(metadata_path)
        
        return {
            'shares': share_paths,
            'metadata': metadata_path,
            'shares_dir': shares_dir
        }
    
    def load_client_model(self, round_num: int, client_id: str) -> Tuple[Dict[str, torch.Tensor], ModelMetadata]:
        """
        Load client model and metadata
        
        Args:
            round_num: Training round number
            client_id: Client identifier
            
        Returns:
            Tuple of (model_state_dict, metadata)
        """
        model_path = self.paths.get_client_model_path(round_num, client_id)
        metadata_path = self.paths.get_client_metadata_path(round_num, client_id)
        
        # Load model
        model_data = torch.load(model_path, map_location='cpu')
        model_state = model_data['model_state']
        
        # Load metadata
        metadata = ModelMetadata.load(metadata_path)
        
        return model_state, metadata
    
    def load_global_model(self, round_num: int, node_id: str = "n1") -> Tuple[Dict[str, torch.Tensor], ModelMetadata]:
        """
        Load global model and metadata
        
        Args:
            round_num: Training round number  
            node_id: Server/node identifier
            
        Returns:
            Tuple of (model_state_dict, metadata)
        """
        model_path = self.paths.get_global_model_path(round_num, node_id)
        metadata_path = self.paths.get_global_metadata_path(round_num, node_id)
        
        # Load model
        model_data = torch.load(model_path, map_location='cpu')
        model_state = model_data['model_state']
        
        # Load metadata
        metadata = ModelMetadata.load(metadata_path)
        
        return model_state, metadata
    
    def load_client_gradients(self, round_num: int, client_id: str) -> Dict[str, Any]:
        """
        Load client gradients
        
        Args:
            round_num: Training round number
            client_id: Client identifier
            
        Returns:
            Gradients data dictionary
        """
        gradients_path = self.paths.get_client_gradients_path(round_num, client_id)
        return torch.load(gradients_path, map_location='cpu')
    
    def list_available_models(self) -> Dict[str, Dict[int, List[str]]]:
        """
        List all available models in the experiment
        
        Returns:
            Dict mapping model_type -> round -> list of identifiers
        """
        available = {
            'clients': {},
            'global': {},
            'clusters': {}
        }
        
        # Get all rounds
        rounds = self.paths.list_rounds()
        
        for round_num in rounds:
            # Client models
            client_models = self.paths.list_client_models(round_num)
            if client_models:
                available['clients'][round_num] = list(client_models.keys())
            
            # Global models
            global_path = self.paths.get_global_model_path(round_num)
            if os.path.exists(global_path):
                available['global'][round_num] = ['n1']  # Default node ID
                
            # Cluster models
            cluster_dir = self.paths.get_cluster_models_dir(round_num)
            if os.path.exists(cluster_dir):
                cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith('.pt')]
                available['clusters'][round_num] = cluster_files
                
        return available
    
    def save_fragment(self, client_id: str, round_num: int, fragment_index: int, 
                     fragment_data: Dict[str, torch.Tensor], 
                     fragment_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Save SMPC fragment for a client
        
        Args:
            client_id: Client identifier
            round_num: Training round number
            fragment_index: Index of this fragment/share
            fragment_data: Fragment tensor data
            fragment_metadata: Additional metadata
            
        Returns:
            Dict with paths to saved files
        """
        # Ensure directories exist
        self.paths.ensure_directories(round_num)
        shares_dir = self.paths.get_client_shares_dir(round_num, client_id)
        os.makedirs(shares_dir, exist_ok=True)
        
        # Save fragment data
        fragment_path = self.paths.get_fragment_share_path(round_num, client_id, fragment_index)
        
        fragment_file_data = {
            'round': round_num,
            'client_id': client_id,
            'fragment_index': fragment_index,
            'fragment_data': fragment_data,
            'timestamp': time.time(),
            'experiment_name': self.experiment_name,
            **fragment_metadata
        }
        
        torch.save(fragment_file_data, fragment_path)
        
        return {
            'fragment': fragment_path
        }
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information and statistics
        
        Returns:
            Dict with experiment statistics
        """
        available = self.list_available_models()
        
        info = {
            'experiment_name': self.experiment_name,
            'base_directory': self.base_dir,
            'total_rounds': len(self.paths.list_rounds()),
            'rounds': self.paths.list_rounds(),
            'model_counts': {
                'clients': sum(len(clients) for clients in available['clients'].values()),
                'global': len(available['global']),
                'clusters': sum(len(clusters) for clusters in available['clusters'].values())
            }
        }
        
        # Load config if available
        config_path = self.paths.get_config_path()
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                info['config'] = json.load(f)
                
        return info