"""
Model path configuration and management for federated learning experiments.
Provides centralized, consistent path generation for all model types.
"""

import os
from typing import Dict, Optional


class ModelPaths:
    """Centralized path management for federated learning model artifacts"""
    
    def __init__(self, base_dir: str):
        """
        Initialize path manager with experiment base directory
        
        Args:
            base_dir: Base experiment directory (e.g., 'results/cifar10_smpc_c6_r10')
        """
        self.base_dir = base_dir
        self._structure = {
            'models': 'models',
            'clients': 'models/clients',
            'global': 'models/global', 
            'clusters': 'models/clusters',
            'fragments': 'models/fragments',
            'logs': 'logs',
            'metrics': 'metrics'
        }
        
    def get_base_dir(self) -> str:
        """Get the experiment base directory"""
        return self.base_dir
        
    def get_models_dir(self) -> str:
        """Get the main models directory"""
        return os.path.join(self.base_dir, self._structure['models'])
        
    def get_client_models_dir(self, round_num: int) -> str:
        """Get client models directory for a specific round"""
        return os.path.join(
            self.base_dir, 
            self._structure['clients'], 
            f"round_{round_num:03d}"
        )
        
    def get_global_models_dir(self) -> str:
        """Get global models directory"""
        return os.path.join(self.base_dir, self._structure['global'])
        
    def get_cluster_models_dir(self, round_num: int) -> str:
        """Get cluster models directory for a specific round"""
        return os.path.join(
            self.base_dir,
            self._structure['clusters'],
            f"round_{round_num:03d}"
        )
        
    def get_fragments_dir(self, round_num: int) -> str:
        """Get SMPC fragments base directory for a specific round"""
        return os.path.join(
            self.base_dir,
            self._structure['fragments'], 
            f"round_{round_num:03d}"
        )
        
    def get_client_shares_dir(self, round_num: int, client_id: str) -> str:
        """Get specific client's SMPC shares directory"""
        return os.path.join(
            self.get_fragments_dir(round_num),
            f"{client_id}_shares"
        )
        
    def get_logs_dir(self) -> str:
        """Get logs directory"""
        return os.path.join(self.base_dir, self._structure['logs'])
        
    def get_metrics_dir(self) -> str:
        """Get metrics directory"""
        return os.path.join(self.base_dir, self._structure['metrics'])
        
    # Model file paths
    def get_client_model_path(self, round_num: int, client_id: str) -> str:
        """Get client model file path"""
        return os.path.join(
            self.get_client_models_dir(round_num),
            f"{client_id}_model.pt"
        )
        
    def get_client_gradients_path(self, round_num: int, client_id: str) -> str:
        """Get client gradients file path"""
        return os.path.join(
            self.get_client_models_dir(round_num),
            f"{client_id}_gradients.pt"
        )
        
    def get_client_metadata_path(self, round_num: int, client_id: str) -> str:
        """Get client metadata file path"""
        return os.path.join(
            self.get_client_models_dir(round_num),
            f"{client_id}_metadata.json"
        )
        
    def get_global_model_path(self, round_num: int, node_id: str = "n1") -> str:
        """Get global model file path"""
        return os.path.join(
            self.get_global_models_dir(),
            f"round_{round_num:03d}_global.pt"
        )
        
    def get_global_metadata_path(self, round_num: int, node_id: str = "n1") -> str:
        """Get global model metadata file path"""
        return os.path.join(
            self.get_global_models_dir(),
            f"round_{round_num:03d}_metadata.json"
        )
        
    def get_cluster_model_path(self, round_num: int, cluster_id: int, client_id: str) -> str:
        """Get cluster model file path"""
        return os.path.join(
            self.get_cluster_models_dir(round_num),
            f"cluster_{cluster_id}_{client_id}.pt"
        )
        
    def get_fragment_share_path(self, round_num: int, client_id: str, share_id: int) -> str:
        """Get SMPC fragment share file path"""
        return os.path.join(
            self.get_client_shares_dir(round_num, client_id),
            f"share_{share_id}.pt"
        )
        
    def get_fragment_metadata_path(self, round_num: int, client_id: str) -> str:
        """Get SMPC fragment metadata file path"""
        return os.path.join(
            self.get_client_shares_dir(round_num, client_id),
            "metadata.json"
        )
        
    def get_config_path(self) -> str:
        """Get experiment configuration file path"""
        return os.path.join(self.base_dir, "config.json")
        
    def ensure_directories(self, round_num: Optional[int] = None):
        """
        Create all necessary directories for the experiment
        
        Args:
            round_num: If provided, create round-specific directories
        """
        # Base directories
        dirs_to_create = [
            self.get_models_dir(),
            self.get_global_models_dir(),
            self.get_logs_dir(),
            self.get_metrics_dir()
        ]
        
        # Round-specific directories
        if round_num is not None:
            dirs_to_create.extend([
                self.get_client_models_dir(round_num),
                self.get_cluster_models_dir(round_num),
                self.get_fragments_dir(round_num)
            ])
            
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            
    def list_client_models(self, round_num: int) -> Dict[str, Dict[str, str]]:
        """
        List all client models for a specific round
        
        Returns:
            Dict mapping client_id to paths dict
        """
        models = {}
        client_dir = self.get_client_models_dir(round_num)
        
        if os.path.exists(client_dir):
            for filename in os.listdir(client_dir):
                if filename.endswith('_model.pt'):
                    client_id = filename.replace('_model.pt', '')
                    models[client_id] = {
                        'model': self.get_client_model_path(round_num, client_id),
                        'metadata': self.get_client_metadata_path(round_num, client_id),
                        'gradients': self.get_client_gradients_path(round_num, client_id)
                    }
                    
        return models
        
    def list_rounds(self) -> list[int]:
        """List all available rounds in the experiment"""
        rounds = []
        client_base = os.path.join(self.base_dir, self._structure['clients'])
        
        if os.path.exists(client_base):
            for dirname in os.listdir(client_base):
                if dirname.startswith('round_'):
                    try:
                        round_num = int(dirname.split('_')[1])
                        rounds.append(round_num)
                    except (IndexError, ValueError):
                        continue
                        
        return sorted(rounds)