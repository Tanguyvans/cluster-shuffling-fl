"""
Standardized metadata structures for federated learning model artifacts.
Ensures consistent metadata format across all model types.
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json


@dataclass
class ModelInfo:
    """Basic model information"""
    model_type: str  # 'client', 'global', 'cluster', 'fragment'
    round: int
    timestamp: float
    architecture: str
    dataset: str
    num_classes: int
    experiment_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class TrainingMetrics:
    """Training performance metrics"""
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    epochs_trained: int = 0
    learning_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PrivacyInfo:
    """Privacy-preserving technique information"""
    differential_privacy: bool = False
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    dp_noise_multiplier: Optional[float] = None
    smpc_enabled: bool = False
    smpc_scheme: Optional[str] = None  # 'additif', 'shamir'
    smpc_threshold: Optional[int] = None
    clustering_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Provenance:
    """Model provenance and lineage information"""
    created_by: str  # client_id, node_id, etc.
    parent_models: Optional[List[str]] = None  # IDs of models this was derived from
    aggregation_method: Optional[str] = None
    data_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelMetadata:
    """Standardized metadata container for all model types"""
    
    def __init__(self, 
                 model_info: ModelInfo,
                 training_metrics: TrainingMetrics,
                 privacy_info: PrivacyInfo,
                 provenance: Provenance,
                 custom_fields: Optional[Dict[str, Any]] = None):
        self.model_info = model_info
        self.training_metrics = training_metrics
        self.privacy_info = privacy_info
        self.provenance = provenance
        self.custom_fields = custom_fields or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        metadata = {
            'model_info': self.model_info.to_dict(),
            'training_metrics': self.training_metrics.to_dict(),
            'privacy_info': self.privacy_info.to_dict(),
            'provenance': self.provenance.to_dict()
        }
        
        if self.custom_fields:
            metadata['custom_fields'] = self.custom_fields
            
        return metadata
    
    def to_json(self, indent: int = 2) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
        
    def save(self, filepath: str):
        """Save metadata to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
            
    @classmethod
    def load(cls, filepath: str) -> 'ModelMetadata':
        """Load metadata from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        model_info = ModelInfo(**data['model_info'])
        training_metrics = TrainingMetrics(**data['training_metrics'])
        privacy_info = PrivacyInfo(**data['privacy_info'])
        provenance = Provenance(**data['provenance'])
        custom_fields = data.get('custom_fields', {})
        
        return cls(model_info, training_metrics, privacy_info, provenance, custom_fields)


class MetadataFactory:
    """Factory class for creating standardized metadata"""
    
    @staticmethod
    def create_client_metadata(client_id: str,
                              round_num: int, 
                              experiment_name: str,
                              architecture: str,
                              dataset: str,
                              num_classes: int,
                              train_loss: float,
                              train_accuracy: float,
                              val_loss: Optional[float] = None,
                              val_accuracy: Optional[float] = None,
                              test_loss: Optional[float] = None,
                              test_accuracy: Optional[float] = None,
                              epochs_trained: int = 0,
                              learning_rate: Optional[float] = None,
                              data_size: int = 0,
                              privacy_config: Optional[Dict] = None,
                              custom_fields: Optional[Dict] = None) -> ModelMetadata:
        """Create metadata for client model"""
        
        model_info = ModelInfo(
            model_type='client',
            round=round_num,
            timestamp=time.time(),
            architecture=architecture,
            dataset=dataset,
            num_classes=num_classes,
            experiment_name=experiment_name
        )
        
        training_metrics = TrainingMetrics(
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            epochs_trained=epochs_trained,
            learning_rate=learning_rate
        )
        
        privacy_info = PrivacyInfo()
        if privacy_config:
            for key, value in privacy_config.items():
                if hasattr(privacy_info, key):
                    setattr(privacy_info, key, value)
                    
        provenance = Provenance(
            created_by=client_id,
            data_size=data_size
        )
        
        return ModelMetadata(model_info, training_metrics, privacy_info, provenance, custom_fields)
    
    @staticmethod 
    def create_global_metadata(node_id: str,
                              round_num: int,
                              experiment_name: str,
                              architecture: str,
                              dataset: str,
                              num_classes: int,
                              contributors: List[str],
                              aggregation_method: str = "fedavg",
                              test_loss: Optional[float] = None,
                              test_accuracy: Optional[float] = None,
                              privacy_config: Optional[Dict] = None,
                              custom_fields: Optional[Dict] = None) -> ModelMetadata:
        """Create metadata for global model"""
        
        model_info = ModelInfo(
            model_type='global',
            round=round_num,
            timestamp=time.time(),
            architecture=architecture,
            dataset=dataset,
            num_classes=num_classes,
            experiment_name=experiment_name
        )
        
        training_metrics = TrainingMetrics(
            train_loss=0.0,  # Global model doesn't train directly
            train_accuracy=0.0,
            test_loss=test_loss,
            test_accuracy=test_accuracy
        )
        
        privacy_info = PrivacyInfo()
        if privacy_config:
            for key, value in privacy_config.items():
                if hasattr(privacy_info, key):
                    setattr(privacy_info, key, value)
                    
        provenance = Provenance(
            created_by=node_id,
            parent_models=contributors,
            aggregation_method=aggregation_method,
            data_size=len(contributors)
        )
        
        return ModelMetadata(model_info, training_metrics, privacy_info, provenance, custom_fields)
    
    @staticmethod
    def create_cluster_metadata(client_id: str,
                               cluster_id: int,
                               round_num: int,
                               experiment_name: str,
                               architecture: str,
                               dataset: str,
                               num_classes: int,
                               cluster_participants: List[str],
                               privacy_config: Optional[Dict] = None,
                               custom_fields: Optional[Dict] = None) -> ModelMetadata:
        """Create metadata for cluster model"""
        
        model_info = ModelInfo(
            model_type='cluster',
            round=round_num,
            timestamp=time.time(),
            architecture=architecture,
            dataset=dataset,
            num_classes=num_classes,
            experiment_name=experiment_name
        )
        
        training_metrics = TrainingMetrics(
            train_loss=0.0,  # Cluster model is aggregated
            train_accuracy=0.0
        )
        
        privacy_info = PrivacyInfo()
        if privacy_config:
            for key, value in privacy_config.items():
                if hasattr(privacy_info, key):
                    setattr(privacy_info, key, value)
                    
        provenance = Provenance(
            created_by=client_id,
            parent_models=cluster_participants,
            aggregation_method="smpc_sum",
            data_size=len(cluster_participants)
        )
        
        if custom_fields is None:
            custom_fields = {}
        custom_fields['cluster_id'] = cluster_id
        custom_fields['cluster_size'] = len(cluster_participants)
        
        return ModelMetadata(model_info, training_metrics, privacy_info, provenance, custom_fields)
    
    @staticmethod
    def create_fragment_metadata(client_id: str,
                                round_num: int,
                                experiment_name: str,
                                smpc_scheme: str,
                                num_shares: int,
                                threshold: int,
                                share_ids: List[int],
                                custom_fields: Optional[Dict] = None) -> ModelMetadata:
        """Create metadata for SMPC fragments"""
        
        model_info = ModelInfo(
            model_type='fragment',
            round=round_num,
            timestamp=time.time(),
            architecture='N/A',  # Fragments don't have architecture
            dataset='N/A',
            num_classes=0,
            experiment_name=experiment_name
        )
        
        training_metrics = TrainingMetrics(
            train_loss=0.0,
            train_accuracy=0.0
        )
        
        privacy_info = PrivacyInfo(
            smpc_enabled=True,
            smpc_scheme=smpc_scheme,
            smpc_threshold=threshold
        )
        
        provenance = Provenance(
            created_by=client_id,
            data_size=num_shares
        )
        
        if custom_fields is None:
            custom_fields = {}
        custom_fields.update({
            'smpc_scheme': smpc_scheme,
            'num_shares': num_shares,
            'threshold': threshold,
            'share_ids': share_ids
        })
        
        return ModelMetadata(model_info, training_metrics, privacy_info, provenance, custom_fields)