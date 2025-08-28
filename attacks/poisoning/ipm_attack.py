import torch
import numpy as np
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack
from .attack_factory import AttackFactory


class IPMAttack(BasePoisoningAttack):
    """
    Inner Product Manipulation (IPM) attack.
    
    This attack manipulates the inner product between gradients to disrupt
    aggregation while remaining undetected. Based on "Breaking byzantine-tolerant 
    SGD by inner product manipulation".
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.manipulation_strategy = config.get('manipulation_strategy', 'maximize_distance')
        self.target_level = config.get('target_level', 'client')  # client, cluster, global, adaptive
        self.target_client = config.get('target_client', None)  # Client to manipulate against
        self.aggregation_method = config.get('aggregation_method', 'fedavg')
        self.lambda_param = config.get('lambda_param', self.attack_intensity)  # Manipulation strength
        self.cluster_awareness = config.get('cluster_awareness', True)
        self.adaptive_scaling = config.get('adaptive_scaling', True)
        
        # Data storage for different targeting strategies
        self.benign_updates = {}  # Store benign client updates for analysis
        self.gradient_history = []  # Store gradient history for better benign estimation
        self.cluster_info = {}  # Store cluster membership information
        self.global_context = {}  # Store global aggregation context
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IPM primarily works at gradient level.
        
        Returns:
            Original data and labels unchanged
        """
        return data, labels
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply IPM attack following BLADES implementation.
        
        IPM (Inverse Projected Mean): Replace malicious gradients with 
        the negative scaled mean of estimated benign gradients.
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of IPM-manipulated gradient tensors
        """
        if not self.should_attack():
            return gradients
        
        # Store gradients for history-based estimation (before manipulation)
        if len(self.gradient_history) < 5:  # Keep last 5 gradient updates
            # Store flattened version of all gradients combined for better estimation
            flattened_grads = []
            for grad_tensor in gradients.values():
                flattened_grads.append(grad_tensor.flatten())
            combined_grad = torch.cat(flattened_grads)
            self.gradient_history.append(combined_grad.clone().detach())
        else:
            # Remove oldest and add newest
            self.gradient_history.pop(0)
            flattened_grads = []
            for grad_tensor in gradients.values():
                flattened_grads.append(grad_tensor.flatten())
            combined_grad = torch.cat(flattened_grads)
            self.gradient_history.append(combined_grad.clone().detach())
            
        # BLADES-style IPM: Replace gradients with inverted mean
        poisoned_gradients = {}
        
        for param_name, grad_tensor in gradients.items():
            # Apply IPM manipulation: -scale * estimated_benign_mean
            poisoned_gradients[param_name] = self._apply_imp_manipulation(grad_tensor)
                
        return poisoned_gradients
        
    def _apply_imp_manipulation(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply true BLADES IPM (Inner Product Manipulation) attack.
        
        True BLADES IPM Algorithm:
        1. Get mean of all benign clients' gradients 
        2. Return -λ * mean(benign_gradients) where λ = attack_intensity
        
        This matches the BLADES repository implementation exactly.
        """
        # Try to get benign gradients from saved files
        benign_mean = self._get_benign_gradients_mean(gradient)
        
        # Standard scaling: Use attack intensity as lambda parameter
        lambda_param = self.attack_intensity  # 0.2 from config
        
        # True BLADES IPM transformation: -λ * mean(benign_gradients)
        manipulated_gradient = -lambda_param * benign_mean
            
        return manipulated_gradient
        
    def _get_benign_gradients_mean(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Get mean of benign clients' gradients from saved gradient files.
        Now supports cross-cluster targeting based on target_level configuration.
        """
        import os
        import glob
        import time
        
        # Check target level to determine attack strategy
        if self.target_level == "cross_cluster":
            return self._get_cross_cluster_target(current_gradient)
        else:
            return self._get_standard_benign_mean(current_gradient)
    
    def _get_cross_cluster_target(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Cross-cluster IPM: Wait for other cluster gradients and target them.
        This is the sophisticated attack that waits for other clusters.
        """
        import os
        import glob
        import time
        
        print("[IPM] 🎯 Cross-cluster targeting mode activated")
        print("[IPM] ⏳ Waiting for other cluster to generate gradients...")
        
        # Wait up to 10 seconds for other cluster gradients
        wait_timeout = 10
        start_time = time.time()
        check_interval = 0.5
        
        while (time.time() - start_time) < wait_timeout:
            # Look for recently saved gradients from other clients (other clusters)
            gradient_files = glob.glob("results/*/models/clients/round_*/*_gradients.pt")
            
            # Group gradients by potential clusters (exclude c0_1 - the attacker)
            other_cluster_gradients = []
            
            for gradient_file in gradient_files[-20:]:  # Check most recent files
                try:
                    client_id = os.path.basename(gradient_file).split("_gradients.pt")[0]
                    
                    if client_id == "c0_1":  # Skip attacking client
                        continue
                    
                    # Load gradient data
                    gradient_data = torch.load(gradient_file, map_location='cpu')
                    if isinstance(gradient_data, dict) and 'gradients' in gradient_data:
                        client_gradients = gradient_data['gradients']
                        
                        # Find matching gradient tensor
                        for grad_tensor in client_gradients:
                            if isinstance(grad_tensor, torch.Tensor) and grad_tensor.shape == current_gradient.shape:
                                other_cluster_gradients.append(grad_tensor)
                                break
                                
                except Exception:
                    continue
            
            # If we have enough gradients from other cluster(s), use them
            if len(other_cluster_gradients) >= 2:
                # Calculate mean of other cluster gradients
                other_cluster_mean = torch.stack(other_cluster_gradients[-3:]).mean(dim=0)  # Use last 3
                
                # Scale up to compensate for cluster dilution
                scale_factor = 3.0  # Overcome averaging within our cluster
                scaled_target = scale_factor * other_cluster_mean
                
                # Detailed cross-cluster attack logging
                print(f"\n[IPM] 🎯 CROSS-CLUSTER ATTACK SUCCESSFUL!")
                print(f"[IPM] 📊 Target Analysis:")
                print(f"  └─ Other cluster gradient count: {len(other_cluster_gradients[-3:])}")
                print(f"  └─ Target cluster mean norm: {torch.norm(other_cluster_mean).item():.6f}")
                print(f"  └─ After scaling (x{scale_factor}): {torch.norm(scaled_target).item():.6f}")
                print(f"[IPM] ⚡ Attack Strategy:")
                print(f"  └─ Formula: attack_grad = -λ * scale * mean(other_cluster_grads)")
                print(f"  └─ Scale factor: {scale_factor} (compensates for our cluster dilution)")
                print(f"  └─ Attack intensity (λ): {self.lambda_param}")
                print(f"  └─ Expected cluster cancellation: ~{self.lambda_param * scale_factor:.1f}x other cluster impact")
                
                # Store cross-cluster attack info for JSON logging
                if not hasattr(self, 'cross_cluster_attack_info'):
                    self.cross_cluster_attack_info = []
                
                attack_info = {
                    'attack_type': 'cross_cluster_ipm',
                    'target_gradient_count': len(other_cluster_gradients[-3:]),
                    'target_norm': torch.norm(other_cluster_mean).item(),
                    'scale_factor': scale_factor,
                    'scaled_target_norm': torch.norm(scaled_target).item(),
                    'attack_intensity': self.lambda_param,
                    'expected_cancellation_ratio': self.lambda_param * scale_factor,
                    'parameter_shape': str(other_cluster_mean.shape)
                }
                self.cross_cluster_attack_info.append(attack_info)
                
                return scaled_target
                
            time.sleep(check_interval)
        
        print(f"[IPM] ⏰ Cross-cluster timeout after {wait_timeout}s, falling back to standard IPM")
        return self._get_standard_benign_mean(current_gradient)
    
    def _get_standard_benign_mean(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Standard BLADES IPM: Get mean of all available benign gradients.
        """
        import os
        import glob
        
        # Try to find saved gradient files in results directory
        results_dirs = glob.glob("results/*/models/clients/round_*")
        
        benign_gradients = []
        current_round = None
        
        # Find the most recent round with gradient files
        for results_dir in sorted(results_dirs, reverse=True):
            gradient_files = glob.glob(os.path.join(results_dir, "*_gradients.pt"))
            if gradient_files:
                # Extract round number from path
                round_part = results_dir.split("/")[-1]  # round_XXX
                if round_part.startswith("round_"):
                    current_round = int(round_part.split("_")[1])
                    break
        
        if current_round is None:
            print("[IPM] No saved gradients found, using current gradient as fallback")
            return current_gradient
            
        # Load all gradient files from the most recent round
        gradient_files = glob.glob(f"results/*/models/clients/round_{current_round:03d}/*_gradients.pt")
        
        for gradient_file in gradient_files:
            try:
                # Extract client ID from filename
                client_id = os.path.basename(gradient_file).split("_gradients.pt")[0]
                
                # Skip malicious clients (assume attacking client is malicious)
                # In your config, malicious client is c0_1
                if client_id in ["c0_1"]:  # Skip malicious client
                    continue
                    
                # Load gradient data
                gradient_data = torch.load(gradient_file, map_location='cpu')
                if isinstance(gradient_data, dict) and 'gradients' in gradient_data:
                    client_gradients = gradient_data['gradients']
                    
                    # Find the gradient tensor that matches current gradient shape
                    for grad_tensor in client_gradients:
                        if isinstance(grad_tensor, torch.Tensor) and grad_tensor.shape == current_gradient.shape:
                            benign_gradients.append(grad_tensor)
                            print(f"[IPM] Loaded gradient from benign client {client_id} (shape: {grad_tensor.shape})")
                            break
                            
            except Exception as e:
                print(f"[IPM] Failed to load gradient file {gradient_file}: {e}")
                continue
        
        if benign_gradients:
            # Calculate mean of all benign gradients
            benign_mean = torch.stack(benign_gradients).mean(dim=0)
            print(f"[IPM] Using mean of {len(benign_gradients)} benign gradients for true BLADES IPM")
            return benign_mean
        else:
            print("[IPM] No compatible benign gradients found, using current gradient as fallback")
            return current_gradient
    
    def _get_target_benign_estimate(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Estimate benign gradients based on the configured target level.
        """
        if self.target_level == 'client':
            return self._estimate_client_level_benign(gradient)
        elif self.target_level == 'cluster':
            return self._estimate_cluster_level_benign(gradient)
        elif self.target_level == 'global':
            return self._estimate_global_level_benign(gradient)
        elif self.target_level == 'adaptive':
            return self._estimate_adaptive_benign(gradient)
        else:
            # Fallback to client level
            return self._estimate_client_level_benign(gradient)
    
    def _estimate_client_level_benign(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Client-level targeting: Estimate individual client's benign behavior.
        """
        # Use current gradient as estimate (assumes similar data distribution)
        estimated_mean = gradient
        
        # Enhance with gradient history if available
        if len(self.gradient_history) > 0:
            recent_grads = self.gradient_history[-2:]  # Last 2 flattened gradients
            if len(recent_grads) > 0:
                try:
                    # Current gradient flattened
                    current_flat = gradient.flatten()
                    # Average with recent history
                    history_avg = torch.stack(recent_grads).mean(dim=0)
                    
                    # Ensure sizes match - take minimum size to avoid out-of-bounds
                    min_size = min(current_flat.numel(), history_avg.numel())
                    current_portion = current_flat[:min_size]
                    history_portion = history_avg[:min_size]
                    
                    combined_flat = (current_portion + history_portion) / 2.0
                    # Reshape back to original gradient shape using current gradient data
                    estimated_mean = gradient.clone()
                    estimated_mean.flatten()[:min_size] = combined_flat
                except Exception as e:
                    # Fallback to current gradient if history processing fails
                    print(f"[IPM] History processing failed: {e}, using current gradient")
                    estimated_mean = gradient
                
        return estimated_mean
    
    def _estimate_cluster_level_benign(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Cluster-level targeting: Estimate cluster's collective benign behavior.
        
        Note: In a real cluster attack, the malicious client would need access to 
        other cluster members' gradients or aggregated cluster updates. Since we
        don't have direct access, we use historical patterns and assumptions.
        """
        # Check if clustering is actually enabled
        if not self.cluster_awareness:
            print("[IPM] Cluster targeting requested but clustering disabled - using client-level")
            return self._estimate_client_level_benign(gradient)
            
        # If we have actual cluster context from the FL system, use it
        if self.cluster_info and 'cluster_aggregate' in self.cluster_info:
            # Use actual cluster aggregation information if available
            cluster_aggregate = self.cluster_info['cluster_aggregate']
            estimated_mean = cluster_aggregate
            
        # If we have benign updates from other cluster members
        elif self.benign_updates and len(self.benign_updates) > 0:
            # Estimate cluster direction from available benign updates
            benign_grads = []
            for client_id, client_updates in self.benign_updates.items():
                if 'gradients' in client_updates:
                    # Flatten and combine all gradients from this client
                    client_grad_flat = []
                    for param_grad in client_updates['gradients'].values():
                        client_grad_flat.append(param_grad.flatten())
                    benign_grads.append(torch.cat(client_grad_flat))
            
            if benign_grads:
                # Average benign cluster members' gradients
                cluster_benign_avg = torch.stack(benign_grads).mean(dim=0)
                # Extract relevant portion and reshape
                min_size = min(gradient.numel(), cluster_benign_avg.numel())
                estimated_mean = gradient.clone()
                estimated_mean.flatten()[:min_size] = cluster_benign_avg[:min_size]
            else:
                estimated_mean = gradient  # Fallback
                
        # Use gradient history to estimate cluster's learning trajectory  
        elif len(self.gradient_history) >= 3:
            try:
                # Assume cluster has consistent learning direction over time
                recent_grads = self.gradient_history[-3:]  # Last 3 flattened gradients
                cluster_trend_flat = torch.stack(recent_grads).mean(dim=0)
                
                # DON'T use current gradient (which is from attacking client)
                # Instead, estimate what benign cluster members would produce
                estimated_mean = gradient.clone()
                min_size = min(gradient.numel(), cluster_trend_flat.numel())
                
                # Use ONLY historical trend, not current attacking client's gradient
                estimated_mean.flatten()[:min_size] = cluster_trend_flat[:min_size]
                
            except Exception as e:
                print(f"[IPM] Cluster history processing failed: {e}, using historical average")
                estimated_mean = gradient
        else:
            # No cluster information available - assume other cluster members 
            # behave similarly to this client's past behavior
            print("[IPM] No cluster context - estimating from client history")
            estimated_mean = gradient
            
        return estimated_mean
    
    def _estimate_global_level_benign(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Global-level targeting: Estimate global model's expected learning direction.
        """
        estimated_mean = gradient
        
        # Use longer history for global trend estimation
        if len(self.gradient_history) >= 5:
            try:
                # Global learning should follow consistent long-term trend
                long_term_grads = self.gradient_history[-5:]  # Last 5 flattened gradients
                global_trend_flat = torch.stack(long_term_grads).mean(dim=0)
                
                # Extract relevant portion for current gradient and reshape
                current_flat = gradient.flatten()
                min_size = min(current_flat.numel(), global_trend_flat.numel())
                
                # Create global portion by taking relevant part
                estimated_mean = gradient.clone()
                global_portion_flat = global_trend_flat[:min_size]
                current_portion_flat = current_flat[:min_size]
                combined_flat = 0.5 * current_portion_flat + 0.5 * global_portion_flat
                
                estimated_mean.flatten()[:min_size] = combined_flat
            except Exception as e:
                print(f"[IPM] Global history processing failed: {e}, using current gradient")
                estimated_mean = gradient
            
        # If we have global context information
        if self.global_context and 'global_learning_direction' in self.global_context:
            global_direction = self.global_context['global_learning_direction']
            estimated_mean = 0.4 * gradient + 0.6 * global_direction
            
        return estimated_mean
    
    def _estimate_adaptive_benign(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Adaptive targeting: Choose the best estimation strategy dynamically.
        """
        # Decide strategy based on available information
        if self.cluster_info and self.cluster_awareness:
            return self._estimate_cluster_level_benign(gradient)
        elif self.global_context:
            return self._estimate_global_level_benign(gradient)
        else:
            return self._estimate_client_level_benign(gradient)
    
    def _calculate_target_scale_factor(self, gradient: torch.Tensor) -> float:
        """
        Calculate scaling factor based on target level and configuration.
        """
        base_scale = max(0.8, min(1.2, self.lambda_param * 5))  # Base BLADES scaling
        
        if not self.adaptive_scaling:
            return base_scale
            
        # Adjust scaling based on target level
        if self.target_level == 'client':
            # Client-level: moderate scaling for stealth
            return base_scale * 0.9
        elif self.target_level == 'cluster':
            # Cluster-level: stronger scaling to disrupt cluster aggregation
            return base_scale * 1.1
        elif self.target_level == 'global':
            # Global-level: maximum scaling for global disruption
            return base_scale * 1.2
        elif self.target_level == 'adaptive':
            # Adaptive: scale based on confidence in estimate
            confidence = len(self.gradient_history) / 10.0  # 0-1 confidence
            return base_scale * (1.0 + confidence * 0.2)
        else:
            return base_scale
    
    def _estimate_benign_mean(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Estimate the mean of benign gradients using available information.
        
        In real FL, we can't access other clients' gradients directly,
        so we use the current gradient as our best estimate of the
        learning direction that benign clients would follow.
        """
        # Strategy 1: Use current gradient as benign estimate
        # This assumes malicious client started with similar data distribution
        estimated_mean = current_gradient
        
        # Strategy 2: Could enhance with historical gradients if available
        if hasattr(self, 'gradient_history') and len(self.gradient_history) > 0:
            # Average recent gradients to get better estimate
            recent_grads = self.gradient_history[-3:]  # Last 3 gradients
            stacked_grads = torch.stack([current_gradient] + recent_grads)
            estimated_mean = torch.mean(stacked_grads, dim=0)
            
        return estimated_mean
        
    def _manipulate_inner_product(self, gradient: torch.Tensor, param_name: str,
                                 round_info: Dict[str, Any]) -> torch.Tensor:
        """Manipulate gradient to affect inner product with other clients."""
        
        if self.manipulation_strategy == 'maximize_distance':
            return self._maximize_distance_manipulation(gradient, param_name)
        elif self.manipulation_strategy == 'minimize_similarity':
            return self._minimize_similarity_manipulation(gradient, param_name)
        elif self.manipulation_strategy == 'direction_flip':
            return self._direction_flip_manipulation(gradient, param_name)
        elif self.manipulation_strategy == 'orthogonal_projection':
            return self._orthogonal_projection_manipulation(gradient, param_name)
        else:
            raise ValueError(f"Unknown manipulation_strategy: {self.manipulation_strategy}")
            
    def _maximize_distance_manipulation(self, gradient: torch.Tensor, param_name: str) -> torch.Tensor:
        """Maximize distance from average benign gradients."""
        if not self.benign_updates:
            # No benign updates available, apply default manipulation
            return gradient - self.lambda_param * gradient
            
        # Compute average benign gradient for this parameter
        benign_grads = []
        for client_updates in self.benign_updates.values():
            if param_name in client_updates:
                benign_grads.append(client_updates[param_name])
                
        if not benign_grads:
            return gradient
            
        avg_benign_grad = torch.stack(benign_grads).mean(dim=0)
        
        # Move gradient away from benign average
        direction_away = gradient - avg_benign_grad
        direction_away = direction_away / (torch.norm(direction_away) + 1e-8)  # Normalize
        
        # Apply manipulation
        manipulated_gradient = gradient + self.lambda_param * direction_away * torch.norm(gradient)
        
        return manipulated_gradient
        
    def _minimize_similarity_manipulation(self, gradient: torch.Tensor, param_name: str) -> torch.Tensor:
        """Minimize cosine similarity with benign gradients."""
        if not self.benign_updates:
            return gradient
            
        # Compute target direction to minimize similarity
        benign_grads = []
        for client_updates in self.benign_updates.values():
            if param_name in client_updates:
                benign_grads.append(client_updates[param_name])
                
        if not benign_grads:
            return gradient
            
        # Find direction that minimizes similarity (orthogonal or opposite)
        avg_benign_grad = torch.stack(benign_grads).mean(dim=0)
        
        # Create orthogonal component
        gradient_flat = gradient.flatten()
        benign_flat = avg_benign_grad.flatten()
        
        # Project gradient onto benign direction
        dot_product = torch.dot(gradient_flat, benign_flat)
        benign_norm_sq = torch.dot(benign_flat, benign_flat)
        
        if benign_norm_sq > 1e-8:
            projection = (dot_product / benign_norm_sq) * benign_flat
            orthogonal_component = gradient_flat - projection
            
            # Enhance orthogonal component
            enhanced_orthogonal = orthogonal_component * (1 + self.lambda_param)
            
            manipulated_gradient = (projection + enhanced_orthogonal).reshape(gradient.shape)
        else:
            manipulated_gradient = gradient
            
        return manipulated_gradient
        
    def _direction_flip_manipulation(self, gradient: torch.Tensor, param_name: str) -> torch.Tensor:
        """Flip gradient direction while maintaining magnitude."""
        # Simple direction flip with some randomness to avoid detection
        flip_mask = torch.rand_like(gradient) < self.lambda_param
        
        manipulated_gradient = gradient.clone()
        manipulated_gradient[flip_mask] = -manipulated_gradient[flip_mask]
        
        return manipulated_gradient
        
    def _orthogonal_projection_manipulation(self, gradient: torch.Tensor, param_name: str) -> torch.Tensor:
        """Project gradient onto orthogonal space of benign gradients."""
        if not self.benign_updates:
            return gradient
            
        benign_grads = []
        for client_updates in self.benign_updates.values():
            if param_name in client_updates:
                benign_grads.append(client_updates[param_name].flatten())
                
        if not benign_grads:
            return gradient
            
        # Perform Gram-Schmidt orthogonalization
        gradient_flat = gradient.flatten()
        
        # Orthogonalize against benign gradients
        orthogonal_component = gradient_flat.clone()
        
        for benign_grad in benign_grads:
            # Subtract projection onto benign gradient
            dot_product = torch.dot(orthogonal_component, benign_grad)
            benign_norm_sq = torch.dot(benign_grad, benign_grad)
            
            if benign_norm_sq > 1e-8:
                projection = (dot_product / benign_norm_sq) * benign_grad
                orthogonal_component = orthogonal_component - projection * self.lambda_param
                
        return orthogonal_component.reshape(gradient.shape)
        
    def update_benign_references(self, all_client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """Update reference to benign client updates."""
        # This method can be called by the FL system to provide benign updates
        # for more sophisticated manipulation
        self.benign_updates = all_client_updates
        
    def update_cluster_context(self, cluster_info: Dict[str, Any]):
        """
        Update cluster context information for cluster-aware attacks.
        
        Args:
            cluster_info: Dictionary containing:
                - 'cluster_id': ID of the cluster this client belongs to
                - 'cluster_members': List of other clients in the same cluster
                - 'cluster_aggregate': Aggregated gradient from cluster (if available)
                - 'expected_cluster_direction': Expected learning direction for cluster
        """
        self.cluster_info = cluster_info
        
    def update_global_context(self, global_info: Dict[str, Any]):
        """
        Update global context information for global-level attacks.
        
        Args:
            global_info: Dictionary containing:
                - 'global_learning_direction': Expected global model learning direction
                - 'round_number': Current federated learning round
                - 'global_aggregate': Global aggregated gradient (if available)
        """
        self.global_context = global_info
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information."""
        info = super().get_attack_info()
        info.update({
            'manipulation_strategy': self.manipulation_strategy,
            'target_level': self.target_level,
            'target_client': self.target_client,
            'aggregation_method': self.aggregation_method,
            'lambda_param': self.lambda_param,
            'cluster_awareness': self.cluster_awareness,
            'adaptive_scaling': self.adaptive_scaling,
            'benign_clients_tracked': len(self.benign_updates),
            'gradient_history_length': len(self.gradient_history),
            'cluster_context_available': bool(self.cluster_info),
            'global_context_available': bool(self.global_context)
        })
        
        # Add cross-cluster attack information if available
        if hasattr(self, 'cross_cluster_attack_info') and self.cross_cluster_attack_info:
            info.update({
                'cross_cluster_attacks': self.cross_cluster_attack_info,
                'total_cross_cluster_attacks': len(self.cross_cluster_attack_info),
                'attack_variant': 'cross_cluster_ipm',
                'cross_cluster_summary': {
                    'average_target_norm': sum(attack['target_norm'] for attack in self.cross_cluster_attack_info) / len(self.cross_cluster_attack_info),
                    'average_cancellation_ratio': sum(attack['expected_cancellation_ratio'] for attack in self.cross_cluster_attack_info) / len(self.cross_cluster_attack_info),
                    'scale_factor_used': self.cross_cluster_attack_info[0]['scale_factor'] if self.cross_cluster_attack_info else None
                }
            })
            
        return info


# Register the attack with the factory
AttackFactory.register_attack('ipm', IPMAttack)
AttackFactory.register_attack('inner_product_manipulation', IPMAttack)