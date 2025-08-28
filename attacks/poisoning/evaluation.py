import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os


class PoisoningEvaluator:
    """
    Evaluation framework for poisoning attacks in federated learning.
    
    Provides metrics and analysis tools to assess attack effectiveness
    and defense robustness.
    """
    
    def __init__(self, num_classes: int = 10, save_path: str = "results/poisoning_evaluation"):
        self.num_classes = num_classes
        self.save_path = save_path
        self.results = {}
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def evaluate_model_performance(self, model, clean_test_data: torch.Tensor, 
                                 clean_test_labels: torch.Tensor,
                                 poisoned_test_data: Optional[torch.Tensor] = None,
                                 poisoned_test_labels: Optional[torch.Tensor] = None,
                                 experiment_name: str = "default") -> Dict[str, float]:
        """
        Evaluate model performance on clean and poisoned test data.
        
        Args:
            model: Trained model to evaluate
            clean_test_data: Clean test dataset
            clean_test_labels: Clean test labels
            poisoned_test_data: Test data with poisoning triggers
            poisoned_test_labels: Expected labels for poisoned data
            experiment_name: Name for this evaluation experiment
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = model(clean_test_data)
            clean_predictions = torch.argmax(clean_outputs, dim=1)
            clean_accuracy = accuracy_score(clean_test_labels.cpu(), clean_predictions.cpu())
            
            metrics['clean_accuracy'] = clean_accuracy
            
            # Poisoned accuracy (if available)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                poisoned_outputs = model(poisoned_test_data)
                poisoned_predictions = torch.argmax(poisoned_outputs, dim=1)
                
                # Attack Success Rate (ASR) - how often poisoned samples are classified as target
                asr = accuracy_score(poisoned_test_labels.cpu(), poisoned_predictions.cpu())
                metrics['attack_success_rate'] = asr
                
                # Backdoor effectiveness
                if len(torch.unique(poisoned_test_labels)) == 1:
                    # All poisoned samples should have same target label
                    target_class = poisoned_test_labels[0].item()
                    target_predictions = (poisoned_predictions == target_class).float().mean().item()
                    metrics['backdoor_success_rate'] = target_predictions
        
        # Store results
        self.results[experiment_name] = metrics
        
        return metrics
        
    def evaluate_defense_effectiveness(self, baseline_metrics: Dict[str, float],
                                     defended_metrics: Dict[str, float],
                                     defense_name: str = "unknown") -> Dict[str, float]:
        """
        Compare attack effectiveness with and without defenses.
        
        Args:
            baseline_metrics: Metrics without any defenses
            defended_metrics: Metrics with defenses applied
            defense_name: Name of the defense mechanism
            
        Returns:
            Defense effectiveness metrics
        """
        effectiveness = {}
        
        # Clean accuracy preservation
        if 'clean_accuracy' in baseline_metrics and 'clean_accuracy' in defended_metrics:
            clean_preservation = defended_metrics['clean_accuracy'] / baseline_metrics['clean_accuracy']
            effectiveness['clean_accuracy_preservation'] = clean_preservation
            
        # Attack success rate reduction
        if 'attack_success_rate' in baseline_metrics and 'attack_success_rate' in defended_metrics:
            asr_reduction = 1.0 - (defended_metrics['attack_success_rate'] / baseline_metrics['attack_success_rate'])
            effectiveness['attack_success_rate_reduction'] = asr_reduction
            
        # Backdoor success rate reduction
        if 'backdoor_success_rate' in baseline_metrics and 'backdoor_success_rate' in defended_metrics:
            bsr_reduction = 1.0 - (defended_metrics['backdoor_success_rate'] / baseline_metrics['backdoor_success_rate'])
            effectiveness['backdoor_success_rate_reduction'] = bsr_reduction
            
        # Overall defense score (weighted combination)
        weights = {'clean_preservation': 0.3, 'attack_reduction': 0.7}
        defense_score = 0
        
        if 'clean_accuracy_preservation' in effectiveness:
            defense_score += weights['clean_preservation'] * effectiveness['clean_accuracy_preservation']
        if 'attack_success_rate_reduction' in effectiveness:
            defense_score += weights['attack_reduction'] * effectiveness['attack_success_rate_reduction']
            
        effectiveness['overall_defense_score'] = defense_score
        
        self.results[f"{defense_name}_effectiveness"] = effectiveness
        
        return effectiveness
        
    def analyze_gradient_changes(self, clean_gradients: Dict[str, torch.Tensor],
                               poisoned_gradients: Dict[str, torch.Tensor],
                               attack_name: str = "unknown") -> Dict[str, float]:
        """
        Analyze the impact of poisoning on gradient updates.
        
        Args:
            clean_gradients: Gradients from clean training
            poisoned_gradients: Gradients from poisoned training
            attack_name: Name of the attack
            
        Returns:
            Gradient analysis metrics
        """
        analysis = {}
        
        # Calculate gradient norms
        clean_norm = 0
        poisoned_norm = 0
        cosine_similarity = 0
        l2_distance = 0
        
        param_count = 0
        
        for param_name in clean_gradients.keys():
            if param_name not in poisoned_gradients:
                continue
                
            clean_grad = clean_gradients[param_name].flatten()
            poisoned_grad = poisoned_gradients[param_name].flatten()
            
            # Gradient norms
            clean_norm += torch.norm(clean_grad).item() ** 2
            poisoned_norm += torch.norm(poisoned_grad).item() ** 2
            
            # Cosine similarity
            dot_product = torch.dot(clean_grad, poisoned_grad).item()
            norm_product = torch.norm(clean_grad).item() * torch.norm(poisoned_grad).item()
            if norm_product > 1e-8:
                cosine_similarity += dot_product / norm_product
                
            # L2 distance
            l2_distance += torch.norm(clean_grad - poisoned_grad).item() ** 2
            
            param_count += 1
            
        if param_count > 0:
            analysis['clean_gradient_norm'] = np.sqrt(clean_norm)
            analysis['poisoned_gradient_norm'] = np.sqrt(poisoned_norm)
            analysis['average_cosine_similarity'] = cosine_similarity / param_count
            analysis['gradient_l2_distance'] = np.sqrt(l2_distance)
            analysis['gradient_norm_ratio'] = np.sqrt(poisoned_norm) / np.sqrt(clean_norm) if clean_norm > 1e-8 else 0
            
        self.results[f"{attack_name}_gradient_analysis"] = analysis
        
        return analysis
        
    def generate_confusion_matrix(self, true_labels: torch.Tensor, 
                                predicted_labels: torch.Tensor,
                                title: str = "Confusion Matrix",
                                save_name: Optional[str] = None) -> np.ndarray:
        """Generate and save confusion matrix visualization."""
        cm = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
                        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_path, f"{save_name}_confusion_matrix.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
            
        plt.show()
        
        return cm
        
    def plot_attack_comparison(self, experiments: List[str], 
                             metrics: List[str] = None,
                             save_name: str = "attack_comparison"):
        """
        Plot comparison of different attacks and defenses.
        
        Args:
            experiments: List of experiment names to compare
            metrics: List of metrics to plot
            save_name: Name for saved plot
        """
        if metrics is None:
            metrics = ['clean_accuracy', 'attack_success_rate']
            
        # Prepare data
        experiment_data = {}
        for exp in experiments:
            if exp in self.results:
                experiment_data[exp] = self.results[exp]
                
        if not experiment_data:
            print("No experimental data found for plotting")
            return
            
        # Create plot
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            exp_names = []
            values = []
            
            for exp_name, data in experiment_data.items():
                if metric in data:
                    exp_names.append(exp_name)
                    values.append(data[metric])
                    
            if values:
                bars = axes[i].bar(range(len(exp_names)), values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xticks(range(len(exp_names)))
                axes[i].set_xticklabels(exp_names, rotation=45, ha='right')
                axes[i].set_ylim(0, 1.0)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                               
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_path, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack comparison plot saved to {save_path}")
        
        plt.show()
        
    def save_results(self, filename: str = "poisoning_evaluation_results.json"):
        """Save all evaluation results to JSON file."""
        save_path = os.path.join(self.save_path, filename)
        
        # Convert any tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
                
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Evaluation results saved to {save_path}")
        
    def load_results(self, filename: str = "poisoning_evaluation_results.json"):
        """Load evaluation results from JSON file."""
        load_path = os.path.join(self.save_path, filename)
        
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                self.results = json.load(f)
            print(f"Evaluation results loaded from {load_path}")
        else:
            print(f"No results file found at {load_path}")
            
    def print_summary(self):
        """Print a summary of all evaluation results."""
        print("\n=== Poisoning Attack Evaluation Summary ===")
        
        for experiment_name, metrics in self.results.items():
            print(f"\n{experiment_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
                    
        print("\n=== End Summary ===\n")