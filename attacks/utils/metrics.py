"""
Attack evaluation metrics
"""

import torch
import numpy as np
import inversefed


def calculate_attack_metrics(reconstructed_images, ground_truth_images, normalization_stats):
    """
    Calculate comprehensive attack success metrics
    
    Args:
        reconstructed_images: Reconstructed images tensor
        ground_truth_images: Original images tensor  
        normalization_stats: (mean, std) normalization constants
        
    Returns:
        dict: Dictionary containing various metrics
    """
    dm, ds = normalization_stats
    
    # Basic metrics
    mse = (reconstructed_images.detach() - ground_truth_images).pow(2).mean().item()
    psnr = inversefed.metrics.psnr(reconstructed_images, ground_truth_images, factor=1/ds)
    
    # Per-image PSNR
    num_images = reconstructed_images.shape[0]
    per_image_psnr = []
    per_image_mse = []
    
    for i in range(num_images):
        img_psnr = inversefed.metrics.psnr(
            reconstructed_images[i:i+1], 
            ground_truth_images[i:i+1], 
            factor=1/ds
        )
        img_mse = (reconstructed_images[i] - ground_truth_images[i]).pow(2).mean().item()
        
        per_image_psnr.append(img_psnr)
        per_image_mse.append(img_mse)
    
    # Statistical measures
    metrics = {
        'mse': mse,
        'psnr_avg': psnr,
        'psnr_best': max(per_image_psnr),
        'psnr_worst': min(per_image_psnr),
        'psnr_std': np.std(per_image_psnr),
        'per_image_psnr': per_image_psnr,
        'per_image_mse': per_image_mse,
        'success_rate_20db': len([p for p in per_image_psnr if p > 20]) / len(per_image_psnr) * 100,
        'success_rate_25db': len([p for p in per_image_psnr if p > 25]) / len(per_image_psnr) * 100,
    }
    
    return metrics


def evaluate_privacy_protection(attack_results):
    """
    Evaluate the effectiveness of privacy protection based on attack results
    
    Args:
        attack_results: List of attack result dictionaries
        
    Returns:
        dict: Privacy protection evaluation
    """
    if not attack_results:
        return {'protection_level': 'unknown', 'avg_psnr': 0}
    
    avg_psnr = np.mean([r['psnr'] for r in attack_results])
    success_rate = len([r for r in attack_results if r['psnr'] > 20]) / len(attack_results) * 100
    
    # Classify protection level
    if avg_psnr < 15:
        protection_level = 'strong'
    elif avg_psnr < 20:
        protection_level = 'moderate'
    elif avg_psnr < 25:
        protection_level = 'weak'
    else:
        protection_level = 'vulnerable'
    
    return {
        'protection_level': protection_level,
        'avg_psnr': avg_psnr,
        'success_rate': success_rate,
        'total_attacks': len(attack_results),
        'recommendation': get_protection_recommendation(protection_level, avg_psnr)
    }


def get_protection_recommendation(protection_level, avg_psnr):
    """Get privacy protection recommendations based on attack success"""
    recommendations = {
        'strong': "✅ Excellent privacy protection! Current mechanisms are effective.",
        'moderate': "⚠️ Moderate protection. Consider enabling differential privacy or increasing SMPC parameters.",
        'weak': "🔸 Weak protection. Enable both SMPC and differential privacy with stronger parameters.",
        'vulnerable': "🚨 VULNERABLE! Training data easily reconstructed. Immediately enable strong privacy mechanisms!"
    }
    
    return recommendations.get(protection_level, "Unknown protection level")