"""
Attack result visualization utilities
"""

import torch
import torchvision
import os
import numpy as np


def save_attack_results(output, ground_truth, normalization_stats, round_num, client_id, output_dir, save_individual=True):
    """
    Save attack reconstruction results with proper organization
    
    Args:
        output: Reconstructed images
        ground_truth: Original images
        normalization_stats: (mean, std) for denormalization
        round_num: Training round number
        client_id: Client identifier
        output_dir: Output directory path
        save_individual: Whether to save individual images
    """
    os.makedirs(output_dir, exist_ok=True)
    dm, ds = normalization_stats
    
    # Denormalize images
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    ground_truth_denorm = torch.clamp(ground_truth * ds + dm, 0, 1)
    
    num_images = output.shape[0]
    
    # Save reconstructed images grid
    torchvision.utils.save_image(
        output_denorm,
        f'{output_dir}/round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save original images grid
    torchvision.utils.save_image(
        ground_truth_denorm,
        f'{output_dir}/round_{round_num}_client_{client_id}_original.png',
        nrow=5
    )
    
    # Save side-by-side comparison
    comparison = torch.cat([ground_truth_denorm, output_denorm], dim=0)
    torchvision.utils.save_image(
        comparison,
        f'{output_dir}/round_{round_num}_client_{client_id}_comparison.png',
        nrow=num_images
    )
    
    # Save individual images if requested
    if save_individual:
        individual_dir = os.path.join(output_dir, 'individual')
        os.makedirs(individual_dir, exist_ok=True)
        
        for i in range(num_images):
            # Original
            torchvision.utils.save_image(
                ground_truth_denorm[i],
                f'{individual_dir}/round_{round_num}_client_{client_id}_img_{i}_original.png'
            )
            # Reconstructed
            torchvision.utils.save_image(
                output_denorm[i],
                f'{individual_dir}/round_{round_num}_client_{client_id}_img_{i}_reconstructed.png'
            )
    
    print(f"Saved attack results to {output_dir}/round_{round_num}_client_{client_id}_*.png")
    if save_individual:
        print(f"Saved individual images to {individual_dir}/")


def generate_attack_summary_table(results):
    """Generate a formatted summary table of attack results"""
    if not results:
        return "No attack results to display"
    
    print(f"\n{'='*80}")
    print("=== GRADIENT INVERSION ATTACK SUMMARY ===")
    print(f"{'='*80}")
    print(f"{'Round':<6} {'Client':<10} {'PSNR (dB)':<12} {'Time (min)':<12} {'Grad Norm':<12} {'Loss':<8}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['round']:<6} {r['client_id']:<10} {r['psnr']:<12.2f} {r['attack_time']/60:<12.1f} "
              f"{r['gradient_norm']:<12.4f} {r['loss']:<8.4f}")


def generate_privacy_evaluation_report(results, privacy_mechanism="unknown"):
    """Generate a comprehensive privacy evaluation report"""
    if not results:
        print("No results available for privacy evaluation")
        return
    
    from .metrics import evaluate_privacy_protection
    
    evaluation = evaluate_privacy_protection(results)
    all_psnr = [r['psnr'] for r in results]
    
    print(f"\n{'='*80}")
    print(f"=== PRIVACY PROTECTION EVALUATION ===")
    print(f"Privacy Mechanism: {privacy_mechanism}")
    print(f"{'='*80}")
    
    # Overall statistics
    print(f"\n=== Overall Attack Statistics ===")
    print(f"Total attacks performed: {evaluation['total_attacks']}")
    print(f"Average PSNR: {evaluation['avg_psnr']:.2f} dB")
    print(f"Best reconstruction PSNR: {np.max(all_psnr):.2f} dB")
    print(f"Worst reconstruction PSNR: {np.min(all_psnr):.2f} dB")
    print(f"PSNR standard deviation: {np.std(all_psnr):.2f} dB")
    
    # Success rates
    good_reconstructions = [r for r in results if r['psnr'] > 20]
    excellent_reconstructions = [r for r in results if r['psnr'] > 25]
    
    print(f"\n=== Attack Success Rates ===")
    print(f"Good reconstructions (PSNR > 20 dB): {len(good_reconstructions)}/{len(results)} ({len(good_reconstructions)/len(results)*100:.1f}%)")
    print(f"Excellent reconstructions (PSNR > 25 dB): {len(excellent_reconstructions)}/{len(results)} ({len(excellent_reconstructions)/len(results)*100:.1f}%)")
    
    # Privacy protection assessment
    print(f"\n=== Privacy Protection Assessment ===")
    print(f"Protection Level: {evaluation['protection_level'].upper()}")
    print(f"Recommendation: {evaluation['recommendation']}")
    
    # Analysis by round if multiple rounds
    rounds = sorted(list(set([r['round'] for r in results])))
    if len(rounds) > 1:
        print(f"\n=== Analysis by Training Round ===")
        for round_num in rounds:
            round_results = [r for r in results if r['round'] == round_num]
            if round_results:
                avg_psnr = np.mean([r['psnr'] for r in round_results])
                print(f"Round {round_num}: {len(round_results)} clients, Avg PSNR = {avg_psnr:.2f} dB")
    
    return evaluation