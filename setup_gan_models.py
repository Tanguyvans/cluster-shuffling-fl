#!/usr/bin/env python3
"""
Setup script for downloading/training GAN models for GIFD attack

This script helps you obtain the necessary GAN checkpoints for running GIFD attacks.
"""

import os
import sys
import torch
import argparse
from pathlib import Path


def setup_dcgan_cifar10():
    """Setup DCGAN for CIFAR-10"""
    print("📦 Setting up DCGAN for CIFAR-10...")
    
    # Check if checkpoint exists
    dcgan_path = Path("GIFD_Gradient_Inversion_Attack/inversefed/genmodels/cifar10_dcgan")
    checkpoint_path = dcgan_path / "netG_cifar10.pth"
    
    if checkpoint_path.exists():
        print("✅ DCGAN checkpoint already exists!")
        return
    
    print("\n⚠️  DCGAN checkpoint not found!")
    print("\nYou have two options:")
    print("\n1. Train DCGAN on CIFAR-10 (recommended):")
    print("   cd GIFD_Gradient_Inversion_Attack/inversefed/genmodels/cifar10_dcgan/")
    print("   python dcgan.py --dataset cifar10 --dataroot ./data --imageSize 32 --cuda --niter 100")
    print("\n2. Download pre-trained checkpoint:")
    print("   Unfortunately, the GIFD paper doesn't provide pre-trained DCGAN weights.")
    print("   You'll need to train it yourself using the command above.")
    print("\n   Training takes ~1-2 hours on GPU and produces good results after 50-100 epochs.")
    

def setup_stylegan2():
    """Setup StyleGAN2 (for FFHQ, not ideal for CIFAR-10)"""
    print("📦 Setting up StyleGAN2...")
    
    stylegan_path = Path("GIFD_Gradient_Inversion_Attack/inversefed/genmodels/stylegan2_io")
    checkpoint_path = stylegan_path / "stylegan2-ffhq-config-f.pt"
    
    if checkpoint_path.exists():
        print("✅ StyleGAN2 checkpoint already exists!")
        return
    
    print("\n⚠️  StyleGAN2 checkpoint not found!")
    print("\nTo download StyleGAN2-FFHQ checkpoint (1.1GB):")
    print("   cd GIFD_Gradient_Inversion_Attack/inversefed/genmodels/stylegan2_io/")
    print("   gdown --id 1JCBiKY_yUixTa6F1eflABL88T4cii2GR")
    print("\n⚠️  Note: This model is trained on faces (FFHQ), not CIFAR-10!")
    print("   For CIFAR-10, use DCGAN instead.")


def setup_biggan():
    """Setup BigGAN (downloads automatically)"""
    print("📦 Setting up BigGAN...")
    print("✅ BigGAN weights are downloaded automatically when first used.")
    print("   The model will be cached in ~/.cache/torch/hub/")
    

def check_shape_predictor():
    """Check for shape predictor needed by StyleGAN2"""
    shape_predictor_path = Path("GIFD_Gradient_Inversion_Attack/shape_predictor_68_face_landmarks.dat")
    
    if not shape_predictor_path.exists():
        print("\n⚠️  Shape predictor not found (needed for StyleGAN2)!")
        print("\nTo download:")
        print("   cd GIFD_Gradient_Inversion_Attack/")
        print("   gdown --id 1c1qtz3MVTAvJpYvsMIR5MoSvdiwN2DGb")


def main():
    parser = argparse.ArgumentParser(
        description='Setup GAN models for GIFD attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all           # Check all GAN models
  %(prog)s --dcgan         # Setup DCGAN for CIFAR-10 (recommended)
  %(prog)s --stylegan2     # Setup StyleGAN2 (for faces, not CIFAR-10)
  %(prog)s --biggan        # Setup BigGAN
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Setup all GAN models')
    parser.add_argument('--dcgan', action='store_true', help='Setup DCGAN for CIFAR-10')
    parser.add_argument('--stylegan2', action='store_true', help='Setup StyleGAN2')
    parser.add_argument('--biggan', action='store_true', help='Setup BigGAN')
    
    args = parser.parse_args()
    
    # Default to checking all if no specific option
    if not any([args.all, args.dcgan, args.stylegan2, args.biggan]):
        args.all = True
    
    print("🎯 GAN Model Setup for GIFD Attack")
    print("="*50)
    
    if args.all or args.dcgan:
        setup_dcgan_cifar10()
        print()
    
    if args.all or args.stylegan2:
        setup_stylegan2()
        check_shape_predictor()
        print()
    
    if args.all or args.biggan:
        setup_biggan()
        print()
    
    print("\n" + "="*50)
    print("📋 Recommendations for CIFAR-10:")
    print("   1. Use DCGAN (--gan dcgan) - specifically trained on CIFAR-10")
    print("   2. BigGAN (--gan biggan) - general purpose, downloads automatically")
    print("   3. Avoid StyleGAN2 for CIFAR-10 - it's trained on faces (FFHQ)")
    print("\n💡 For best results with CIFAR-10, train DCGAN using the command above.")


if __name__ == "__main__":
    main()