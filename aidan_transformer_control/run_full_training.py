#!/usr/bin/env python3
"""
Full Training Script for CartPole Control Transformer
This script runs a complete GPT-2 transformer training for CartPole control inference.
"""

import os
import sys
import torch
import warnings
from quinine import QuinineArgumentParser
from schema import schema
from models import build_model
from trainSequential_ebonye_cartpole import train
import wandb

def main():
    """Main training function with proper setup and error handling."""
    
    # Set up CUDA environment
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        response = input("Continue with CPU training? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Parse configuration
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    
    # Validate dataset exists
    dataset_path = args.dataset_filesfolder
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder '{dataset_path}' not found!")
        print("Please run generate_cartpole_data.py first to create the dataset.")
        sys.exit(1)
    
    # Count dataset files
    pickle_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    print(f"Found {len(pickle_files)} dataset files in {dataset_path}")
    
    if len(pickle_files) == 0:
        print("ERROR: No pickle files found in dataset folder!")
        print("Please run generate_cartpole_data.py first to create the dataset.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    
    # Initialize model
    print("Initializing GPT-2 model...")
    print(f"Model configuration:")
    print(f"  - Embedding dimension: {args.model.n_embd}")
    print(f"  - Number of layers: {args.model.n_layer}")
    print(f"  - Number of heads: {args.model.n_head}")
    print(f"  - Max positions: {args.model.n_positions}")
    print(f"  - State dimensions: {args.model.n_dims}")
    
    model = build_model(args.model)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    # Initialize wandb
    if not args.test_run:
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            name=args.wandb.name,
            notes=args.wandb.notes,
            config=args.__dict__
        )
        print("Weights & Biases initialized")
    
    # Print training configuration
    print(f"\nTraining configuration:")
    print(f"  - Epochs: {args.training.epochs}")
    print(f"  - Batch size: {args.training.batch_size}")
    print(f"  - Learning rate: {args.training.learning_rate}")
    print(f"  - Training steps: {args.training.train_steps}")
    print(f"  - Save every: {args.training.save_every_steps} steps")
    print(f"  - Chunk size: {args.use_chunk}")
    
    # Estimate training time
    estimated_batches = len(pickle_files) * args.training.epochs
    print(f"  - Estimated total batches: {estimated_batches:,}")
    
    # Start training
    print("\nStarting training...")
    print("=" * 50)
    
    try:
        train(model, args)
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if not args.test_run:
            wandb.finish()
        print("Cleanup completed")

if __name__ == "__main__":
    main() 