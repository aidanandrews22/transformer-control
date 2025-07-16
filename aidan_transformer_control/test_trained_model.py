#!/usr/bin/env python3
"""
Test script for trained CartPole control transformer model.
This script loads a trained model and tests its inference capabilities.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from quinine import QuinineArgumentParser
from schema import schema
from models import build_model
import pickle
import glob

def load_latest_checkpoint(output_dir):
    """Load the latest checkpoint from the output directory."""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoints[-1]
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    return torch.load(latest_checkpoint, map_location='cpu')

def test_model_inference(model, test_data, device='cpu'):
    """Test model inference on sample data."""
    model.eval()
    
    with torch.no_grad():
        # Get a sample trajectory
        xs, ys = test_data[0]  # First trajectory
        
        # Take first part of trajectory as input
        seq_len = min(100, xs.shape[0] - 1)  # Use first 100 steps
        xs_input = xs[:seq_len].unsqueeze(0).to(device)  # Add batch dimension
        ys_input = ys[:seq_len].unsqueeze(0).to(device)
        
        # Run inference
        control_pred, state_pred = model(xs_input, ys_input, inf="yes")
        
        # Get predictions
        control_pred = control_pred.cpu().numpy()
        state_pred = state_pred.cpu().numpy()
        
        return {
            'input_states': xs_input.cpu().numpy(),
            'input_controls': ys_input.cpu().numpy(),
            'predicted_controls': control_pred,
            'predicted_states': state_pred,
            'actual_states': xs[1:seq_len+1].numpy(),
            'actual_controls': ys[1:seq_len+1].numpy()
        }

def plot_results(results):
    """Plot the inference results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot states
    state_names = ['x', 'x_dot', 'theta', 'theta_dot']
    for i, name in enumerate(state_names):
        if i < 2:
            ax = axes[0, i]
        else:
            ax = axes[1, i-2]
            
        ax.plot(results['actual_states'][0, :, i], label=f'Actual {name}', alpha=0.7)
        ax.plot(results['predicted_states'][0, :, i], label=f'Predicted {name}', alpha=0.7)
        ax.set_title(f'State: {name}')
        ax.legend()
        ax.grid(True)
    
    # Plot controls
    ax = axes[0, 2]
    ax.plot(results['actual_controls'][0, :, 0], label='Actual Force', alpha=0.7)
    ax.plot(results['predicted_controls'][0, :, 0], label='Predicted Force', alpha=0.7)
    ax.set_title('Control Force')
    ax.legend()
    ax.grid(True)
    
    # Plot controller mode
    ax = axes[1, 2]
    ax.plot(results['actual_controls'][0, :, 1], label='Actual Mode', alpha=0.7)
    ax.plot(results['predicted_controls'][0, :, 1], label='Predicted Mode', alpha=0.7)
    ax.set_title('Controller Mode')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_inference_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main testing function."""
    
    # Parse configuration
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = build_model(args.model)
    
    # Load checkpoint
    try:
        checkpoint = load_latest_checkpoint(args.out_dir)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading checkpoint: {e}")
        print("Please train the model first using run_full_training.py")
        return
    
    model.to(device)
    
    # Load test data
    print("Loading test data...")
    dataset_path = args.dataset_filesfolder
    pickle_files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    
    if not pickle_files:
        print(f"No test data found in {dataset_path}")
        return
    
    # Load first pickle file for testing
    test_file = os.path.join(dataset_path, pickle_files[0])
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Loaded test data with {len(test_data)} trajectories")
    
    # Test model inference
    print("Testing model inference...")
    results = test_model_inference(model, test_data, device)
    
    # Calculate some basic metrics
    state_mse = np.mean((results['predicted_states'] - results['actual_states'][:1]) ** 2)
    control_mse = np.mean((results['predicted_controls'] - results['actual_controls'][:1]) ** 2)
    
    print(f"\nInference Results:")
    print(f"  State MSE: {state_mse:.6f}")
    print(f"  Control MSE: {control_mse:.6f}")
    
    # Plot results
    print("Plotting results...")
    plot_results(results)
    print("Results saved as 'model_inference_test.png'")

if __name__ == "__main__":
    main() 