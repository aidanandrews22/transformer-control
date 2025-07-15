#!/usr/bin/env python3
"""Test script to verify CartPole data generation works correctly."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_cartpole_data import CartPoleDataGenerator


def test_single_trajectory():
    """Test generating a single trajectory."""
    print("Testing single trajectory generation...")
    
    generator = CartPoleDataGenerator(trajectory_length=200)
    states, controls = generator.generate_single_trajectory()
    
    print(f"States shape: {states.shape}")  # Should be (200, 4)
    print(f"Controls shape: {controls.shape}")  # Should be (200, 2)
    
    # Check data ranges
    print(f"State ranges:")
    print(f"  x: [{states[:, 0].min():.3f}, {states[:, 0].max():.3f}]")
    print(f"  x_dot: [{states[:, 1].min():.3f}, {states[:, 1].max():.3f}]")
    print(f"  theta: [{states[:, 2].min():.3f}, {states[:, 2].max():.3f}]")
    print(f"  theta_dot: [{states[:, 3].min():.3f}, {states[:, 3].max():.3f}]")
    
    print(f"Control ranges:")
    print(f"  force: [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}]")
    print(f"  mode: [{controls[:, 1].min():.3f}, {controls[:, 1].max():.3f}]")
    
    return states, controls


def test_batch_generation():
    """Test generating a batch of trajectories."""
    print("\nTesting batch generation...")
    
    generator = CartPoleDataGenerator(trajectory_length=100)
    xs, ys = generator.generate_batch(batch_size=4)
    
    print(f"Batch xs shape: {xs.shape}")  # Should be (4, 100, 4)
    print(f"Batch ys shape: {ys.shape}")  # Should be (4, 100, 2)
    print(f"Data types: xs={xs.dtype}, ys={ys.dtype}")
    
    return xs, ys


def plot_trajectory(states, controls, title="CartPole Trajectory"):
    """Plot a single trajectory."""
    time = np.arange(len(states))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    # States
    axes[0, 0].plot(time, states[:, 0])
    axes[0, 0].set_title('Cart Position (x)')
    axes[0, 0].set_ylabel('Position [m]')
    
    axes[0, 1].plot(time, states[:, 1])
    axes[0, 1].set_title('Cart Velocity (x_dot)')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    
    axes[0, 2].plot(time, states[:, 2])
    axes[0, 2].set_title('Pole Angle (theta)')
    axes[0, 2].set_ylabel('Angle [rad]')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Upright')
    axes[0, 2].axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='Hanging')
    axes[0, 2].legend()
    
    axes[1, 0].plot(time, states[:, 3])
    axes[1, 0].set_title('Pole Angular Velocity (theta_dot)')
    axes[1, 0].set_ylabel('Angular Velocity [rad/s]')
    axes[1, 0].set_xlabel('Time Step')
    
    # Controls
    axes[1, 1].plot(time, controls[:, 0])
    axes[1, 1].set_title('Control Force')
    axes[1, 1].set_ylabel('Force [N]')
    axes[1, 1].set_xlabel('Time Step')
    
    axes[1, 2].plot(time, controls[:, 1])
    axes[1, 2].set_title('Control Mode')
    axes[1, 2].set_ylabel('Mode (0=Swing-up, 1=LQR)')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing CartPole Data Generation")
    print("=" * 60)
    
    # Test single trajectory
    states, controls = test_single_trajectory()
    
    # Test batch generation
    xs, ys = test_batch_generation()
    
    # Plot first trajectory from batch
    print("\nPlotting first trajectory...")
    plot_trajectory(
        xs[0].numpy(), 
        ys[0].numpy(), 
        "First Trajectory from Batch"
    )
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    
    # Show some statistics
    print(f"\nDataset Statistics:")
    print(f"Batch size: {xs.shape[0]}")
    print(f"Trajectory length: {xs.shape[1]}")
    print(f"State dimension: {xs.shape[2]}")
    print(f"Control dimension: {ys.shape[2]}")
    
    # Check for successful control (upright pole)
    upright_count = 0
    for i in range(xs.shape[0]):
        final_angle = xs[i, -1, 2].item()  # Final theta
        if abs(((final_angle + np.pi) % (2 * np.pi)) - np.pi) < 0.5:  # Within ~28 degrees of upright
            upright_count += 1
    
    print(f"Trajectories ending near upright: {upright_count}/{xs.shape[0]}")


if __name__ == "__main__":
    main() 