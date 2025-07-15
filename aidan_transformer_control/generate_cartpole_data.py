import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import argparse
from typing import Tuple, List
import gymnasium as gym

from aidan_gym import ContinuousCartPoleWrapper, CartPoleSwingUpController


class CartPoleDataGenerator:
    """Generate training data from CartPole swing-up controller trajectories."""
    
    def __init__(self, max_force: float = 500.0, trajectory_length: int = 400):
        """
        Initialize the data generator.
        
        Args:
            max_force: Maximum force for the controller
            trajectory_length: Number of timesteps per trajectory
        """
        self.max_force = max_force
        self.trajectory_length = trajectory_length
        
    def generate_single_trajectory(self, 
                                 mass_pole_modifier: float = 1.0,
                                 mass_cart_modifier: float = 1.0, 
                                 length_modifier: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single trajectory using the hybrid controller.
        
        Args:
            mass_pole_modifier: Pole mass multiplier
            mass_cart_modifier: Cart mass multiplier  
            length_modifier: Pole length multiplier
            
        Returns:
            Tuple of (states, controls) where:
            - states: (trajectory_length, 4) array of [x, x_dot, theta, theta_dot]
            - controls: (trajectory_length, 2) array of [force, mode_flag]
        """
        # Create environment with modified parameters
        base_env = gym.make('CartPole-v1', render_mode=None)
        env = ContinuousCartPoleWrapper(
            base_env, 
            max_force=self.max_force,
            mass_pole_modifier=mass_pole_modifier,
            mass_cart_modifier=mass_cart_modifier,
            length_modifier=length_modifier
        )
        
        # Create controller
        controller = CartPoleSwingUpController(env)
        
        # Storage for trajectory data
        states = []
        controls = []
        
        # Reset environment
        state, _ = env.reset()
        
        # Generate trajectory
        for step in range(self.trajectory_length):
            # Store current state
            states.append(state.copy())
            
            # Get control action
            action = controller.get_action(state)
            force = action[0]
            
            # Create control vector: [force, mode_flag]
            # mode_flag: 0 = swing-up, 1 = LQR
            mode_flag = 1.0 if controller.current_mode == "LQR" else 0.0
            control_vector = [force, mode_flag]
            controls.append(control_vector)
            
            # Step environment
            state, _, terminated, truncated, _ = env.step(action)
            
            # Handle termination
            if terminated or truncated:
                # Pad with final state if trajectory ends early
                final_state = state.copy()
                final_control = [0.0, mode_flag]  # Zero force when terminated
                
                while len(states) < self.trajectory_length:
                    states.append(final_state)
                    controls.append(final_control)
                break
        
        env.close()
        
        return np.array(states), np.array(controls)
    
    def generate_batch(self, 
                      batch_size: int,
                      mass_pole_range: Tuple[float, float] = (0.3, 1.0),
                      mass_cart_range: Tuple[float, float] = (1.8, 2.2),
                      length_range: Tuple[float, float] = (1.0, 1.5)) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of trajectories with randomized parameters.
        
        Args:
            batch_size: Number of trajectories to generate
            mass_pole_range: (min, max) range for pole mass modifier
            mass_cart_range: (min, max) range for cart mass modifier
            length_range: (min, max) range for length modifier
            
        Returns:
            Tuple of (xs, ys, cartmass, polemass, polelength):
            - xs: (batch_size, trajectory_length, 4) state sequences
            - ys: (batch_size, trajectory_length, 2) control sequences
            - cartmass: (batch_size,) array of cart masses
            - polemass: (batch_size,) array of pole masses
            - polelength: (batch_size,) array of pole lengths
        """
        batch_states = []
        batch_controls = []
        cartmasses = []
        polemasses = []
        polelengths = []
        
        for _ in range(batch_size):
            # Sample random parameters - these are the actual values, not modifiers
            cartmass = np.random.uniform(*mass_cart_range)
            polemass = np.random.uniform(*mass_pole_range)
            polelength = np.random.uniform(*length_range)
            
            # Calculate modifiers for the environment (relative to default values)
            # Default CartPole-v1: cart_mass=1.0, pole_mass=0.1, pole_length=0.5
            mass_cart_mod = cartmass / 1.0
            mass_pole_mod = polemass / 0.1
            length_mod = polelength / 0.5
            
            # Generate single trajectory
            states, controls = self.generate_single_trajectory(
                mass_pole_modifier=mass_pole_mod,
                mass_cart_modifier=mass_cart_mod,
                length_modifier=length_mod
            )
            
            batch_states.append(states)
            batch_controls.append(controls)
            cartmasses.append(cartmass)
            polemasses.append(polemass)
            polelengths.append(polelength)
        
        # Convert to tensors
        xs = torch.tensor(np.array(batch_states), dtype=torch.float32)
        ys = torch.tensor(np.array(batch_controls), dtype=torch.float32)
        cartmass_array = np.array(cartmasses)
        polemass_array = np.array(polemasses)
        polelength_array = np.array(polelengths)
        
        return xs, ys, cartmass_array, polemass_array, polelength_array
    
    def save_dataset(self, 
                    output_dir: str,
                    num_batches: int,
                    batch_size: int,
                    prefix: str = "batch"):
        """
        Generate and save dataset as pickle files.
        
        Args:
            output_dir: Directory to save pickle files
            num_batches: Number of batches to generate
            batch_size: Size of each batch
            prefix: Prefix for pickle filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_batches} batches of {batch_size} trajectories each...")
        print(f"Total trajectories: {num_batches * batch_size}")
        print(f"Trajectory length: {self.trajectory_length} timesteps")
        
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Generate batch
            xs, ys, cartmass, polemass, polelength = self.generate_batch(batch_size)
            
            # Save as pickle
            pickle_file = f"{prefix}_{batch_idx}.pkl"
            pickle_path = os.path.join(output_dir, pickle_file)
            
            with open(pickle_path, 'wb') as f:
                pickle.dump((xs, ys, cartmass, polemass, polelength), f)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Data format: xs.shape = {xs.shape}, ys.shape = {ys.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate CartPole training data')
    parser.add_argument('--output_dir', type=str, default='dataset_cartpole_aidan',
                       help='Output directory for dataset')
    parser.add_argument('--num_batches', type=int, default=100,
                       help='Number of batches to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Trajectories per batch')
    parser.add_argument('--trajectory_length', type=int, default=400,
                       help='Timesteps per trajectory')
    parser.add_argument('--max_force', type=float, default=500.0,
                       help='Maximum control force')
    
    args = parser.parse_args()
    
    # Create data generator
    generator = CartPoleDataGenerator(
        max_force=args.max_force,
        trajectory_length=args.trajectory_length
    )
    
    # Generate and save dataset
    generator.save_dataset(
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 