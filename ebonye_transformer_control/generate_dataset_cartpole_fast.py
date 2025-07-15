# Fast cartpole data generation using direct physics computation
# This replaces the slow gym environment with pure mathematical computation

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from curriculum import Curriculum
from quinine import QuinineArgumentParser
import yaml
from schema import schema
import random
from scipy.linalg import solve_continuous_are

torch.backends.cudnn.benchmark = True


def get_valid_masses_and_lengths_uniform(size=1, cartmass=2, polemasslowerbound=0.3, polemassupperbound=1.0, 
                                       polelengthlowerbound=1.0, polelengthupperbound=1.5):
    """Samples valid masses and lengths for the cartpole system using a uniform distribution."""
    cartmass = np.ones(size) * cartmass
    polemass = np.random.uniform(polemasslowerbound, polemassupperbound, size)
    polelength = np.random.uniform(polelengthlowerbound, polelengthupperbound, size)
    return cartmass, polemass, polelength


def find_nearest_upright(theta):
    """Find the nearest upright position (multiple of 2π)"""
    # Handle NaN values
    if np.isnan(theta) or np.isinf(theta):
        return 0.0
    
    # Clamp theta to reasonable bounds to prevent overflow
    theta = np.clip(theta, -100*np.pi, 100*np.pi)
    
    try:
        return np.pi * 2 * round(theta / (2 * np.pi))
    except (ValueError, OverflowError):
        return 0.0


def swingup_lqr_controller(state, switched, cartmass, polemass, polelength):
    """
    Swingup LQR controller that switches between energy-based swingup and LQR stabilization
    """
    # Check for NaN or inf values in state
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return 0.0, switched
    
    x, theta, x_dot, theta_dot = state
    
    # Clamp state values to prevent numerical issues
    x = np.clip(x, -10, 10)
    theta = np.clip(theta, -100*np.pi, 100*np.pi)
    x_dot = np.clip(x_dot, -50, 50)
    theta_dot = np.clip(theta_dot, -50, 50)
    
    g = 9.81
    
    # Energy-based swingup control
    eq_theta = find_nearest_upright(theta)
    
    # Calculate energy
    kinetic_energy = 0.5 * polemass * polelength**2 * theta_dot**2
    potential_energy = polemass * g * polelength * (1 - np.cos(theta - eq_theta))
    total_energy = kinetic_energy + potential_energy
    
    # Desired energy at upright position
    desired_energy = polemass * g * polelength * 2
    
    # Switch to LQR if close to upright and low energy
    angle_from_upright = abs(theta - eq_theta)
    if angle_from_upright < 0.5 and abs(theta_dot) < 2.0:
        switched = True
    
    if switched:
        # LQR controller for stabilization
        # Linearized system matrices around upright position
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, polemass * g / cartmass, 0, 0],
            [0, (cartmass + polemass) * g / (cartmass * polelength), 0, 0]
        ])
        
        B = np.array([
            [0],
            [0],
            [1 / cartmass],
            [1 / (cartmass * polelength)]
        ])
        
        # LQR weights
        Q = np.diag([10, 100, 1, 1])  # Penalize position and angle more
        R = np.array([[1]])
        
        try:
            # Solve Riccati equation
            P = solve_continuous_are(A, B, Q, R)
            K = R**-1 @ B.T @ P
            
            # Control law: u = -K(x - x_desired)
            state_error = np.array([x, theta - eq_theta, x_dot, theta_dot])
            u = -K @ state_error
            u = float(u[0])
        except (np.linalg.LinAlgError, ValueError):
            u = 0.0
    else:
        # Energy-based swingup control
        energy_error = total_energy - desired_energy
        
        # Control law based on energy and position
        if abs(energy_error) < 0.1:
            u = 0.0
        else:
            # Swing up control
            u = -np.sign(energy_error) * np.sign(theta_dot * np.cos(theta)) * 10.0
    
    # Clamp control output
    u = np.clip(u, -50, 50)
    
    # Final NaN check
    if np.isnan(u) or np.isinf(u):
        u = 0.0
    
    return u, switched


def cartpole_dynamics(state, u, cartmass, polemass, polelength):
    """
    Cartpole dynamics with numerical stability improvements
    """
    # Check for NaN or inf values
    if np.any(np.isnan(state)) or np.isnan(u) or np.isinf(u):
        return np.zeros(4)
    
    x, theta, x_dot, theta_dot = state
    
    # Clamp inputs to prevent numerical issues
    x = np.clip(x, -10, 10)
    theta = np.clip(theta, -100*np.pi, 100*np.pi)
    x_dot = np.clip(x_dot, -50, 50)
    theta_dot = np.clip(theta_dot, -50, 50)
    u = np.clip(u, -50, 50)
    
    g = 9.81
    total_mass = cartmass + polemass
    polemass_length = polemass * polelength
    
    # Precompute trigonometric functions
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Avoid division by very small numbers
    cos_theta = np.clip(cos_theta, -0.999, 0.999)
    
    # Compute dynamics
    try:
        temp = (u + polemass_length * theta_dot**2 * sin_theta) / total_mass
        numer = g * sin_theta - cos_theta * temp
        denom = polelength * (4.0/3.0 - polemass * cos_theta**2 / total_mass)
        
        # Check for near-zero denominator
        if abs(denom) < 1e-10:
            denom = 1e-10 * np.sign(denom) if denom != 0 else 1e-10
        
        theta_acc = numer / denom
        x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass
        
        # Check for numerical issues
        if np.isnan(theta_acc) or np.isinf(theta_acc):
            theta_acc = 0.0
        if np.isnan(x_acc) or np.isinf(x_acc):
            x_acc = 0.0
        
        # Clamp accelerations
        theta_acc = np.clip(theta_acc, -100, 100)
        x_acc = np.clip(x_acc, -100, 100)
        
        return np.array([x_dot, theta_dot, x_acc, theta_acc])
    
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.zeros(4)


def rk4_step(state, u, dt, cartmass, polemass, polelength):
    """
    Runge-Kutta 4th order integration step with stability checks
    """
    k1 = cartpole_dynamics(state, u, cartmass, polemass, polelength)
    k2 = cartpole_dynamics(state + dt/2 * k1, u, cartmass, polemass, polelength)
    k3 = cartpole_dynamics(state + dt/2 * k2, u, cartmass, polemass, polelength)
    k4 = cartpole_dynamics(state + dt * k3, u, cartmass, polemass, polelength)
    
    new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Check for NaN or inf values
    if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
        return state  # Return previous state if numerical issues
    
    # Clamp state to reasonable bounds
    new_state[0] = np.clip(new_state[0], -10, 10)  # x position
    new_state[1] = np.clip(new_state[1], -100*np.pi, 100*np.pi)  # theta
    new_state[2] = np.clip(new_state[2], -50, 50)  # x_dot
    new_state[3] = np.clip(new_state[3], -50, 50)  # theta_dot
    
    return new_state


def run_single_trajectory_fast(cartmass, polemass, polelength, initial_state, n_points):
    """
    Run a single trajectory using direct physics computation
    """
    dt = 0.02  # 50Hz control rate
    
    # Initialize
    state = np.array(initial_state, dtype=np.float64)
    switched = False
    
    states = np.zeros((n_points, 4))
    controls = np.zeros(n_points)
    modes = np.zeros(n_points)
    
    for i in range(n_points):
        # Store current state
        states[i] = state.copy()
        
        # Get control action
        u, switched = swingup_lqr_controller(state, switched, cartmass, polemass, polelength)
        controls[i] = u
        modes[i] = 1 if switched else 0
        
        # Check for trajectory failure (NaN or extreme values)
        if (np.any(np.isnan(state)) or np.any(np.isinf(state)) or 
            abs(state[0]) > 8 or abs(state[2]) > 40 or abs(state[3]) > 40):
            # Trajectory failed, return partial data
            return states[:i+1], controls[:i+1], modes[:i+1]
        
        # Integrate dynamics
        state = rk4_step(state, u, dt, cartmass, polemass, polelength)
    
    return states, controls, modes


def generate_batch_fast(bsize, cartmass, polemass, polelength, n_points, device='cpu'):
    """
    Generate a batch of trajectories using fast physics computation
    """
    batch_states = []
    batch_controls = []
    
    for i in range(bsize):
        # Generate random initial conditions
        max_retries = 5
        for retry in range(max_retries):
            # Random initial state
            x0 = np.random.uniform(-2, 2)
            theta0 = np.random.uniform(-np.pi, np.pi)
            x_dot0 = np.random.uniform(-2, 2)
            theta_dot0 = np.random.uniform(-2, 2)
            
            initial_state = [x0, theta0, x_dot0, theta_dot0]
            
            try:
                states, controls, modes = run_single_trajectory_fast(
                    cartmass[i], polemass[i], polelength[i], initial_state, n_points
                )
                
                # Check if trajectory completed successfully
                if len(states) == n_points and not np.any(np.isnan(states)) and not np.any(np.isnan(controls)):
                    # Pad controls to match n_points
                    if len(controls) < n_points:
                        controls = np.pad(controls, (0, n_points - len(controls)), 'edge')
                    
                    # Combine control and mode
                    control_values = np.column_stack([controls, modes])
                    
                    batch_states.append(states.T)  # Transpose to (4, n_points)
                    batch_controls.append(control_values)
                    break
                    
            except Exception as e:
                if retry == max_retries - 1:
                    # Use fallback trajectory
                    states = np.zeros((n_points, 4))
                    controls = np.zeros(n_points)
                    modes = np.zeros(n_points)
                    control_values = np.column_stack([controls, modes])
                    
                    batch_states.append(states.T)
                    batch_controls.append(control_values)
                continue
    
    # Convert to tensors
    xs = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
    ys = torch.tensor(np.array(batch_controls), dtype=torch.float32, device=device)
    
    return xs, ys


def save_pickle(data, pickle_path):
    """Saves data to a pickle file."""
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def append_to_dataset_logger(iteration, cartmass, polemass, polelength, xs_shape, log_file):
    """Appends simulation details to a log file for tracking."""
    with open(log_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("Cartmass: " + str(cartmass) + "\n")
        f.write("Polemass: " + str(polemass) + "\n")
        f.write("Polelength: " + str(polelength) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """Generate training data"""
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_filesfolder, args.pickle_folder), exist_ok=True)
    
    curriculum = Curriculum(args.training.curriculum)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for i in tqdm(range(args.training.train_steps)):
        # Sample random system parameters
        cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(
            size=args.training.batch_size,
            cartmass=2.0,
            polemasslowerbound=0.3,
            polemassupperbound=1.0,
            polelengthlowerbound=1.0,
            polelengthupperbound=1.5
        )
        
        # Generate batch
        xs, control_values = generate_batch_fast(
            args.training.batch_size, cartmass, polemass, polelength,
            curriculum.n_points, device=device
        )
        
        # Convert to numpy for saving
        xs_np = xs.cpu().numpy()
        control_values_np = control_values.cpu().numpy()
        
        # Save as pickle
        pickle_path = os.path.join(args.dataset_filesfolder, args.pickle_folder, f"batch_{i}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump((xs_np, control_values_np, cartmass, polemass, polelength), f)


def make_test_data(args):
    """Generate test data"""
    os.makedirs(os.path.join(args.dataset_filesfolder, args.pickle_folder_test), exist_ok=True)
    
    curriculum = Curriculum(args.training.curriculum)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for i in tqdm(range(args.training.test_pendulums)):
        # Sample random system parameters
        cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(
            size=args.training.batch_size,
            cartmass=2.0,
            polemasslowerbound=0.3,
            polemassupperbound=1.0,
            polelengthlowerbound=1.0,
            polelengthupperbound=1.5
        )
        
        # Generate batch
        xs, control_values = generate_batch_fast(
            args.training.batch_size, cartmass, polemass, polelength,
            curriculum.n_points, device=device
        )
        
        # Convert to numpy for saving
        xs_np = xs.cpu().numpy()
        control_values_np = control_values.cpu().numpy()
        
        # Save as pickle
        pickle_path = os.path.join(args.dataset_filesfolder, args.pickle_folder_test, f"batch_{i}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump((xs_np, control_values_np, cartmass, polemass, polelength), f)


def main(args):
    """Main function"""
    print("Generating training data...")
    make_train_data(args)
    
    print("Generating test data...")
    make_test_data(args)
    
    print("Dataset generation complete!")


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    main(args) 
