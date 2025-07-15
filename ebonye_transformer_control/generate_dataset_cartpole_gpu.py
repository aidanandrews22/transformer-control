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
torch.set_float32_matmul_precision("high")  # Silence torch.compile warnings

# Pre-compute LQR gains cache to eliminate CPU bottleneck
_lqr_gains_cache = {}

def get_lqr_gains(cartmass, polemass, polelength, device='cuda'):
    """Pre-compute and cache LQR gains for given system parameters"""
    key = (float(cartmass), float(polemass), float(polelength))
    
    if key not in _lqr_gains_cache:
        g = 9.81
        
        # Linearized system matrices around upright equilibrium
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, polemass * g / cartmass, 0],
            [0, 0, 0, 1],
            [0, 0, (cartmass + polemass) * g / (cartmass * polelength), 0]
        ])
        
        B = np.array([
            [0],
            [1 / cartmass],
            [0],
            [1 / (cartmass * polelength)]
        ])
        
        Q = np.diag([10, 100, 1, 1])
        R = np.array([[1]])
        
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            _lqr_gains_cache[key] = torch.tensor(K[0], device=device, dtype=torch.float32)
        except:
            # Fallback analytic gain for inverted pendulum
            _lqr_gains_cache[key] = torch.tensor([
                1.0,  # x position
                2.0,  # x velocity  
                10.0 / polelength,  # theta error
                2.0 * torch.sqrt(torch.tensor(g / polelength, device=device))  # theta_dot
            ], device=device, dtype=torch.float32)
    
    return _lqr_gains_cache[key]


def get_valid_masses_and_lengths_uniform(size=1, cartmass=2, polemasslowerbound=0.3, polemassupperbound=1.0, 
                                       polelengthlowerbound=1.0, polelengthupperbound=1.5):
    """Samples valid masses and lengths for the cartpole system using a uniform distribution."""
    cartmass = np.ones(size) * cartmass
    polemass = np.random.uniform(polemasslowerbound, polemassupperbound, size)
    polelength = np.random.uniform(polelengthlowerbound, polelengthupperbound, size)
    return cartmass, polemass, polelength


def find_nearest_upright_batch(theta):
    """Find the nearest upright position for batch of angles"""
    return 2 * np.pi * torch.round(theta / (2 * np.pi))


def swingup_lqr_controller_batch(states, switched, cartmass, polemass, polelength, device='cuda'):
    """
    Batched version of swingup_lqr_controller for GPU parallelization
    Fixed state ordering to match gym env: [x, x_dot, theta, theta_dot]
    
    Args:
        states: torch.Tensor of shape (batch_size, 4) - [x, x_dot, theta, theta_dot]
        switched: torch.Tensor of shape (batch_size,) - boolean flags for LQR mode
        cartmass: torch.Tensor of shape (batch_size,) - cart masses
        polemass: torch.Tensor of shape (batch_size,) - pole masses  
        polelength: torch.Tensor of shape (batch_size,) - pole lengths
        
    Returns:
        u: torch.Tensor of shape (batch_size,) - control actions
        switched: torch.Tensor of shape (batch_size,) - updated boolean flags
    """
    batch_size = states.shape[0]
    # Fixed state ordering to match gym env
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    
    g = 9.81
    
    # Find nearest upright position
    eq_theta = find_nearest_upright_batch(theta)
    
    # Calculate energy for swing-up control
    kinetic_energy = 0.5 * polemass * polelength**2 * theta_dot**2
    potential_energy = polemass * g * polelength * (1 - torch.cos(theta - eq_theta))
    total_energy = kinetic_energy + potential_energy
    desired_energy = polemass * g * polelength * 2
    
    # Switch to LQR if close to upright and low energy
    angle_from_upright = torch.abs(theta - eq_theta)
    switch_condition = (angle_from_upright < 0.5) & (torch.abs(theta_dot) < 2.0)
    switched = switched | switch_condition
    
    # Initialize control outputs
    u = torch.zeros(batch_size, device=device)
    
    # LQR control for switched trajectories - using pre-computed gains
    lqr_mask = switched
    if lqr_mask.any():
        lqr_indices = torch.where(lqr_mask)[0]
        
        # Batch process LQR gains (pre-computed, no SciPy calls)
        for i in lqr_indices:
            K = get_lqr_gains(cartmass[i].item(), polemass[i].item(), polelength[i].item(), device)
            
            state_error = torch.tensor([
                x[i].item(), 
                x_dot[i].item(),
                theta[i].item() - eq_theta[i].item(), 
                theta_dot[i].item()
            ], device=device)
            
            u[i] = -torch.sum(K * state_error)
    
    # Energy-based swing-up control for non-switched trajectories
    swingup_mask = ~switched
    if swingup_mask.any():
        energy_error = total_energy[swingup_mask] - desired_energy[swingup_mask]
        
        # Control law based on energy and position
        u_swingup = torch.zeros(swingup_mask.sum(), device=device)
        
        # Only apply control if energy error is significant
        significant_error = torch.abs(energy_error) > 0.1
        if significant_error.any():
            theta_swingup = theta[swingup_mask][significant_error]
            theta_dot_swingup = theta_dot[swingup_mask][significant_error]
            energy_error_sig = energy_error[significant_error]
            
            u_swingup[significant_error] = -torch.sign(energy_error_sig) * torch.sign(theta_dot_swingup * torch.cos(theta_swingup)) * 10.0
        
        u[swingup_mask] = u_swingup
    
    # Clamp control output
    u = torch.clamp(u, -50, 50)
    
    return u, switched


def cartpole_dynamics_batch(states, u, cartmass, polemass, polelength, device='cuda'):
    """
    Batched cartpole dynamics computation on GPU
    Fixed state ordering to match gym env: [x, x_dot, theta, theta_dot]
    
    Args:
        states: torch.Tensor of shape (batch_size, 4) - [x, x_dot, theta, theta_dot]
        u: torch.Tensor of shape (batch_size,) - control inputs
        cartmass: torch.Tensor of shape (batch_size,) - cart masses
        polemass: torch.Tensor of shape (batch_size,) - pole masses
        polelength: torch.Tensor of shape (batch_size,) - pole lengths
        
    Returns:
        derivatives: torch.Tensor of shape (batch_size, 4) - state derivatives
    """
    # Fixed state ordering to match gym env
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    
    g = 9.81
    total_mass = cartmass + polemass
    polemass_length = polemass * polelength
    
    # Precompute trigonometric functions
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Compute dynamics (exact same equations as gym env)
    temp = (u + polemass_length * theta_dot**2 * sin_theta) / total_mass
    numer = g * sin_theta - cos_theta * temp
    denom = polelength * (4.0/3.0 - polemass * cos_theta**2 / total_mass)
    
    theta_acc = numer / denom
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass
    
    # Return derivatives in gym env order: [x_dot, x_acc, theta_dot, theta_acc]
    derivatives = torch.stack([x_dot, x_acc, theta_dot, theta_acc], dim=1)
    
    return derivatives


def rk4_step_batch(states, u, dt, cartmass, polemass, polelength, device='cuda'):
    """
    Batched RK4 integration step on GPU
    """
    k1 = cartpole_dynamics_batch(states, u, cartmass, polemass, polelength, device)
    k2 = cartpole_dynamics_batch(states + dt/2 * k1, u, cartmass, polemass, polelength, device)
    k3 = cartpole_dynamics_batch(states + dt/2 * k2, u, cartmass, polemass, polelength, device)
    k4 = cartpole_dynamics_batch(states + dt * k3, u, cartmass, polemass, polelength, device)
    
    new_states = states + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return new_states


@torch.compile(backend="inductor")  # Fuse entire simulation loop into single CUDA graph
def generate_batch_gpu(batch_size, cartmass, polemass, polelength, n_points, device='cuda'):
    """
    GPU-parallelized batch generation using torch tensors with kernel fusion
    """
    # Convert to torch tensors on GPU
    cartmass = torch.tensor(cartmass, dtype=torch.float32, device=device)
    polemass = torch.tensor(polemass, dtype=torch.float32, device=device)
    polelength = torch.tensor(polelength, dtype=torch.float32, device=device)
    
    # Generate random initial conditions on GPU
    x0 = torch.rand(batch_size, device=device) * 4 - 2  # Uniform(-2, 2)
    x_dot0 = torch.rand(batch_size, device=device) * 4 - 2  # Uniform(-2, 2)
    theta0 = torch.rand(batch_size, device=device) * 2 * np.pi - np.pi  # Uniform(-π, π)
    theta_dot0 = torch.rand(batch_size, device=device) * 4 - 2  # Uniform(-2, 2)
    
    # Initialize state tensor in gym env order: [x, x_dot, theta, theta_dot]
    states = torch.stack([x0, x_dot0, theta0, theta_dot0], dim=1)  # (batch_size, 4)
    
    # Initialize storage tensors
    all_states = torch.zeros((batch_size, 4, n_points), device=device)
    all_controls = torch.zeros((batch_size, n_points), device=device)
    all_modes = torch.zeros((batch_size, n_points), device=device)
    
    # Initialize control state
    switched = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    dt = 0.025  # Match original timestep
    
    # Run simulation - this entire loop gets compiled into single CUDA graph
    for i in range(n_points):
        # Store current state
        all_states[:, :, i] = states
        
        # Get control actions
        u, switched = swingup_lqr_controller_batch(
            states, switched, cartmass, polemass, polelength, device
        )
        
        # Store control and mode
        all_controls[:, i] = u
        all_modes[:, i] = switched.float()
        
        # Integrate dynamics
        if i < n_points - 1:  # Don't integrate on last step
            states = rk4_step_batch(states, u, dt, cartmass, polemass, polelength, device)
    
    # Combine controls and modes
    control_values = torch.stack([all_controls, all_modes], dim=-1)  # (batch_size, n_points, 2)
    
    return all_states, control_values


def make_train_data(args):
    """Generate training data using GPU parallelization"""
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_filesfolder, args.pickle_folder), exist_ok=True)
    
    curriculum = Curriculum(args.training.curriculum)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Torch compile enabled for kernel fusion")
    
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
        
        # Generate batch on GPU
        xs, control_values = generate_batch_gpu(
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
    """Generate test data using GPU parallelization"""
    os.makedirs(os.path.join(args.dataset_filesfolder, args.pickle_folder_test), exist_ok=True)
    
    curriculum = Curriculum(args.training.curriculum)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        
        # Generate batch on GPU
        xs, control_values = generate_batch_gpu(
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
