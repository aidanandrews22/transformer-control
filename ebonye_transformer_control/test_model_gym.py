#!/usr/bin/env python3
"""
Script to test trained transformer model predictions in gym environment.
Loads a trained model and runs it in the cartpole environment to visualize performance.
"""

import os
import torch
import numpy as np
import yaml
import imageio
import matplotlib.pyplot as plt
from gym_continuous_cartpole import ContinuousCartPoleEnv
from gym_cartpole_swingup_lqr import swingup_lqr_controller
import models

def load_model(run_path, epoch=3, step=3000):
    """Load trained model from checkpoint"""
    print(f"Loading model from {run_path}")
    
    # Load config
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    conf_ns = dict_to_namespace(conf)
    
    # Build model
    model = models.build_model(conf_ns.model)
    
    # Load checkpoint
    checkpoint_path = os.path.join(run_path, f"checkpoint_epoch{epoch}_step{step}.pt")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        # Try loading from state.pt
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location="cpu")
        state_dict = state["model_state_dict"]
    
    # Remove "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, conf, device

def run_model_episode(model, device, env, initial_state, max_steps=500, context_length=560):
    """
    Run a single episode using the trained model for control prediction.
    Uses a rolling window of past states/controls as context.
    """
    obs, _ = env.reset(options={"init_state": initial_state})
    
    states = [obs.copy()]
    controls = [0.0]  # Start with zero control
    modes = [0.0]     # Start with swing-up mode
    rewards = []
    frames = []
    
    for step in range(max_steps):
        # Render current frame
        frame = env.render()
        frames.append(frame)
        
        # Prepare model input (use last context_length steps)
        start_idx = max(0, len(states) - context_length)
        
        # Create input tensors
        xs = torch.tensor(states[start_idx:], dtype=torch.float32, device=device).unsqueeze(0)  # (1, seq_len, 4)
        ys = torch.tensor(list(zip(controls[start_idx:], modes[start_idx:])), dtype=torch.float32, device=device).unsqueeze(0)  # (1, seq_len, 2)
        
        # Pad sequences if needed
        if xs.shape[1] < context_length:
            pad_length = context_length - xs.shape[1]
            xs = torch.cat([torch.zeros(1, pad_length, 4, device=device), xs], dim=1)
            ys = torch.cat([torch.zeros(1, pad_length, 2, device=device), ys], dim=1)
        
        # Get model prediction
        with torch.no_grad():
            # Use regular forward pass instead of inference mode
            control_pred, state_pred = model(xs, ys)
            
            # Use the last prediction (most recent timestep)  
            predicted_control = control_pred[0, -1, 0].item()  # Extract control value
            predicted_mode = control_pred[0, -1, 1].item()    # Extract mode
            
        # Apply predicted control
        obs, reward, done, _, _, applied_action = env.step([predicted_control])
        
        # Store results
        states.append(obs.copy())
        controls.append(applied_action[0])
        modes.append(predicted_mode)
        rewards.append(reward)
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    return states, controls, modes, rewards, frames

def run_lqr_episode(env, initial_state, cartmass, polemass, polelength, max_steps=500):
    """Run episode using the original LQR controller for comparison"""
    obs, _ = env.reset(options={"init_state": initial_state})
    
    states = [obs.copy()]
    controls = []
    modes = []
    rewards = []
    frames = []
    switched = False
    
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)
        
        # Get LQR control
        action, switched = swingup_lqr_controller(obs, switched, cartmass, polemass, polelength)
        obs, reward, done, _, _, applied_action = env.step(action)
        
        states.append(obs.copy())
        controls.append(applied_action[0])
        modes.append(1.0 if switched else 0.0)
        rewards.append(reward)
        
        if done:
            print(f"LQR episode finished at step {step}")
            break
    
    return states, controls, modes, rewards, frames

def plot_comparison(model_results, lqr_results, save_path="comparison.png"):
    """Plot comparison between model and LQR performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    model_states, model_controls, model_modes, model_rewards = model_results
    lqr_states, lqr_controls, lqr_modes, lqr_rewards = lqr_results
    
    # Convert to numpy arrays
    model_states = np.array(model_states)
    lqr_states = np.array(lqr_states)
    
    time_model = np.arange(len(model_states)) * 0.025  # 25ms timestep
    time_lqr = np.arange(len(lqr_states)) * 0.025
    
    # Plot states
    state_labels = ['Cart Position (x)', 'Cart Velocity', 'Pole Angle (θ)', 'Pole Angular Velocity']
    for i, label in enumerate(state_labels):
        ax = axes[i//2, i%2]
        ax.plot(time_model, model_states[:, i], label='Transformer Model', linewidth=2)
        ax.plot(time_lqr, lqr_states[:, i], label='LQR Controller', linewidth=2, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    # run_path = "./output_gym_full/5bb7f256-6136-4fe6-a4af-5fc369080866/"
    run_path = "./output_gym_full/cf6e284b-99f0-4623-a641-04e3c402cc37/"
    
    # Load trained model
    model, conf, device = load_model(run_path, epoch=100, step=100000)
    
    # Environment parameters (should match training data)
    cartmass = 2.0
    polemass = 0.5
    polelength = 1.0
    
    # Initialize environment
    env = ContinuousCartPoleEnv(
        masscart=cartmass, 
        masspole=polemass, 
        length=polelength,
        render_mode="rgb_array"
    )

    if hasattr(env, 'screen_width'):
        env.screen_width = 1200  # Make wider
    
    # Test initial condition (inverted pendulum)
    initial_state = [0.0, 0.0, np.pi, 0.0]  # [x, x_dot, theta, theta_dot]
    
    print("Running transformer model episode...")
    model_results = run_model_episode(model, device, env, initial_state, context_length=560)
    model_states, model_controls, model_modes, model_rewards, model_frames = model_results
    
    print(f"Model episode: {len(model_states)} steps, total reward: {sum(model_rewards):.2f}")
    
    print("Running LQR controller episode...")
    lqr_results = run_lqr_episode(env, initial_state, cartmass, polemass, polelength)
    lqr_states, lqr_controls, lqr_modes, lqr_rewards, lqr_frames = lqr_results
    
    print(f"LQR episode: {len(lqr_states)} steps, total reward: {sum(lqr_rewards):.2f}")
    
    # Save videos (skip if ffmpeg not available)
    print("Saving videos...")
    try:
        imageio.mimsave('transformer_model_cartpole.mp4', model_frames, fps=40)
        imageio.mimsave('lqr_controller_cartpole.mp4', lqr_frames, fps=40)
        print("Videos saved successfully!")
    except ValueError:
        print("Video saving failed - ffmpeg not available. Saving sample frames instead...")
        # Save sample frames
        imageio.imwrite('transformer_model_frame_0.png', model_frames[0])
        imageio.imwrite('transformer_model_frame_final.png', model_frames[-1])
        imageio.imwrite('lqr_controller_frame_0.png', lqr_frames[0])
        imageio.imwrite('lqr_controller_frame_final.png', lqr_frames[-1])
    
    # Plot comparison
    print("Plotting comparison...")
    plot_comparison(
        (model_states, model_controls, model_modes, model_rewards),
        (lqr_states, lqr_controls, lqr_modes, lqr_rewards)
    )
    
    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Transformer Model:")
    print(f"  Steps: {len(model_states)}")
    print(f"  Total Reward: {sum(model_rewards):.2f}")
    print(f"  Average Reward: {np.mean(model_rewards):.2f}")
    print(f"  Final Pole Angle: {model_states[-1][2]:.3f} rad")
    print(f"  Final Cart Position: {model_states[-1][0]:.3f}")
    
    print(f"\nLQR Controller:")
    print(f"  Steps: {len(lqr_states)}")
    print(f"  Total Reward: {sum(lqr_rewards):.2f}")
    print(f"  Average Reward: {np.mean(lqr_rewards):.2f}")
    print(f"  Final Pole Angle: {lqr_states[-1][2]:.3f} rad")
    print(f"  Final Cart Position: {lqr_states[-1][0]:.3f}")

if __name__ == "__main__":
    main()
