#!/usr/bin/env python3
"""
Enhanced visualization script for transformer model predictions in gym environment.
Features larger window and camera following the cartpole.
"""

import os
import torch
import numpy as np
import yaml
import imageio
import matplotlib.pyplot as plt
import pygame
import sys
from gym_continuous_cartpole import ContinuousCartPoleEnv
from gym_cartpole_swingup_lqr import swingup_lqr_controller
import models

class LargeCartPoleEnv(ContinuousCartPoleEnv):
    """Enhanced CartPole environment with larger rendering and camera following"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Much larger screen for better visualization
        self.screen_width = 1400
        self.screen_height = 900
        self.follow_cart = True  # Camera follows the cart
        
    def render(self):
        if self.state is None:
            return None
            
        x, x_dot, theta, theta_dot = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Enhanced CartPole - Transformer Model")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Dynamic world width based on cart position if following
        if self.follow_cart:
            # Center the view on the cart with some padding
            view_range = 8.0  # How much of the world to show around the cart
            world_center = x[0]
            world_left = world_center - view_range/2
            world_right = world_center + view_range/2
            world_width = view_range
        else:
            world_width = self.x_threshold * 2
            world_left = -self.x_threshold
            world_right = self.x_threshold
            
        scale = self.screen_width / world_width
        
        # Create surface
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((135, 206, 235))  # Sky blue background

        # Ground line
        carty = self.screen_height * 0.7  # Lower the ground
        
        # Cart position on screen
        if self.follow_cart:
            cartx = self.screen_width / 2.0  # Cart always in center when following
        else:
            cartx = (x[0] - world_left) * scale
        
        # Draw a longer ground line
        ground_y = int(carty)
        pygame.draw.line(self.surf, (101, 67, 33), (0, ground_y), (self.screen_width, ground_y), 3)
        
        # Draw grid for better depth perception
        for i in range(-20, 21):
            world_x = world_left + i * (world_width / 40)
            screen_x = int((world_x - world_left) * scale)
            if 0 <= screen_x <= self.screen_width:
                pygame.draw.line(self.surf, (200, 200, 200), (screen_x, 0), (screen_x, self.screen_height), 1)
        
        # Cart dimensions (larger for better visibility)
        cart_width = 60
        cart_height = 40
        
        # Draw cart
        cart_rect = pygame.Rect(cartx - cart_width/2, carty - cart_height/2, cart_width, cart_height)
        pygame.draw.rect(self.surf, (50, 50, 50), cart_rect)
        pygame.draw.rect(self.surf, (0, 0, 0), cart_rect, 3)
        
        # Draw cart wheels
        wheel_radius = 8
        wheel_y = carty + cart_height/2 - wheel_radius
        pygame.draw.circle(self.surf, (100, 100, 100), (int(cartx - cart_width/3), int(wheel_y)), wheel_radius)
        pygame.draw.circle(self.surf, (100, 100, 100), (int(cartx + cart_width/3), int(wheel_y)), wheel_radius)
        pygame.draw.circle(self.surf, (0, 0, 0), (int(cartx - cart_width/3), int(wheel_y)), wheel_radius, 2)
        pygame.draw.circle(self.surf, (0, 0, 0), (int(cartx + cart_width/3), int(wheel_y)), wheel_radius, 2)
        
        # Pole
        pole_length = scale * (2 * self.length)
        pole_width = 8
        
        # Pole end position
        pole_end_x = cartx + pole_length * np.sin(theta)
        pole_end_y = carty - pole_length * np.cos(theta)
        
        # Draw pole
        pygame.draw.line(self.surf, (139, 69, 19), (int(cartx), int(carty)), (int(pole_end_x), int(pole_end_y)), pole_width)
        
        # Draw pole pivot
        pygame.draw.circle(self.surf, (255, 0, 0), (int(cartx), int(carty)), 6)
        
        # Draw pole tip
        pygame.draw.circle(self.surf, (0, 0, 255), (int(pole_end_x), int(pole_end_y)), 8)
        
        # Add information overlay
        font = pygame.font.Font(None, 36)
        info_texts = [
            f"Cart Position: {x[0]:.2f}",
            f"Cart Velocity: {x_dot:.2f}",
            f"Pole Angle: {theta:.2f} rad ({np.degrees(theta):.1f}°)",
            f"Pole Angular Velocity: {theta_dot:.2f}",
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.surf.blit(text_surface, (10, 10 + i * 40))
        
        # Add center crosshair when following
        if self.follow_cart:
            pygame.draw.line(self.surf, (255, 0, 0), (self.screen_width//2 - 20, self.screen_height//2), (self.screen_width//2 + 20, self.screen_height//2), 2)
            pygame.draw.line(self.surf, (255, 0, 0), (self.screen_width//2, self.screen_height//2 - 20), (self.screen_width//2, self.screen_height//2 + 20), 2)
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

def load_model(run_path, epoch=100, step=100000):
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

def run_model_episode_live(model, device, env, initial_state, max_steps=1000, context_length=128):
    """Run episode with live visualization"""
    obs, _ = env.reset(options={"init_state": initial_state})
    
    states = [obs.copy()]
    controls = [0.0]
    modes = [0.0]
    
    print("\\n=== LIVE CARTPOLE VISUALIZATION ===")
    print("Watch the cartpole balance! Press Ctrl+C to stop early.")
    print("The camera will follow the cart as it moves.")
    
    try:
        for step in range(max_steps):
            # Render current state
            env.render()
            
            # Small delay to make it visible
            import time
            time.sleep(0.05)
            
            # Prepare model input
            start_idx = max(0, len(states) - context_length)
            
            # Create input tensors
            xs = torch.tensor(np.array(states[start_idx:]), dtype=torch.float32, device=device).unsqueeze(0)
            ys = torch.tensor(np.array(list(zip(controls[start_idx:], modes[start_idx:]))), dtype=torch.float32, device=device).unsqueeze(0)
            
            # Pad sequences if needed
            if xs.shape[1] < context_length:
                pad_length = context_length - xs.shape[1]
                xs = torch.cat([torch.zeros(1, pad_length, 4, device=device), xs], dim=1)
                ys = torch.cat([torch.zeros(1, pad_length, 2, device=device), ys], dim=1)
            
            # Get model prediction
            with torch.no_grad():
                control_pred, state_pred = model(xs, ys)
                predicted_control = control_pred[0, -1, 0].item()
                predicted_mode = control_pred[0, -1, 1].item()
            
            # Apply predicted control
            obs, reward, done, _, _, applied_action = env.step([predicted_control])
            
            # Store results
            states.append(obs.copy())
            controls.append(applied_action[0])
            modes.append(predicted_mode)
            
            # Print some info every 50 steps
            if step % 50 == 0:
                print(f"Step {step}: Cart pos={obs[0]:.2f}, Pole angle={obs[2]:.2f} rad, Control={predicted_control:.2f}")
            
            if done:
                print(f"Episode finished at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\\nVisualization stopped by user")
    
    return states, controls, modes

def main():
    # Configuration - update with your actual run path
    run_path = "./output_gym_full/cf6e284b-99f0-4623-a641-04e3c402cc37/"
    
    # Load trained model
    model, conf, device = load_model(run_path, epoch=100, step=100000)
    
    # Environment parameters
    cartmass = 2.0
    polemass = 0.5
    polelength = 1.0
    
    # Initialize enhanced environment
    env = LargeCartPoleEnv(
        masscart=cartmass,
        masspole=polemass,
        length=polelength,
        render_mode="human"  # Use human mode for live visualization
    )
    
    # Test initial condition (inverted pendulum)
    initial_state = [0.0, 0.0, np.pi, 0.0]  # [x, x_dot, theta, theta_dot]
    
    print("Running enhanced transformer model visualization...")
    print("The window will be large and the camera will follow the cartpole.")
    
    # Run with live visualization
    states, controls, modes = run_model_episode_live(model, device, env, initial_state)
    
    print(f"\\nVisualization complete! Total steps: {len(states)}")
    
    # Close environment
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()