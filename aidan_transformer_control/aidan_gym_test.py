import os
import math
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from aidan_gym import ContinuousCartPoleWrapper, CartPoleSwingUpController


def run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir):
    # Create base environment and wrap it with proper parameters
    base_env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = ContinuousCartPoleWrapper(
        base_env,
        max_force=500.0,  # Same as in run_complete_demonstration
        start_hanging=False,  # We'll set initial state manually
        mass_pole_modifier=masspole/0.1,  # Convert to modifier (default masspole is 0.1)
        mass_cart_modifier=masscart/1.0,  # Convert to modifier (default masscart is 1.0)
        length_modifier=length/0.5        # Convert to modifier (default length is 0.5)
    )
    
    # Create controller AFTER environment is fully set up
    controller = CartPoleSwingUpController(env)
    
    # Reset environment first to initialize properly
    obs, _ = env.reset()
    
    # Now manually set the initial state (don't use start_hanging option)
    env.env.unwrapped.state = np.array([0.0, 0.0, theta_init, thetadot_init])
    obs = env.env.unwrapped.state.copy()
    
    frames, states, actions = [], [obs], []
    max_steps = 560

    for _ in range(max_steps):
        # Get action from controller - this returns np.array([force])
        action = controller.get_action(obs)
        
        # Step environment with the action
        obs, reward, done, truncated, info = env.step(action)
        
        # Render and collect data
        frame = env.render()
        frames.append(frame)
        states.append(obs)
        actions.append(action[0])  # Extract scalar from array
        
        if done or truncated:
            break

    env.close()

    # Final state analysis
    states = np.array(states)
    actions = np.array(actions).squeeze()
    final_theta = states[-1, 2]
    final_theta_dot = states[-1, 3]
    
    # Check if stabilized (same logic as original)
    theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    
    # Output path and naming
    folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_th{theta_init:.2f}_dth{thetadot_init:.2f}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'cartpole.mp4'), frames, fps=50, macro_block_size=1)

    # Save control plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()

    # Save state plots
    plt.figure(figsize=(15, 10))
    labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    colors = ['red', 'green', 'orange', 'purple']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(range(len(states)), states[:, i], label=labels[i], color=colors[i], s=10)
        plt.title(labels[i] + ' Over Time')
        plt.xlabel('Time Step')
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'states.png'))
    plt.close()

    return {
        "path": path,
        "masscart": masscart,
        "masspole": masspole,
        "length": length,
        "theta_init": theta_init,
        "thetadot_init": thetadot_init,
        "stabilized": stabilized,
        "final_state": states[-1]
    }


if __name__ == "__main__":
    num_runs = 20
    results = []
    save_dir = os.path.join(os.getcwd(), 'videos', 'aidan_cartpole_gym_runs')
    os.makedirs(save_dir, exist_ok=True)

    for run_idx in range(num_runs):
        masscart = 2.0
        masspole = np.random.uniform(0.5, 1.0)
        length = np.random.uniform(1.0, 1.5)
        theta_init = np.random.uniform(np.pi - np.pi/2, np.pi + np.pi/2)
        thetadot_init = np.random.uniform(-1.0, 1.0)

        result = run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir)
        results.append(result)
        print(f"[{run_idx+1}/{num_runs}] Stabilized: {result['stabilized']} — {result['path']}")

    # Summary
    total_stabilized = sum(r['stabilized'] for r in results)
    print(f"\n✅ {total_stabilized} / {num_runs} systems stabilized.")

