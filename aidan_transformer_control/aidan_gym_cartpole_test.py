import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import gymnasium as gym
from aidan_gym import ContinuousCartPoleWrapper, CartPoleSwingUpController


def run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir):
    # Create environment using aidan_gym's wrapper
    base_env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = ContinuousCartPoleWrapper(
        base_env, 
        max_force=500.0,  # Same as in aidan_gym
        start_hanging=False,  # We'll set custom initial conditions
        mass_pole_modifier=masspole/0.1,  # Convert absolute mass to modifier (default is 0.1)
        mass_cart_modifier=masscart/1.0,  # Convert absolute mass to modifier (default is 1.0)
        length_modifier=length/0.5  # Convert absolute length to modifier (default is 0.5)
    )
    
    # Create controller
    controller = CartPoleSwingUpController(env)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Set custom initial state (matching gym_cartpole_test.py format)
    env.env.unwrapped.state = np.array([0.0, 0.0, theta_init, thetadot_init])
    obs = env.env.unwrapped.state.copy()
    
    frames, states, actions = [], [obs], []
    modes = []  # Track controller mode at each step
    max_steps = 560  # Same as gym_cartpole_test.py
    
    for step in range(max_steps):
        # Get action from controller
        action = controller.get_action(obs)
        modes.append(controller.current_mode)  # Track mode
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Render and store frame
        frame = env.render()
        frames.append(frame)
        states.append(obs)
        actions.append(action[0])  # Extract scalar from array
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Process results (same as gym_cartpole_test.py)
    states = np.array(states)
    actions = np.array(actions).squeeze()
    final_theta = states[-1, 2]
    final_theta_dot = states[-1, 3]
    
    # Check if stabilized (same criteria)
    theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    
    # Output path and naming (same format)
    folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_th{theta_init:.2f}_dth{thetadot_init:.2f}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)
    
    # Save video
    imageio.mimsave(os.path.join(path, 'cartpole.mp4'), frames, fps=50, macro_block_size=1)
    
    # Save control plot (same style)
    plt.figure(figsize=(10, 5))
    actions_arr = np.array(actions)
    modes_arr = np.array(modes)
    timesteps = np.arange(len(actions_arr))
    # Indices for each mode
    swingup_idx = np.where(modes_arr == 'Swing-up')[0]
    lqr_idx = np.where(modes_arr == 'LQR')[0]
    # Plot blue for Swing-up, red for LQR
    plt.scatter(timesteps[swingup_idx], actions_arr[swingup_idx], label='Swing-up', color='blue', s=10)
    plt.scatter(timesteps[lqr_idx], actions_arr[lqr_idx], label='LQR', color='red', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()
    
    # Save state plots (same style)
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
    num_runs = 100
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