import numpy as np
import gymnasium as gym
from gymnasium import spaces
from aidan_gym import ContinuousCartPoleWrapper, CartPoleSwingUpController
import json
import os
from datetime import datetime

def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

def run_debug_test(max_steps: int = 20000, mass_pole_modifier: float = 1.0, 
                   mass_cart_modifier: float = 1.0, length_modifier: float = 1.0,
                   theta_init: float = None, thetadot_init: float = None,
                   test_id: int = 0):
    """Run a single test with detailed logging"""
    
    # Create environment
    base_env = gym.make('CartPole-v1', render_mode=None)
    start_hanging = not (theta_init is not None or thetadot_init is not None)
    env = ContinuousCartPoleWrapper(base_env, max_force=500.0, start_hanging=start_hanging, 
                                   mass_pole_modifier=mass_pole_modifier, 
                                   mass_cart_modifier=mass_cart_modifier, 
                                   length_modifier=length_modifier)
    controller = CartPoleSwingUpController(env)
    
    # Data collection
    debug_data = {
        'test_id': test_id,
        'parameters': {
            'mass_pole_modifier': mass_pole_modifier,
            'mass_cart_modifier': mass_cart_modifier,
            'length_modifier': length_modifier,
            'theta_init': theta_init,
            'thetadot_init': thetadot_init,
            'actual_mass_pole': float(env.masspole),
            'actual_length': float(env.length),
            'actual_mass_cart': float(env.masscart),
        },
        'termination_reason': None,
        'final_step': 0,
        'max_balanced_duration': 0,
        'observations_out_of_bounds': [],
        'position_bounds_exceeded': [],
        'max_position': 0,
        'min_position': 0,
        'success': False,
        'observation_space_bounds': {
            'low': env.observation_space.low.tolist(),
            'high': env.observation_space.high.tolist()
        }
    }
    
    # Reset
    state, _ = env.reset()
    
    # Set custom initial conditions if provided
    if theta_init is not None or thetadot_init is not None:
        initial_theta = theta_init if theta_init is not None else np.pi + np.random.uniform(-0.1, 0.1)
        initial_thetadot = thetadot_init if thetadot_init is not None else np.random.uniform(-0.5, 0.5)
        
        env.env.unwrapped.state = np.array([
            0.0,  # x
            0.0,  # x_dot
            initial_theta,  # theta
            initial_thetadot  # theta_dot
        ])
        state = env.env.unwrapped.state.copy()
    
    # Track balancing
    upright_start_time = None
    balanced_duration = 0
    max_balanced_duration = 0
    
    # Get observation space bounds
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    
    for step in range(max_steps):
        # Check if observation is in bounds
        if np.any(state < obs_low) or np.any(state > obs_high):
            out_of_bounds = {
                'step': step,
                'state': state.tolist(),
                'violations': []
            }
            for i, (val, low, high) in enumerate(zip(state, obs_low, obs_high)):
                if val < low or val > high:
                    out_of_bounds['violations'].append({
                        'index': i,
                        'value': float(val),
                        'low': float(low),
                        'high': float(high)
                    })
            debug_data['observations_out_of_bounds'].append(out_of_bounds)
        
        # Track position
        x_pos = state[0]
        debug_data['max_position'] = max(debug_data['max_position'], x_pos)
        debug_data['min_position'] = min(debug_data['min_position'], x_pos)
        
        if abs(x_pos) > env.env.unwrapped.x_threshold:
            debug_data['position_bounds_exceeded'].append({
                'step': step,
                'position': float(x_pos),
                'threshold': float(env.env.unwrapped.x_threshold)
            })
        
        # Get control action
        action = controller.get_action(state)
        
        # Step environment
        try:
            state, reward, terminated, truncated, _ = env.step(action)
        except Exception as e:
            debug_data['termination_reason'] = f'Exception: {str(e)}'
            debug_data['final_step'] = step
            break
        
        # Track balancing
        theta_upright = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
        is_balanced = abs(theta_upright) < 0.1
        
        if is_balanced:
            if upright_start_time is None:
                upright_start_time = step
            balanced_duration = step - upright_start_time
            max_balanced_duration = max(max_balanced_duration, balanced_duration)
            
            if balanced_duration >= 50:
                debug_data['success'] = True
                debug_data['termination_reason'] = 'Success - balanced for 50 steps'
                debug_data['final_step'] = step
                debug_data['max_balanced_duration'] = max_balanced_duration
                break
        else:
            if upright_start_time is not None:
                upright_start_time = None
                balanced_duration = 0
        
        if terminated:
            debug_data['termination_reason'] = f'Environment terminated at step {step}'
            debug_data['final_step'] = step
            debug_data['max_balanced_duration'] = max_balanced_duration
            break
        
        if step >= max_steps - 1:
            debug_data['termination_reason'] = 'Max steps reached'
            debug_data['final_step'] = step
            debug_data['max_balanced_duration'] = max_balanced_duration
            break
    
    env.close()
    
    # Convert all data to serializable format
    debug_data = convert_to_serializable(debug_data)
    
    return debug_data

def main():
    """Run debug tests on failed parameters"""
    
    # Failed parameters from the test run
    failed_params = [
        (5.100882068953918, 2.0759367486473232, 2.9741563463513883, 0.16603461819573084),
        (6.063688670012266, 2.8919420402202576, 3.2468848838882973, -0.9966636803909399),
        (5.22691247171182, 2.8427557219932766, 4.658246389244559, 0.6573301358100474),
        (5.54210569275412, 2.905097269151479, 4.409178789253825, -0.8091895433387919),
        (6.512506648504499, 2.9943810021126716, 1.636108493201009, 0.038966840843770445),
        (5.539759653664504, 2.3310878736112537, 3.019374047242246, -0.17408863901412253),
        (5.961468833027099, 2.9633298437715787, 2.304344641312192, 0.9684678905776576)
    ]
    
    # Create logs directory
    os.makedirs('aidan_transformer_control/diagnostic_logs', exist_ok=True)
    
    print("Running debug tests on failed parameters...")
    print("=" * 60)
    
    for i, params in enumerate(failed_params):
        mass_pole_modifier, length_modifier, theta_init, thetadot_init = params
        print(f"\nTest {i+1}/{len(failed_params)}:")
        print(f"  mass_pole={mass_pole_modifier:.2f}, length={length_modifier:.2f}")
        print(f"  theta_init={theta_init:.2f}, thetadot_init={thetadot_init:.2f}")
        
        debug_data = run_debug_test(
            mass_pole_modifier=mass_pole_modifier,
            length_modifier=length_modifier,
            theta_init=theta_init,
            thetadot_init=thetadot_init,
            test_id=i
        )
        
        # Print summary
        print(f"  Result: {'SUCCESS' if debug_data['success'] else 'FAILED'}")
        print(f"  Reason: {debug_data['termination_reason']}")
        print(f"  Final step: {debug_data['final_step']}")
        print(f"  Max balanced duration: {debug_data['max_balanced_duration']} steps")
        print(f"  Position range: [{debug_data['min_position']:.2f}, {debug_data['max_position']:.2f}]")
        print(f"  Observations out of bounds: {len(debug_data['observations_out_of_bounds'])} times")
        print(f"  Position bounds exceeded: {len(debug_data['position_bounds_exceeded'])} times")
        
        # Save detailed log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aidan_transformer_control/diagnostic_logs/failed_test_{i}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(debug_data, f, indent=2)
        print(f"  Detailed log saved to: {filename}")

if __name__ == "__main__":
    main() 