import numpy as np
import gymnasium as gym
from gymnasium import spaces, wrappers
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional
import warnings
import argparse
import time

class ContinuousCartPoleWrapper(gym.Wrapper):
    """Wrapper to convert discrete CartPole to continuous control
    
    Wraps the standard Gymnasium CartPole-v1 environment to accept continuous actions.
    Action space: Box(-1, 1) mapped to force direction
    """
    
    def __init__(self, env, max_force: float = 30.0, start_hanging: bool = True, mass_pole_modifier: float = 1.0, mass_cart_modifier: float = 1.0, length_modifier: float = 1.0):
        super().__init__(env)
        self.max_force = max_force
        self.start_hanging = start_hanging
        
        # Override action space to accept physical force
        self.action_space = spaces.Box(low=-max_force, high=max_force, shape=(1,), dtype=np.float32)
        
        # Store original force magnitude and override it
        self.original_force_mag = env.unwrapped.force_mag
        env.unwrapped.force_mag = max_force
        
        # Modify physical parameters slightly
        env.unwrapped.masspole *= mass_pole_modifier
        env.unwrapped.masscart *= mass_cart_modifier
        env.unwrapped.length *= length_modifier
        
        # Physical parameters from wrapped environment
        self.gravity = env.unwrapped.gravity
        self.masscart = env.unwrapped.masscart
        self.masspole = env.unwrapped.masspole
        self.total_mass = env.unwrapped.total_mass
        self.length = env.unwrapped.length
        self.polemass_length = env.unwrapped.polemass_length
        self.tau = env.unwrapped.tau
        
        # Increase position threshold for swing-up
        env.unwrapped.x_threshold = 4.0
        
    def step(self, action):
        force = float(action[0])          # already in Newtons
        self.env.unwrapped.force_mag = abs(force)
        discrete_action = 1 if force >= 0 else 0
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # Custom reward for swing-up task
        x, x_dot, theta, theta_dot = obs
        reward = np.cos(theta) + 0.1 * np.cos(2 * theta) - 0.01 * abs(x)
        
        # Don't terminate for angle, only for position bounds
        if terminated and abs(theta) > 0.2:
            terminated = False
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        if self.start_hanging:
            # Set pole to hanging position with small initial velocity to avoid tiny swings
            self.env.unwrapped.state = np.array([
                0.0,  # x
                0.0,  # x_dot
                np.pi + np.random.uniform(-0.1, 0.1),  # theta (hanging down)
                np.random.uniform(-0.5, 0.5)   # theta_dot (small initial angular velocity)
            ])
            obs = self.env.unwrapped.state.copy()
        
        return obs, info


class CartPoleSwingUpController:
    """Hybrid controller for CartPole swing-up and stabilization.
    
    Combines energy-based swing-up control with LQR stabilization.
    """
    
    def __init__(self, env):
        self.env = env
        
        # Extract physical parameters
        self.m_cart = env.masscart
        self.m_pole = env.masspole
        self.l = env.length
        self.g = env.gravity
        
        # Natural frequency: w_0 = sqrt(g/L)
        self.w_0 = np.sqrt(self.g / self.l)
        
        self.k_e = 9.0             # large enough to see motion within a second
        self.k_p = 1.0              # gentle recentring
        self.k_d = 2*np.sqrt(self.k_p)   # critical damping on cart

        
        # LQR parameters
        self.lqr_gain = self._compute_lqr_gain()
        
        # Switching parameters
        self.switch_angle = 0.5  # ~11 degrees from upright (tighter threshold)
        self.switch_velocity = 2.0  # rad/s (lower velocity threshold)
        
        # Logging and tracking
        self.current_mode = "Swing-up"
        self.mode_switches = 0
        self.last_mode = "Swing-up"
        self.lqr_time = 0
        self.swingup_time = 0
        
    def _compute_lqr_gain(self) -> np.ndarray:
        """Compute LQR gain matrix for upright stabilization."""
        # Linearize the nonlinear dynamics around the unstable upright
        # equilibrium (θ ≈ 0). Using small-angle approximations we obtain
        # the continuous-time system matrices ẋ = A x + B u below:
        M = self.m_cart + self.m_pole

        # solution from the book for x_dot = A x + B u
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m_pole * self.g / self.m_cart, 0],
            [0, 0, 0, 1],
            [0, 0,  M  * self.g / (self.m_cart * self.l), 0]
        ])

        B = np.array([
            [0],
            [1 / self.m_cart],
            [0],
            [-1 / (self.m_cart * self.l)]
        ])
        
        # LQR cost matrices
        # Q = np.diag([1.0, 0.1, 100.0, 1.0])  # Heavy penalty on angle, less on position
        # R = np.array([[0.01]])  # Reduced control cost for more aggressive control
        Q = np.diag([1, 1, 1, 1])
        R = np.array([[1]]) 
        
        # Solve the Continuous-time Algebraic Riccati Equation (CARE)
        #   Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0
        # and compute the infinite-horizon LQR gain K = R⁻¹ Bᵀ P.
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K.flatten()
        except Exception as e:
            warnings.warn(f"LQR computation failed: {e}")
            # Fallback gains
            return np.array([10.0, 15.0, 100.0, 20.0])
    
    def _calculate_pole_energy(self, theta: float, theta_dot: float) -> float:
        """Calculate total energy of the pole."""
        # E_pole = E_p = KE + PE = (1/2) * m_p * L^2 * (dθ/dt)^2 + m_p * g * L * (1 - cos(θ))
        kinetic = 0.5 * self.m_pole * (self.l ** 2) * (theta_dot ** 2)
        
        # PE = m_p * g * L * (1 + cos(θ))
        # E_p includes term: m_p * g * L * (1 + cos(θ))
        potential_term = self.m_pole * self.g * self.l * (1 + np.cos(theta))
        
        return kinetic + potential_term
    
    def _swing_up_control(self, state: np.ndarray) -> float:
        """Energy-based swing-up controller."""
        # Extract state components
        # System: x = [x, θ, dx/dt, dθ/dt]ᵀ
        x, x_dot, theta, theta_dot = state

        # theta = 0  ⇒  pole down   (old notes)
        # theta = π  ⇒  pole up
        theta        = (theta + np.pi) % (2*np.pi) - np.pi      # theta ∈ (-π, π]
        theta_dot    = theta_dot

        
        # Calculate pole energy
        # E_pole = E_p = KE + PE = (1/2) * m_p * L^2 * (dθ/dt)^2 + m_p * g * L * (1 - cos(θ))
        E_p = self._calculate_pole_energy(theta, theta_dot)
        
        # Desired energy at upright position
        # E_d = 2 * m_p * g * L
        E_d = 2 * self.m_pole * self.g * self.l
        
        # Energy error
        # Error: Ẽ = E_p - E_d
        E_tilde = (E_p - E_d) / (0.5 * E_d)
        
        # theta_brake    = 0.5  # only brake when pole is already near upright

        # 1. Angular-speed limiter: if the pole is close to upright *and*
        #    |θ̇| is too high, force braking.
        # if abs(theta_dot) > self.switch_velocity and abs(theta) < theta_brake:
        #     sign = -2                      # hard brake
        # else:
        #     # applying a deadband to prevent the system from having high x_dot going into LQR
        #     deadband = 0.5 * E_d              # 50 % quiet zone around target
        #     if   E_tilde < -deadband:  sign = 1   # pump (this does not effect the swing up control)
        #     elif E_tilde >  deadband:  sign = -1   # brake (forces swing up controller to apply negative force to slow down the cart)
        #     else:                    sign =  0   # let it coast (do not apply any pole force if E_tilde is within the deadband)
        # Control law
        # u = k_e * (dθ/dt) * cos(θ) * Ẽ - k_p * x - k_d * (dx/dt)
        # print(f"Calculating swing-up control: sign={sign}, k_e={self.k_e}, theta_dot={theta_dot}, cos(theta)={np.cos(theta)}, E_tilde={E_tilde}, k_p={self.k_p}, x={x}, k_d={self.k_d}, x_dot={x_dot}")
        # sign = 1
        # u = sign * self.k_e * theta_dot * np.cos(theta) * (E_tilde) - self.k_p * x - self.k_d * x_dot
        u = self.k_e * theta_dot * np.cos(theta) * np.tanh(E_tilde) - self.k_p * x - self.k_d * x_dot
        # print(f"u={u}")
        u = np.clip(u, -20.0, 20.0)
        # Saturate control to physical limits
        # real cart force
        # print(f"Calculating force: m_cart={self.m_cart}, m_pole={self.m_pole}, sin(theta)={np.sin(theta)}, theta_dot={theta_dot}, theta_dot^2={theta_dot**2}, cos(theta)={np.cos(theta)}, g={self.g}, l={self.l}")
        force = (self.m_cart + self.m_pole*np.sin(theta)**2)*u - self.m_pole*np.sin(theta)*(self.l*theta_dot**2 + self.g*np.cos(theta))
        # print(f"force={force}")
        # print(f"max_force={self.env.max_force}")
        # print(f"Force applied: {np.clip(force, -self.env.max_force, self.env.max_force)}")
        return np.clip(force, -self.env.max_force, self.env.max_force)

    
    def _lqr_control(self, state: np.ndarray) -> float:
        """LQR controller for upright stabilization."""
        x, x_dot, theta, theta_dot = state
        
        # Wrap angular error to (−π, π] to obtain a smooth error signal.
        theta_error = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        state_error = np.array([x, x_dot, theta_error, theta_dot])
        
        # LQR control law
        return -self.lqr_gain @ state_error
    
    def get_action(self, state: np.ndarray) -> float:
        """Compute control action based on current state."""
        x, x_dot, theta, theta_dot = state
        
        # Hybrid supervisor: choose LQR when close to upright, else swing-up.
        curr_theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        # print(f"curr_theta={curr_theta}, theta_dot={theta_dot}, switch_angle={self.switch_angle}, switch_velocity={self.switch_velocity}")
        if abs(curr_theta) < self.switch_angle:
            # Use LQR for stabilization
            force = self._lqr_control(state)            
            self.current_mode = "LQR"
            self.lqr_time += 1
            # Log LQR control
            # print(f"LQR: θ={np.degrees(curr_theta):5.1f}° F={force:6.1f}N")
        else:
            # Use swing-up controller
            force = self._swing_up_control(state)
            self.current_mode = "Swing-up"
            self.swingup_time += 1
            # Log swing-up energy and control
            E_p = self._calculate_pole_energy(theta, theta_dot)
            E_d = 2 * self.m_pole * self.g * self.l
            E_err = E_p - E_d
            # print(f"SU: θ={np.degrees(theta):5.1f}° E={E_p:5.2f} Err={E_err:6.2f} F={force:6.1f}N")
        
        # Track mode switches
        if self.current_mode != self.last_mode:
            self.mode_switches += 1
            self.last_mode = self.current_mode
        
        # Return the physical force directly
        return np.array([force], dtype=np.float32)
    

def run_complete_demonstration(max_steps: int = 20000, stats: bool = False, mass_pole_modifier: float = 1.0, mass_cart_modifier: float = 1.0, length_modifier: float = 1.0, test: bool = False, no_visual: bool = False, theta_init: Optional[float] = None, thetadot_init: Optional[float] = None):
    """Run complete CartPole swing-up demonstration with real-time visualization, 
    recording, and analysis plots."""
    if stats:
        from main_log import run_complete_demonstration_logging
        run_complete_demonstration_logging(max_steps=max_steps, stats=stats, mass_pole_modifier=mass_pole_modifier, mass_cart_modifier=mass_cart_modifier, length_modifier=length_modifier)
        return
    
    if not no_visual:
        print("CartPole Swing-Up Control - Complete Demonstration")
        print("=" * 60)
        print("This will show real-time simulation while recording data for analysis")
        print("=" * 60)
    
    # Create environment and controller
    render_mode = None if no_visual else 'human'
    base_env = gym.make('CartPole-v1', render_mode=render_mode)
    # Don't start hanging when using custom initial conditions
    start_hanging = not (theta_init is not None or thetadot_init is not None)
    env = ContinuousCartPoleWrapper(base_env, max_force=500.0, start_hanging=start_hanging, mass_pole_modifier=mass_pole_modifier, mass_cart_modifier=mass_cart_modifier, length_modifier=length_modifier)
    controller = CartPoleSwingUpController(env)
    
    # Data collection
    states = []
    actions = []
    rewards = []
    modes = []
    
    # Reset and start
    state, _ = env.reset()
    
    # Set custom initial conditions if provided
    if theta_init is not None or thetadot_init is not None:
        # Use provided values or defaults
        initial_theta = theta_init if theta_init is not None else np.pi + np.random.uniform(-0.1, 0.1)
        initial_thetadot = thetadot_init if thetadot_init is not None else np.random.uniform(-0.5, 0.5)
        
        env.env.unwrapped.state = np.array([
            0.0,  # x
            0.0,  # x_dot
            initial_theta,  # theta
            initial_thetadot  # theta_dot
        ])
        state = env.env.unwrapped.state.copy()
    
    if not no_visual:
        print(f"\nStarting simulation (max {max_steps} steps)...")
        print("Watch the real-time visualization window!")
    
    # Logging variables
    last_log_time = 0
    log_interval = 500  # Log every 500 steps
    upright_start_time = None
    balanced_duration = 0
    max_balanced_duration = 0
    success_count = 0
    failed_count = 0
    
    for step in range(max_steps):

        # Render environment
        if not no_visual:
            env.render()
        
        # Get control action
        action = controller.get_action(state)
        
        # Store data
        states.append(state.copy())
        actions.append(action[0])
        modes.append(controller.current_mode)
        
        # Step environment
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        
        # Track balancing performance
        theta_upright = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
        is_balanced = abs(theta_upright) < 0.1  # Within ~6 degrees
        
        if is_balanced:
            if upright_start_time is None:
                upright_start_time = step
            balanced_duration = step - upright_start_time
            max_balanced_duration = max(max_balanced_duration, balanced_duration)
            
            # Exit if balanced for 1 seconds (assuming 50 Hz, 1s = 50 steps)
            if balanced_duration >= 50:
                if not no_visual:
                    print(f"\n🎉 SUCCESS! Pole balanced for 5 seconds ({balanced_duration} steps)")
                    print(f"Simulation completed successfully at step {step}")
                return True
        else:
            if upright_start_time is not None:
                upright_start_time = None
                balanced_duration = 0
        
        # Periodic logging
        if not no_visual and step - last_log_time >= log_interval:
            angle_deg = np.degrees(theta_upright)
            cart_pos = state[0]
            mode = controller.current_mode
            print(f"Step {step:4d}: Mode={mode:8s} | Angle={angle_deg:6.1f}° | Cart={cart_pos:5.2f}m | Balanced={balanced_duration:3d} steps")
            last_log_time = step
        
        if terminated or step >= max_steps:
            if not no_visual:
                print(f"\nSimulation terminated at step {step}")
            break
    
    env.close()
    return False

# def main():
#     parser = argparse.ArgumentParser(description='CartPole Swing-Up Control')
#     parser.add_argument('--max_steps', type=int, default=20000, help='Maximum simulation steps')
#     parser.add_argument('--stats', action='store_true', help='Show statistics')
#     parser.add_argument('--test', action='store_true', help='Run robust parameter test')
#     parser.add_argument('--no_visual', action='store_true', help='Do not show visualization')
#     parser.add_argument('--start_hanging', action='store_true', help='Use hanging initial conditions for tests')
#     args = parser.parse_args()

#     if args.test:
#         failed_params = []  
#         num_tests = 100  # Number of random parameter combinations to test
        
#         for i in range(num_tests):
#             # Randomly sample parameters (similar to aidan_gym_test.py)
#             mass_pole_modifier = np.random.uniform(0.5, 1.0) / 0.1  # Convert to modifier (default masspole is 0.1)
#             length_modifier = np.random.uniform(1.0, 1.5) / 0.5     # Convert to modifier (default length is 0.5)
            
#             # Initial conditions based on flag
#             if args.start_hanging:
#                 # Use default hanging initial conditions (no custom theta/thetadot)
#                 theta_init = None
#                 thetadot_init = None
#                 print(f"Test {i+1}/{num_tests}: mass_pole_modifier={mass_pole_modifier:.2f}, length_modifier={length_modifier:.2f}, hanging_start=True")
#             else:
#                 # Random initial conditions (similar to aidan_gym_test.py)
#                 theta_init = np.random.uniform(np.pi - np.pi/2, np.pi + np.pi/2)
#                 thetadot_init = np.random.uniform(-1.0, 1.0)
#                 print(f"Test {i+1}/{num_tests}: mass_pole_modifier={mass_pole_modifier:.2f}, length_modifier={length_modifier:.2f}, theta_init={theta_init:.2f}, thetadot_init={thetadot_init:.2f}")
            
#             success = run_complete_demonstration(
#                 max_steps=args.max_steps, 
#                 stats=args.stats, 
#                 mass_pole_modifier=mass_pole_modifier, 
#                 mass_cart_modifier=1.0, 
#                 length_modifier=length_modifier, 
#                 no_visual=args.no_visual,
#                 theta_init=theta_init,
#                 thetadot_init=thetadot_init
#             )
            
#             if not success:
#                 if args.start_hanging:
#                     failed_params.append((mass_pole_modifier, length_modifier, "hanging"))
#                 else:
#                     failed_params.append((mass_pole_modifier, length_modifier, theta_init, thetadot_init))
        
#         print(f"\nTest Results:")
#         print(f"Successful: {num_tests - len(failed_params)}/{num_tests}")
#         print(f"Failed parameters: {failed_params}")
#     else:
#         run_complete_demonstration(max_steps=args.max_steps, stats=args.stats, no_visual=args.no_visual)

# main()