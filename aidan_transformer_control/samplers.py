import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import math
import torch
import numpy as np
# from workCon import checking 
# from workCon_ebonye import checking 
from workCon_linearsys import checking
from workCon_cartpole import checking as checking_cartpole
import crocoddyl
# from cartpole_optimal_trajectory import get_trajectory
from scipy.signal import place_poles


class DataSampler:
    # def __init__(self, n_dims, init_conditions):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        # self.X0 = init_conditions
        

    def sample_xs(self):
        raise NotImplementedError

def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError

def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class PendulumSampler(DataSampler):
    # def __init__(self, n_dims, init_conditions):
    #     super().__init__(n_dims, init_conditions)

    def __init__(self, n_dims):
        super().__init__(n_dims)    
        # self.theta_limit = np.pi  
        # self.thetadot_limit = 10.0  
        self.theta_limit = np.pi/4
        self.thetadot_limit = 6.0 ###### 2/8/2025 (ebonye): changing bounds to make it easier for the model to learn

    def sample_val_initial_conditions(self):
        """
        Samples initial conditions for validation set, ensuring initial theta and thetadot don't overlap with training set

        Returns:
            list: A list containing:
                - theta_init (float): The initial angular position (theta) in radians.
                - thetadot_init (float): The initial angular velocity (thetadot).
        """
        epsilon = 1e-6  
        theta_ranges = [(-3 * np.pi / 2, -np.pi - epsilon), (np.pi + epsilon, 3 * np.pi / 2)]
        theta_choice = np.random.choice([0, 1])
        theta_init = np.random.uniform(*theta_ranges[theta_choice])
        thetadot_ranges = [(-20.0, -11.0), (11.0, 20.0)]
        thetadot_choice = np.random.choice([0, 1])
        thetadot_init = np.random.uniform(*thetadot_ranges[thetadot_choice])
        return [theta_init, thetadot_init]

    # def sample_initial_conditions(self):
    def sample_initial_conditions(self, size=1):
        """
        Samples general initial conditions for the pendulum system within the limits.

        Returns:
            list: A list containing:
                - theta_init (float): The initial angular position (theta) in radians, sampled uniformly from [-pi, pi].
                - thetadot_init (float): The initial angular velocity (thetadot), sampled uniformly from [-10, 10].
        """
        # theta_init = np.random.uniform(-self.theta_limit, self.theta_limit)
        # theta_init = np.random.uniform(np.pi/5, np.pi/2) ###### 2/8/2025 (ebonye): changing bounds to make it easier for the model to learn
        # theta_init = np.random.uniform(np.pi/5, 7*np.pi/6) ###### 3/5/2025 (ebonye): making more difficult
        # theta_init = np.random.uniform(np.pi-(np.pi/2), np.pi+(np.pi/2)) ###### 3/9/2025 (ebonye): upswing initial conditions
        # thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)
        # theta_init = np.random.uniform(np.pi - (3*np.pi/4), np.pi + (3*np.pi/4)) ###### 3/9/2025 (ebonye): upswing initial conditions
        # thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)

        theta_init = np.random.uniform(np.pi - (3*np.pi/4), np.pi + (3*np.pi/4), size=size)
        thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit, size=size)

        init_conditions = np.column_stack((theta_init, thetadot_init))

        # return [theta_init, thetadot_init]
        return init_conditions

    def calculate_lyapunov_derivative(self, theta, thetadot, u, m=1, l=1, b=0.5, g=9.81):
        """
        Calculates the time derivative of the Lyapunov function for a pendulum system, 
        given its current state and control input. The Lyapunov derivative provides insights 
        into the stability of the system and whether the control law drives the system toward 
        the desired state.

        Args:
            theta (float or torch.Tensor): The angular position (in radians) of the pendulum.
            thetadot (float or torch.Tensor): The angular velocity of the pendulum.
            u (float or torch.Tensor): The control input applied to the system.
            m (float, optional): The mass of the pendulum. Defaults to 1.
            l (float, optional): The length of the pendulum. Defaults to 1.
            b (float, optional): The damping coefficient. Defaults to 0.5.
            g (float, optional): The acceleration due to gravity. Defaults to 9.81.

        Returns:
            torch.Tensor: The time derivative of the Lyapunov function, indicating the rate of change of the system's energy under the given state and control input. (All values should be negative)
        """
        theta = torch.tensor(theta, dtype=torch.float32) if not isinstance(theta, torch.Tensor) else theta
        thetadot = torch.tensor(thetadot, dtype=torch.float32) if not isinstance(thetadot, torch.Tensor) else thetadot
        u = torch.tensor(u, dtype=torch.float32) if not isinstance(u, torch.Tensor) else u
        dV_dtheta = m * g * l * torch.sin(theta)
        dV_dthetadot = m * l**2 * thetadot
        ddot_theta = (-b * thetadot + m * g * l * torch.sin(theta) + u) / (m * l**2)

        dV_dt = dV_dtheta * thetadot + dV_dthetadot * ddot_theta
        return dV_dt
    
    # def generate_xs_dataset(self, n_points, b_size, val = "no", mass=1,length=1):
    # def generate_xs_dataset(self, n_points, mass=1, length=1):
    def generate_xs_dataset(self, n_points, b_size, mass=1, length=1):
        """
        Generates datasets for training or evaluation by simulating the states and control values of pendulum system.

        Args:
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            # b_size (int): The batch size
            val (str, optional): A flag to indicate whether this is a validation dataset or not. Defaults to "no".
            mass (float, optional): The mass of the system being simulated. Defaults to 1.
            length (float, optional): The length of the pendulum or system being simulated. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): A tensor of time steps for each trajectory in the batch.
                - xs_datasets (torch.Tensor): A tensor of state datasets (theta and thetadot) for each trajectory.
                - control_values_batches (torch.Tensor): A tensor of control values for each trajectory.
                - k_values (np.ndarray): The gain matrix of the pendulum simulation.
        """
        # T_batches = []
        # xs_datasets = []
        # control_values_batches = []
        n_stop = n_points
        n_points = n_points/100
        # if self.X0 is None:
        #     self.X0 = self.sample_initial_conditions() #2/8/2025 (ebonye): each batch has the same initial conditions


        # for _ in range(b_size):     
        # ebonye 2/28/2025: removing loop to mix mass/length in a batch right before training       
        # X0 = self.sample_initial_conditions() #get starting initial values of theta and thetadot
        X0 = self.sample_initial_conditions(size=b_size)
        T, theta, thetadot, control_values = checking(X0, n_points, masses=mass, lengths=length, method='rk4', dt=0.01)
        # T, theta, thetadot, control_values, k_values = checking(X0, n_points, method='rk4', dt=0.01,mass = mass,length = length) # where simulation happens
        # T, theta, thetadot, control_values = checking(X0, n_points, method='rk4', dt=0.01,mass = mass,length = length) # where simulation happens
        # T, theta, thetadot, control_values, k_values = checking(self.X0, n_points, method='rk4', dt=0.01,mass = mass,length = length)
        # T = T[:n_stop]
        # theta = theta[:n_stop]
        # thetadot = thetadot[:n_stop]
        # control_values = control_values[:n_stop]

        # import pdb; pdb.set_trace()
        # T = T[:, :n_stop]
        T = T[:n_stop]
        theta = theta[:, :n_stop]
        thetadot = thetadot[:, :n_stop]
        control_values = control_values[:, :n_stop]
        
        # xs_dataset = np.column_stack((theta, thetadot))
        xs_dataset = np.stack((theta, thetadot), axis=2)


        # T_batches.append(T) 
        # xs_datasets.append(xs_dataset)
        # control_values_batches.append(control_values)

        T_batches = T
        xs_datasets = xs_dataset
        control_values_batches = control_values


        # T_batches = np.array(T_batches)
        # xs_datasets = np.array(xs_datasets)
        # control_values_batches = np.array(control_values_batches)
        T_batches = torch.tensor(T_batches).float()
        xs_datasets = torch.tensor(xs_datasets).float()
        control_values_batches = torch.tensor(control_values_batches).float()

        if torch.cuda.is_available():
            T_batches = T_batches.cuda()
            xs_datasets = xs_datasets.cuda()
            control_values_batches = control_values_batches.cuda()

        return T_batches, xs_datasets, control_values_batches #, k_values

class CartPoleSampler2(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)
        self.theta_limit = np.pi / 2
        self.thetadot_limit = 1.0

    def sample_initial_conditions(self, size=1):
        """
        Samples initial conditions for the cart-pole system.

        Returns:
            list: A list containing:
                - x_init (float): The initial position of the cart.
                - theta_init (float): The initial angle of the pole in radians.
                - xdot_init (float): The initial velocity of the cart.
                - thetadot_init (float): The initial angular velocity of the pole.
        """
        x_init = np.zeros(size)  # Assuming the cart starts at the origin
        theta_init = np.random.uniform(-self.theta_limit, self.theta_limit, size=size)  # Sampling theta uniformly within the limit
        xdot_init = np.zeros(size)  # Assuming the cart starts at rest
        thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit, size=size)  # Sampling thetadot uniformly within the limit
        # return [x_init, theta_init, xdot_init, thetadot_init]
        return np.column_stack((x_init, theta_init, xdot_init, thetadot_init))
    
    def generate_xs_dataset(self, n_points, bsize, cartmass=torch.tensor([2]), polemass=torch.tensor([0.2]), polelength=torch.tensor([0.5]), device='cuda:0'):
        """
        Generates a dataset of trajectories for the cart-pole system.

        Args:
            n_points (int): The number of time steps in the trajectory.
            bsize (int): The batch size.
            cartmass (float): Mass of the cart.
            polemass (float): Mass of the pole.
            polelength (float): Length of the pole.

        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): Time steps for the trajectory.
                - xs_datasets (torch.Tensor): State trajectory [x, theta, xdot, thetadot].
                - control_values_batches (torch.Tensor): Control inputs for each time step.
        """
        X0 = self.sample_initial_conditions(size = bsize)
        X0 = torch.tensor(X0, dtype=torch.float32, device=device)  
        total_time = int(n_points * 0.025)  # Assuming dt = 0.01 seconds
        cartmass = torch.tensor(cartmass, dtype=torch.float32).to(X0.device)
        polemass = torch.tensor(polemass, dtype=torch.float32).to(X0.device)
        polelength = torch.tensor(polelength, dtype=torch.float32).to(X0.device)
        # print(f"cartmass device: {cartmass.device}, polemass device: {polemass.device}, polelength device: {polelength.device}")
        T, xs_dataset, control_values, control_modes = checking_cartpole(X0, total_time, m_c=cartmass, m_1=polemass, l_1=polelength, method='rk4', dt=0.025)
        control_values_modes = torch.stack((control_values, control_modes), dim=-1)
        # import pdb; pdb.set_trace()

        # T = np.array(T)
        # states = np.array(states)
        # control_values = np.array(control_values)

        # T = torch.tensor(T)
        # xs_dataset = torch.tensor(states)
        # control_values = torch.tensor(control_values)

        return T, xs_dataset, control_values_modes
        

class CartPoleSampler(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)
        # self.x_limit = 2.4
        self.theta_limit = np.pi / 2
        # self.xdot_limit = 10.0
        self.thetadot_limit = 2.0

    def sample_initial_conditions(self):
        # x_init = np.random.uniform(-self.x_limit, self.x_limit)
        # theta_init = np.random.uniform(-self.theta_limit, self.theta_limit)
        # xdot_init = np.random.uniform(-self.xdot_limit, self.xdot_limit)
        # thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)
        x_init = 0
        theta_init = np.random.uniform(np.pi - (np.pi/2), np.pi + (np.pi/2)) ###### 4/9/2025
        xdot_init = 0
        thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)
        return [x_init, theta_init, xdot_init, thetadot_init]
    
    def get_optimal_trajectory(X0, n_points, cartmass=2, polemass=0.2, polelength=0.5, dt=0.025):
        """
        Generates an optimal trajectory for the cart-pole system using the Crocoddyl library.

        Args:
            X0 (list): Initial state of the system [x, theta, xdot, thetadot].
            n_points (int): Number of time steps in the trajectory.
            cartmass (float): Mass of the cart.
            polemass (float): Mass of the pole.
            polelength (float): Length of the pole.
            dt (float): Time step for simulation.

        Returns:
            tuple: A tuple containing:
                - T (np.ndarray): Time steps for the trajectory.
                - x (np.ndarray): State trajectory [x, theta, xdot, thetadot].
                - u (np.ndarray): Control inputs for each time step.
        """
        # Define the cart-pole system
        
        
    
    def generate_xs_dataset(self, n_points, cartmass=2, polemass=0.2, polelength=0.5):
        T_batches = []
        xs_datasets = []
        control_values_batches = []
        n_stop = n_points
        n_points = n_points/100

        # if self.X0 is None:
        #     self.X0 = self.sample_initial_conditions() #2/8/2025 (ebonye): each batch has the same initial conditions

        # for _ in range(b_size):     
        # ebonye 2/28/2025: removing loop to mix mass/length in a batch right before training       
        X0 = self.sample_initial_conditions()
        T, x, theta, xdot, thetadot, control_values = get_trajectory(X0, n_points, cartmass, polemass, polelength, dt=0.025)

        T = T[:n_stop]
        x = x[:n_stop]
        theta = theta[:n_stop]
        xdot = xdot[:n_stop]    
        thetadot = thetadot[:n_stop]
        control_values = control_values[:n_stop]

        xs_dataset = np.column_stack((x, theta, xdot, thetadot))
        T_batches.append(T)
        xs_datasets.append(xs_dataset)
        control_values_batches.append(control_values)

        T_batches = np.array(T_batches)
        xs_datasets = np.array(xs_datasets)
        control_values_batches = np.array(control_values_batches)
        T_batches = torch.tensor(T_batches).float()
        xs_datasets = torch.tensor(xs_datasets).float()
        control_values_batches = torch.tensor(control_values_batches).float()
        if torch.cuda.is_available():
            T_batches = T_batches.cuda()
            xs_datasets = xs_datasets.cuda()
            control_values_batches = control_values_batches.cuda()
        return T_batches, xs_datasets, control_values_batches

class LinearSystemSampler(DataSampler):
    # def __init__(self, n_dims, init_conditions):
    #     super().__init__(n_dims, init_conditions)

    def __init__(self, n_dims):
        super().__init__(n_dims)    
        self.x1_limit = 1.0
        self.x2_limit = 1.0

    def sample_initial_conditions(self, size=1):
        """
        Samples general initial conditions for the pendulum system within the limits.

        Returns:
            list: A list containing:
                - x1_init (float): The initial position (x1) in radians, sampled uniformly from [-1, 1].
                - x2_init (float): The initial velocity (x2), sampled uniformly from [-1, 1].
        """
        x1_init = np.random.uniform(-self.x1_limit, self.x1_limit, size=size)
        x2_init = np.random.uniform(-self.x2_limit, self.x2_limit, size=size)

        init_conditions = np.column_stack((x1_init, x2_init))
        return init_conditions

    def generate_xs_dataset(self, n_points, b_size, lambda1, lambda2):
        """
        Generates datasets for training or evaluation by simulating the states and control values of linear system.

        Args:
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            # b_size (int): The batch size
            lambda1 (numpy array): The first eigenvalue of the system.
            lambda2 (numpy array): The second eigenvalue of the system.

        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): A tensor of time steps for each trajectory in the batch.
                - xs_datasets (torch.Tensor): A tensor of state datasets (x1 and x2) for each trajectory.
                - control_values_batches (torch.Tensor): A tensor of control values for each trajectory.
        """
        
        n_stop = n_points
        # n_points = n_points/100
        n_points = n_points*0.05
        # n_points = n_points*0.01
        
        X0 = self.sample_initial_conditions(size=b_size)
        # T, x1, x2, control_values = checking(X0, n_points, method='rk4', dt=0.05)
        T, x1, x2, control_values = checking(X0, n_points, lambda1=lambda1, lambda2=lambda2, method='rk4', dt=0.05)
        # T, x1, x2, control_values = checking(X0, n_points, lambda1=lambda1, lambda2=lambda2, method='rk4', dt=0.01)
        
        T = T[:n_stop]
        x1 = x1[:, :n_stop]
        x2 = x2[:, :n_stop]
        control_values = control_values[:, :n_stop]
        
        xs_dataset = np.stack((x1, x2), axis=2)

        T_batches = T
        xs_datasets = xs_dataset
        control_values_batches = control_values

        T_batches = torch.tensor(T_batches).float()
        xs_datasets = torch.tensor(xs_datasets).float()
        control_values_batches = torch.tensor(control_values_batches).float()

        if torch.cuda.is_available():
            T_batches = T_batches.cuda()
            xs_datasets = xs_datasets.cuda()
            control_values_batches = control_values_batches.cuda()

        return T_batches, xs_datasets, control_values_batches
    
    def rk4_step_batch(self, x, u, l1, l2, dt = 0.05):
        """
        RK4 integrator for a batch of linear systems.
        Args:
            x (torch.Tensor): Current state of the system, shape (b_size, n_dims).
            u (torch.Tensor): Control input, shape (b_size, 1).
            l1 (torch.Tensor): First eigenvalue of the system, shape (b_size,).
            l2 (torch.Tensor): Second eigenvalue of the system, shape (b_size,).
            dt (float, optional): Time step for the simulation. Defaults to 0.05.
        Returns:
            torch.Tensor: Next state of the system after applying the RK4 step, shape (b_size, n_dims).
        """
        batch_size = x.shape[0]

        # Build batched A: shape (b_size, n_dims, n_dims)
        A = torch.zeros((batch_size, 2, 2), device=x.device)
        A[:, 0, 0] = l1
        A[:, 1, 1] = l2

        # B is fixed: shape (1, n_dims)
        B = torch.tensor([1.0, 1.0], dtype=x.dtype, device=x.device) #.unsqueeze(0)

        def f(xi, ui):
            """
            Dynamics function for the linear system.
            Args:
                xi (torch.Tensor): Current state, shape (b_size, n_dims).
                ui (torch.Tensor): Control input, shape (b_size, 1).
            Returns:
                torch.Tensor: State derivative, shape (b_size, n_dims).
            """
            # import pdb; pdb.set_trace()
            # print(f"xi shape: {xi.shape}, ui shape: {ui.shape}")
            Ax = torch.bmm(A, xi.unsqueeze(-1)).squeeze(-1)  # Batched matrix multiplication
            Bu = B * ui #.unsqueeze(-1)  # Broadcasting control input
            # print(f"Ax shape: {Ax.shape}")
            # print(f"Bu shape: {Bu.shape}")
            c = Ax + Bu  # Adding the control input
            # print(f" Ax+Bu shape: {c.shape}")
            # print(xi.shape)
            return Ax + Bu
        
        # Perform RK4 integration
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def run_inference_on_model(self, model, X0, n_points, b_size, lambda1, lambda2, device = "cuda:0", dt=0.05):
        """
        Runs inference on a trained model to generate control values for the linear system.

        Args:
            model (torch.nn.Module): A trained model to generate control values.
            X0 (numpy array): Initial state of the system, shape (b_size, n_dims).
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            b_size (int): The batch size
            lambda1 (numpy array): The first eigenvalue of the system.
            lambda2 (numpy array): The second eigenvalue of the system.
        dt (float, optional): The time step for the simulation. Defaults to 0.05.

        Returns:
            torch.Tensor: A tensor of control values generated by the model.
            torch.Tensor: A tensor of time steps for each trajectory.
            torch.Tensor: A tensor of state datasets (x1 and x2) for each trajectory.
        """
        n_points_total = int(n_points/0.05)
        model = model.to(device)
        XData = torch.tensor(X0, dtype=torch.float32, device=device)
        XData = XData.unsqueeze(1)  # Add a time dimension, shape (b_size, 1, n_dims)
        YS = torch.zeros((b_size), device=device)
        YS = YS.unsqueeze(1)  # Initialize YS with the first control input, shape (b_size, 1)
        T = torch.linspace(0, n_points_total * dt, n_points_total, device=device).unsqueeze(0).repeat(b_size, 1)

        counter = 0

        for i in range(n_points_total+1):
            with torch.no_grad():
                # print(f"i: {i}, XData shape: {XData.shape}, YS shape: {YS.shape}, T shape: {T.shape}")
                # import pdb; pdb.set_trace()
                # print(f"i: {i}")
                u_pred, state_pred = model(XData, YS, inf = "no")
                u = u_pred[:, -1]

            # u = u.cpu().numpy()
            previous_state = XData[:, -1, :] #.cpu().numpy()
            next_state = self.rk4_step_batch(previous_state, u, lambda1, lambda2, dt=dt) # shape (b_size, n_dims)

            if i == n_points_total:
                counter = 2
            

            if counter == 0:
                XData = torch.cat((XData, next_state.unsqueeze(1)), dim=1)  # Append the new state to the sequence
                YS = u #.unsqueeze(1)  # Initialize YS with the first control input
                counter = 1
            elif counter == 1:
                XData = torch.cat((XData, next_state.unsqueeze(1)), dim=1)  
                YS = torch.cat((YS, u), dim=1)  # Append the new control input to YS 
            else:
                YS = torch.cat((YS, u), dim=1)  # Append the new control input to YS

            

            # import pdb; pdb.set_trace()
        # Convert the final XData and YS to the desired output format
        T_batches = T
        xs_datasets = XData
        control_values_batches = YS

        return control_values_batches, T_batches, xs_datasets
    
    def get_expert_control_dagger(self, xs, lambda1, lambda2):
        """
        Generates expert control values given the visited states of the dagger dataset.
        Args:
            xs (torch.Tensor): A tensor of visited states (x1 and x2) for each trajectory. shape (b_size, n_points, n_dims).
            lambda1 (torch.Tensor): The first eigenvalue of the system. shape (b_size,).
            lambda2 (torch.Tensor): The second eigenvalue of the system. shape (b_size,).
        Returns:
            torch.Tensor: A tensor of expert control values for each trajectory.
        """
        xs = xs.to(lambda1.device)
        b_size = xs.shape[0]
        
        desired_poles = np.array([-1, -1.1], dtype=np.float32)  # Desired closed-loop poles
       
        # compute gain matrix K for each system in batch
        lambda1 = lambda1.cpu().numpy()
        lambda2 = lambda2.cpu().numpy()
        K = np.zeros((b_size, 1, 2), dtype=np.float32)
        for i in range(b_size):
            A = np.array([[lambda1[i], 0], [0, lambda2[i]]], dtype=np.float32)
            B = np.array([[1.0], [1.0]], dtype=np.float32)
            desired_poles = np.array([-1, -1.1], dtype=np.float32)  # Desired closed-loop poles
            place = place_poles(A, B, desired_poles)
            K[i] = place.gain_matrix
        K = torch.tensor(K, dtype=torch.float32, device=xs.device)  # Convert to tensor and move to the correct device
        # compute control values for each state in the batch
        control_values = -torch.bmm(xs, K.permute(0, 2, 1)).squeeze(-1) # shape (b_size, n_points)

        return control_values  
      
    def generate_windowed_mixed_policy_dagger_dataset(self, n_points, b_size, lambda1, lambda2, model, expert_window = 100, dt=0.05, device = "cuda:0"):
        """
        Generates a mixed policy dagger dataset for training by simulating the states and control values of linear system.
        Args:
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            b_size (int): The batch size
            lambda1 (numpy array): The first eigenvalue of the system.
            lambda2 (numpy array): The second eigenvalue of the system.
            model (torch.nn.Module, optional): A trained model to generate control values. 
            expert_window (int, optional): The number of expert control values to use in the mixed policy. Defaults to 50.
            dt (float, optional): The time step for the simulation. Defaults to 0.05.
        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): A tensor of time steps for each trajectory in the batch.
                - xs_datasets (torch.Tensor): A tensor of state datasets (x1 and x2) for each trajectory.
                - control_values_batches (torch.Tensor): A tensor of control values for each trajectory.
        """  
        # n_points_total = int(n_points/dt)
        n_points_total = int(n_points)
        activation_threshold = 50
        recovery_threshold= 40 

        model = model.to(device)
        lambda1 = torch.tensor(lambda1, dtype=torch.float32, device=device)
        lambda2 = torch.tensor(lambda2, dtype=torch.float32, device=device)

        # Inital conditions
        X0 = self.sample_initial_conditions(size=b_size)
        X0 = torch.tensor(X0, dtype=torch.float32, device=device)

        XData = X0.unsqueeze(1)  # Add a time dimension, shape (b_size, 1, n_dims)
        YS = torch.zeros((b_size, 1), device=device)

        all_states = [XData.clone()]  # Store all states for later use
        all_controls = []

        # expert_start = torch.randint(low=0, high=n_points_total - expert_window+1, size=(b_size,), device=device)  # Random start for expert control values
        expert_start = torch.ones(b_size, dtype=torch.int64, device=device) * 50
        expert_end = expert_start + expert_window

        # Precompute expert gain matrices
        desired_poles = np.array([-1, -1.1], dtype=np.float32)
        K = torch.zeros((b_size, 1, 2), dtype=torch.float32, device=device)
        for i in range(b_size):
            A = torch.tensor([[lambda1[i], 0], [0, lambda2[i]]], dtype=torch.float32, device=device)
            B = torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device)
            place = place_poles(A.cpu().numpy(), B.cpu().numpy(), desired_poles)
            K[i] = torch.tensor(place.gain_matrix, dtype=torch.float32, device=device) #.unsqueeze(0)

        # State magnitude-based intervention
        expert_active = torch.zeros(b_size, dtype=torch.bool, device=device)
        state_norms = torch.zeros(b_size, device=device)
        
        # Start rollout
        for t in range(n_points_total):
            current_state = XData[:, -1, :]  # Get the last state in the sequence
            state_norms = torch.norm(current_state, dim=1) # L2 norm per trajectory

            # # Determine which trajectories use expert at this time step
            use_expert = (t >= expert_start) & (t < expert_end)

            # Update expert activation
            # activate_expert = (~expert_active) & (state_norms > activation_threshold)
            # deactivate_expert = expert_active & (state_norms < recovery_threshold)

            # expert_active[activate_expert] = True
            # expert_active[deactivate_expert] = False

            # use_expert = expert_active.clone()

            u = torch.zeros((b_size,), dtype=torch.float32, device=device)  # Initialize control input

            if use_expert.any():
                # Use expert control values for the specified trajectories
                x_expert = current_state[use_expert]
                K_expert = K[use_expert]
                u[use_expert] = -torch.bmm(x_expert.unsqueeze(1), K_expert.permute(0, 2, 1)).squeeze(-1).squeeze(-1)  # Expert control values

            if (~use_expert).any():
                # Use model predictions for the remaining trajectories
                X_input = XData[~use_expert]
                YS_input = YS[~use_expert]
                with torch.no_grad():
                    u_pred, _ = model(X_input, YS_input, inf="no")
                    u_model = u_pred[:, -1]  # Get the last control input prediction

                # import pdb; pdb.set_trace()
                u[~use_expert] = u_model.squeeze()  # Assign model control values

            # Always use expert control for labels
            u_expert_label = -torch.bmm(current_state.unsqueeze(1), K.permute(0, 2, 1)).squeeze(-1)  # Expert control values for labels

            # Apply RK4 step to get the next state
            next_state = self.rk4_step_batch(current_state, u.unsqueeze(1), lambda1, lambda2, dt=dt)

            # Append the new state and control input to the sequences
            XData = torch.cat((XData, next_state.unsqueeze(1)), dim=1)  # Append the new state to the sequence
            YS = torch.cat((YS, u.unsqueeze(1)), dim=1)

            all_states.append(next_state.unsqueeze(1))  # Store the new state
            # all_controls.append(u.unsqueeze(1))  # Store the control input
            all_controls.append(u_expert_label.unsqueeze(1))  # Store the expert control input for labels

        # Stack final results
        xs_datasets = torch.cat(all_states, dim=1)  # Shape (b_size, n_points_total, n_dims)
        xs_datasets = xs_datasets[:, :n_points_total, :]  # Ensure we only keep the first n_points_total states
        control_values_batches = torch.cat(all_controls, dim=1)  # Shape (b_size, n_points_total)
        control_values_batches = control_values_batches.squeeze(-1)

        T_batches = torch.linspace(0, n_points_total * dt, n_points_total, device=device).unsqueeze(0).repeat(b_size, 1)

        return T_batches, xs_datasets, control_values_batches, lambda1, lambda2
        
    
    
    
    def generate_dagger_dataset(self, n_points, b_size, lambda1, lambda2, model):
        """
        Generates dagger dataset for training by simulating the states and control values of linear system.
        Args:
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            b_size (int): The batch size
            lambda1 (numpy array): The first eigenvalue of the system.
            lambda2 (numpy array): The second eigenvalue of the system.
            model (torch.nn.Module, optional): A trained model to generate control values. 
        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): A tensor of time steps for each trajectory in the batch.
                - xs_datasets (torch.Tensor): A tensor of state datasets (x1 and x2) for each trajectory.
                - control_values_batches (torch.Tensor): A tensor of control values for each trajectory.
        """  
        
        
        # n_stop = n_points
        # n_points = n_points/100
        n_points = n_points*0.05
        # n_points = n_points*0.01

        
        
        X0 = self.sample_initial_conditions(size=b_size)
        prob = 0.5
        mask = np.random.rand(b_size) < prob

        X0_expert = X0[mask]
        lambda1_expert = lambda1[mask]
        lambda2_expert = lambda2[mask]

        X0_dagger = X0[~mask]
        lambda1_dagger = lambda1[~mask]
        lambda1_dagger = torch.tensor(lambda1_dagger).float()
        lambda2_dagger = lambda2[~mask]
        lambda2_dagger = torch.tensor(lambda2_dagger).float()

        T_expert, x1_expert, x2_expert, control_values_expert = checking(X0_expert, n_points, lambda1=lambda1_expert, lambda2=lambda2_expert, method='rk4', dt=0.05)
        xs_expert = np.stack((x1_expert, x2_expert), axis=2)
        xs_expert = torch.tensor(xs_expert).float()
        control_values_expert = torch.tensor(control_values_expert).float()
        control_values_dagger, T_dagger, xs_dagger = self.run_inference_on_model(model, X0_dagger, n_points, b_size - np.sum(mask), lambda1=lambda1_dagger, lambda2=lambda2_dagger)
        control_values_dagger_final = self.get_expert_control_dagger(xs_dagger, lambda1_dagger, lambda2_dagger)

        lambda1_expert = torch.tensor(lambda1_expert).float()
        lambda2_expert = torch.tensor(lambda2_expert).float()


        T_batches = T_expert
        xs_expert = xs_expert.to(xs_dagger.device)  # Ensure both tensors are on the same device
        control_values_expert = control_values_expert.to(control_values_dagger_final.device)  # Ensure both tensors are on the same device
        lambda1_expert = lambda1_expert.to(lambda1_dagger.device)
        lambda2_expert = lambda2_expert.to(lambda2_dagger.device)
        T_dagger = T_dagger.to(xs_dagger.device)
        xs_datasets = torch.cat((xs_expert, xs_dagger), dim=0)
        # control_values = torch.cat((control_values_expert, control_values_dagger), dim=0)
        control_values = torch.cat((control_values_expert, control_values_dagger_final), dim=0)
        lambda1 = torch.cat((lambda1_expert, lambda1_dagger), dim=0)
        lambda2 = torch.cat((lambda2_expert, lambda2_dagger), dim=0)

        permuted_indices = torch.randperm(xs_datasets.size(0))
        xs_datasets = xs_datasets[permuted_indices]
        control_values = control_values[permuted_indices]
        lambda1 = lambda1[permuted_indices]
        lambda2 = lambda2[permuted_indices]

        control_values_batches = control_values

        T_batches = torch.tensor(T_batches).float()

        if torch.cuda.is_available():
            T_batches = T_batches.cuda()
            xs_datasets = xs_datasets.cuda()
            control_values_batches = control_values_batches.cuda()

        return T_batches, xs_datasets, control_values_batches, lambda1, lambda2

    
 