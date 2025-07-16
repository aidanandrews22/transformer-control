import matplotlib.pylab as plt
from matplotlib import rc, animation, patches
rc('animation', html='html5')
import numpy as np
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_are

import torch
import numpy as np
from scipy.linalg import solve_continuous_are


import matplotlib.pylab as plt
from matplotlib import rc, animation, patches
rc('animation', html='html5')
import numpy as np
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_are

import torch
import numpy as np
from scipy.linalg import solve_continuous_are

# def find_nearest_upright(theta):
#     # return np.pi * (2 * torch.round((theta - np.pi) / (2 * np.pi)) + 1)
#     return torch.pi * (2 * torch.round((theta - torch.pi) / (2 * torch.pi)) + 1)

def find_nearest_upright(theta):
    """
    theta: (B,) tensor of angles
    return: (B, 4) tensor of upright points nearest to each theta
    """
    upright_angles = torch.pi * (2 * torch.round((theta - torch.pi) / (2 * torch.pi)) + 1)
    # zeros = torch.zeros_like(upright_angles)
    # upright_points = torch.stack([zeros, upright_angles, zeros, zeros], dim=1)  # shape (B, 4)
    # print(f"upright_points shape: {upright_points.shape}")
    # return upright_points
    return upright_angles


def make_batched_controller(m_c, m_1, l_1, device='cuda:0'):
    """
    Create a batched controller function for the cartpole system,
    where m_c, m_1, l_1 are tensors of shape (batch_size,) on device.
    """
    g = 9.81

    batch_size = m_c.shape[0]

    # Compute LQR gains for each trajectory batch individually using scipy (CPU)
    # We'll do this in a loop because scipy does not support batch.
    # Then transfer to device as tensor batch.
    K_lqr_batch = []
    for i in range(batch_size):
        A_eq = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g*m_1[i].item()/m_c[i].item(), 0, 0],
            [0, g*(m_1[i].item()+m_c[i].item())/(l_1[i].item()*m_c[i].item()), 0, 0]
        ])
        B_eq = np.array([
            0,
            0,
            1/m_c[i].item(),
            1/(l_1[i].item()*m_c[i].item())
        ]).reshape(-1,1)
        Q = np.eye(4)
        R = np.eye(1)
        X_are = solve_continuous_are(A_eq, B_eq, Q, R)
        K_lqr = np.linalg.inv(R) @ B_eq.T @ X_are
        K_lqr_batch.append(K_lqr[0])  # shape (4,)
    K_lqr = torch.tensor(np.stack(K_lqr_batch), dtype=torch.float32, device=device)  # (batch_size, 4)

    eq_point = torch.tensor([0, np.pi, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
    eq_point_sym = torch.tensor([0, -np.pi, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)

    # def f_lin_feedback(state, use_sym):
    def f_lin_feedback(state):
        """
        Batched linear feedback control, state shape: (batch, 4)
        use_sym: boolean mask tensor of shape (batch,) deciding which equilibrium
        """
        # eq_pts = torch.where(use_sym.unsqueeze(-1), eq_point_sym, eq_point)
        eq_theta = find_nearest_upright(state[:, 1])
        eq_pts = torch.stack([torch.zeros_like(eq_theta), eq_theta, torch.zeros_like(eq_theta), torch.zeros_like(eq_theta)], dim=1)

        diff = state - eq_pts
        # Elementwise multiply and sum along last dim
        # u = -(K_lqr * diff).sum(dim=1)
        # print(f"eq_pts: {eq_pts.shape}, K_lqr: {K_lqr.shape}")
        u = -(K_lqr * diff).sum(dim=1)
        # print(f"f_lin_feedback: {u.shape}")
        return u

    def controller(state, t, args):
        """
        state: (batch, 4)
        args: tuple of two tensors (switched: bool batch, use_sym: bool batch)
        returns:
          - control force u: (batch,)
          - updated args (switched, use_sym)
        """
        switched, use_sym = args
        x, theta1, x_dot, theta1_dot = state[:,0], state[:,1], state[:,2], state[:,3]

        dist_eq_point = torch.norm(state[:,1:] - eq_point[:,1:], dim=1)
        dist_eq_point_sym = torch.norm(state[:,1:] - eq_point_sym[:,1:], dim=1)

        not_switched = ~switched
        # switch_to_true = not_switched & ((dist_eq_point <= 0.5) | (dist_eq_point_sym <= 0.5))

        # switch_to_true = not_switched & (((torch.abs(state[:, 1] - eq_point[:, 1]) < 0.5) &
        #                                   (torch.abs(state[:, 3] - eq_point[:, 3]) < 0.5)) |
        #                                     ((torch.abs(state[:, 1] - eq_point_sym[:, 1]) < 0.5) &
        #                                     (torch.abs(state[:, 3] - eq_point_sym[:, 3]) < 0.5)))

        switch_to_true = not_switched & (
            ((torch.abs(state[:, 1] - find_nearest_upright(state[:, 1])) < 0.25) &
             (torch.abs(state[:, 3] - eq_point[:, 3]) < 0.25)) 
            # ((torch.abs(state[:, 1] - find_nearest_upright(eq_point_sym[:, 1])) < 0.15) &
            #  (torch.abs(state[:, 3] - eq_point_sym[:, 3]) < 0.15))
        )

        switched = switched | switch_to_true
        # use_sym = torch.where(dist_eq_point_sym <= 0.5, torch.ones_like(use_sym), use_sym)

        # u_lqr = f_lin_feedback(state, use_sym)
        u_lqr = f_lin_feedback(state)

        cos_theta1 = torch.cos(theta1)
        sin_theta1 = torch.sin(theta1)

        # Energy-based swing-up parameters (scalar)
        # K_e, K_p, K_d = 0.2, 0.8, 1.0
        K_e, K_p, K_d = 0.3, 1.0, 1.0

        # Compute Etilde using per-trajectory parameters with broadcasting
        # print(f"m_1 device: {m_1.device}, l_1 device: {l_1.device}, g device: {torch.tensor(g).device}")
        # print(f"theta1 device: {theta1.device}, theta1_dot device: {theta1_dot.device}")
        # global m_1, l_1, g, m_c
        # m_1 = m_1.to(theta1.device)
        # l_1 = l_1.to(theta1.device)
        # g = torch.tensor(g, device=theta1.device)
        # m_c = m_c.to(theta1.device)
        
        Etilde = (-g * l_1 * m_1
                  + 0.5 * l_1 * m_1 * (-2 * g * cos_theta1 + l_1 * theta1_dot * theta1_dot))

        xpp_desired = K_e * cos_theta1 * theta1_dot * Etilde - K_p * x - K_d * x_dot

        theta1_pp = -cos_theta1 / l_1 * xpp_desired - g / l_1 * sin_theta1

        f_swingup = (m_1 + m_c) * xpp_desired + cos_theta1 * l_1 * m_1 * theta1_pp - sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot

        u = torch.where(switched, u_lqr, f_swingup)
        mode = torch.where(switched, torch.ones_like(u), torch.zeros_like(u))

        return u, (switched, use_sym), mode

    return controller


def make_deriv_vectorized(f_controller, m_c, m_1, l_1, device='cuda:0'):
    """
    Make batched derivative function for RK4 integrator.
    m_c, m_1, l_1: (batch_size,) tensors on device
    """
    g = 9.81

    def deriv(state, t, args):
        x, theta1, x_dot, theta1_dot = state[:,0], state[:,1], state[:,2], state[:,3]
        u, updated_args, mode = f_controller(state, t, args)

        cos_theta1 = torch.cos(theta1)
        sin_theta1 = torch.sin(theta1)

        # Build mass matrix A batch (batch, 2, 2)
        A11 = m_1 + m_c  # (batch,)
        A12 = m_1 * l_1 * cos_theta1
        A21 = A12
        A22 = m_1 * l_1 * l_1
        A = torch.stack([
            torch.stack([A11, A12], dim=1),
            torch.stack([A21, A22], dim=1)
        ], dim=1)  # shape (batch, 2, 2)

        b1 = u + sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot
        b2 = -m_1 * l_1 * g * sin_theta1
        b = torch.stack([b1, b2], dim=1)

        accel = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

        dydx = torch.zeros_like(state)
        dydx[:, 0] = x_dot
        dydx[:, 1] = theta1_dot
        dydx[:, 2] = accel[:, 0]
        dydx[:, 3] = accel[:, 1]

        return dydx, u, updated_args, mode

    return deriv


def rk4_step(deriv_func, y, t, dt, args):
    k1, u1, args1, mode1 = deriv_func(y, t, args)
    k2, u2, args2, mode2 = deriv_func(y + 0.5 * dt * k1, t + 0.5 * dt, args1)
    k3, u3, args3, mode3 = deriv_func(y + 0.5 * dt * k2, t + 0.5 * dt, args2)
    k4, u4, args4, mode4 = deriv_func(y + dt * k3, t + dt, args3)
    y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next, u4, args4, mode4


def simulate_cartpole_batch(m_c, m_1, l_1, y0=None,
                           total_time=14.0, dt=0.025,
                           device='cuda:0'):
    """
    Batched cart-pole simulation with RK4 integration.

    m_c, m_1, l_1: (batch_size,) tensors on device

    Returns:
        t: (num_steps,)
        ys: (num_steps, batch_size, 4)
        controls: (num_steps-1, batch_size)
    """
    batch_size = m_c.shape[0]
    num_steps = int(total_time / dt)
    t = torch.linspace(0, total_time, num_steps, device=device)

    if y0 is None:
        y0 = torch.stack([
            torch.zeros(batch_size, device=device),          # x
            torch.ones(batch_size, device=device) * np.pi,  # theta1
            torch.zeros(batch_size, device=device),          # x_dot
            torch.zeros(batch_size, device=device)           # theta1_dot
        ], dim=1)
    else:
        y0 = y0.to(device)

    ys = torch.zeros((num_steps, batch_size, 4), dtype=torch.float32, device=device)
    ys[0] = y0
    controls = torch.zeros((num_steps - 1, batch_size), dtype=torch.float32, device=device)
    control_modes = torch.zeros((num_steps - 1, batch_size), dtype=torch.float32, device=device)

    controller = make_batched_controller(m_c, m_1, l_1, device=device)
    deriv_func = make_deriv_vectorized(controller, m_c, m_1, l_1, device=device)

    switched = torch.zeros(batch_size, dtype=torch.bool, device=device)
    use_sym = torch.zeros(batch_size, dtype=torch.bool, device=device)
    args = (switched, use_sym)

    for i in range(num_steps - 1):
        y_curr = ys[i]
        t_curr = t[i]
        y_next, u, args, mode = rk4_step(deriv_func, y_curr, t_curr, dt, args)
        ys[i+1] = y_next
        controls[i] = u
        control_modes[i] = mode

    return t, ys, controls, control_modes

def checking(Y0, total_time, m_c=torch.tensor([1.0]), m_1=torch.tensor([1.0]), l_1=torch.tensor([1.0]), method='rk4', dt=0.025, device='cuda:0'):
    """
    Check the batched cart-pole system simulation with given parameters.
    
    Parameters:
    - Y0: initial state of the system (x, theta1, x_dot, theta1_dot) as a tensor of shape (batch_size, 4)
    - total_time: total simulation time
    - m_c: mass of the cart
    - m_1: mass of the pendulum
    - l_1: length of the pendulum
    - method: integration method ('rk4' or 'odeint')
    - dt: time step for the simulation
    - device: device to run the simulation on ('cpu' or 'cuda')
    """
    if method == 'rk4':
        t, y, control_logger, control_mode_logger = simulate_cartpole_batch(m_c=m_c, m_1=m_1, l_1=l_1, y0=Y0, total_time=total_time, dt=dt, device=device)
        y = y.permute(1, 0, 2)  # (batch_size, num_steps, 4)
        control_logger = control_logger.permute(1, 0)  # (batch_size, num_steps-1)
        control_mode_logger = control_mode_logger.permute(1, 0) # (batch_size, num_steps-1)
    else:
        raise NotImplementedError("Only 'rk4' method is implemented in this example.")

    return t, y, control_logger, control_mode_logger



# def make_batched_controller(m_c, m_1, l_1, device='cuda:0'):
#     """
#     Create a batched controller function for the cartpole system,
#     where m_c, m_1, l_1 are tensors of shape (batch_size,) on device.
#     """
#     g = 9.81

#     batch_size = m_c.shape[0]

#     # Compute LQR gains for each trajectory batch individually using scipy (CPU)
#     # We'll do this in a loop because scipy does not support batch.
#     # Then transfer to device as tensor batch.
#     K_lqr_batch = []
#     for i in range(batch_size):
#         A_eq = np.array([
#             [0, 0, 1, 0],
#             [0, 0, 0, 1],
#             [0, g*m_1[i].item()/m_c[i].item(), 0, 0],
#             [0, g*(m_1[i].item()+m_c[i].item())/(l_1[i].item()*m_c[i].item()), 0, 0]
#         ])
#         B_eq = np.array([
#             0,
#             0,
#             1/m_c[i].item(),
#             1/(l_1[i].item()*m_c[i].item())
#         ]).reshape(-1,1)
#         Q = np.eye(4)
#         R = np.eye(1)
#         X_are = solve_continuous_are(A_eq, B_eq, Q, R)
#         K_lqr = np.linalg.inv(R) @ B_eq.T @ X_are
#         K_lqr_batch.append(K_lqr[0])  # shape (4,)
#     K_lqr = torch.tensor(np.stack(K_lqr_batch), dtype=torch.float32, device=device)  # (batch_size, 4)

#     eq_point = torch.tensor([0, np.pi, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
#     eq_point_sym = torch.tensor([0, -np.pi, 0, 0], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)

#     def f_lin_feedback(state, use_sym):
#         """
#         Batched linear feedback control, state shape: (batch, 4)
#         use_sym: boolean mask tensor of shape (batch,) deciding which equilibrium
#         """
#         eq_pts = torch.where(use_sym.unsqueeze(-1), eq_point_sym, eq_point)
#         diff = state - eq_pts
#         # Elementwise multiply and sum along last dim
#         # u = -(K_lqr * diff).sum(dim=1)
#         u = -(K_lqr * diff).sum(dim=1)
#         return u

#     def controller(state, t, args):
#         """
#         state: (batch, 4)
#         args: tuple of two tensors (switched: bool batch, use_sym: bool batch)
#         returns:
#           - control force u: (batch,)
#           - updated args (switched, use_sym)
#         """
#         switched, use_sym = args
#         x, theta1, x_dot, theta1_dot = state[:,0], state[:,1], state[:,2], state[:,3]

#         dist_eq_point = torch.norm(state[:,1:] - eq_point[:,1:], dim=1)
#         dist_eq_point_sym = torch.norm(state[:,1:] - eq_point_sym[:,1:], dim=1)

#         not_switched = ~switched
#         # switch_to_true = not_switched & ((dist_eq_point <= 0.5) | (dist_eq_point_sym <= 0.5))

#         switch_to_true = not_switched & (((torch.abs(state[:, 1] - eq_point[:, 1]) < 0.5) &
#                                           (torch.abs(state[:, 3] - eq_point[:, 3]) < 0.5)) |
#                                             ((torch.abs(state[:, 1] - eq_point_sym[:, 1]) < 0.5) &
#                                             (torch.abs(state[:, 3] - eq_point_sym[:, 3]) < 0.5)))

#         switched = switched | switch_to_true
#         use_sym = torch.where(dist_eq_point_sym <= 0.5, torch.ones_like(use_sym), use_sym)

#         u_lqr = f_lin_feedback(state, use_sym)

#         cos_theta1 = torch.cos(theta1)
#         sin_theta1 = torch.sin(theta1)

#         # Energy-based swing-up parameters (scalar)
#         # K_e, K_p, K_d = 0.2, 0.8, 1.0
#         K_e, K_p, K_d = 0.3, 1.0, 1.0

#         # Compute Etilde using per-trajectory parameters with broadcasting
#         # print(f"m_1 device: {m_1.device}, l_1 device: {l_1.device}, g device: {torch.tensor(g).device}")
#         # print(f"theta1 device: {theta1.device}, theta1_dot device: {theta1_dot.device}")
#         # global m_1, l_1, g, m_c
#         # m_1 = m_1.to(theta1.device)
#         # l_1 = l_1.to(theta1.device)
#         # g = torch.tensor(g, device=theta1.device)
#         # m_c = m_c.to(theta1.device)
        
#         Etilde = (-g * l_1 * m_1
#                   + 0.5 * l_1 * m_1 * (-2 * g * cos_theta1 + l_1 * theta1_dot * theta1_dot))

#         xpp_desired = K_e * cos_theta1 * theta1_dot * Etilde - K_p * x - K_d * x_dot

#         theta1_pp = -cos_theta1 / l_1 * xpp_desired - g / l_1 * sin_theta1

#         f_swingup = (m_1 + m_c) * xpp_desired + cos_theta1 * l_1 * m_1 * theta1_pp - sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot

#         u = torch.where(switched, u_lqr, f_swingup)
#         mode = torch.where(switched, torch.ones_like(u), torch.zeros_like(u))

#         return u, (switched, use_sym), mode

#     return controller


# def make_deriv_vectorized(f_controller, m_c, m_1, l_1, device='cuda:0'):
#     """
#     Make batched derivative function for RK4 integrator.
#     m_c, m_1, l_1: (batch_size,) tensors on device
#     """
#     g = 9.81

#     def deriv(state, t, args):
#         x, theta1, x_dot, theta1_dot = state[:,0], state[:,1], state[:,2], state[:,3]
#         u, updated_args, mode = f_controller(state, t, args)

#         cos_theta1 = torch.cos(theta1)
#         sin_theta1 = torch.sin(theta1)

#         # Build mass matrix A batch (batch, 2, 2)
#         A11 = m_1 + m_c  # (batch,)
#         A12 = m_1 * l_1 * cos_theta1
#         A21 = A12
#         A22 = m_1 * l_1 * l_1
#         A = torch.stack([
#             torch.stack([A11, A12], dim=1),
#             torch.stack([A21, A22], dim=1)
#         ], dim=1)  # shape (batch, 2, 2)

#         b1 = u + sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot
#         b2 = -m_1 * l_1 * g * sin_theta1
#         b = torch.stack([b1, b2], dim=1)

#         accel = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

#         dydx = torch.zeros_like(state)
#         dydx[:, 0] = x_dot
#         dydx[:, 1] = theta1_dot
#         dydx[:, 2] = accel[:, 0]
#         dydx[:, 3] = accel[:, 1]

#         return dydx, u, updated_args, mode

#     return deriv


# def rk4_step(deriv_func, y, t, dt, args):
#     k1, u1, args1, mode1 = deriv_func(y, t, args)
#     k2, u2, args2, mode2 = deriv_func(y + 0.5 * dt * k1, t + 0.5 * dt, args1)
#     k3, u3, args3, mode3 = deriv_func(y + 0.5 * dt * k2, t + 0.5 * dt, args2)
#     k4, u4, args4, mode4 = deriv_func(y + dt * k3, t + dt, args3)
#     y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
#     return y_next, u4, args4, mode4


# def simulate_cartpole_batch(m_c, m_1, l_1, y0=None,
#                            total_time=14.0, dt=0.025,
#                            device='cuda:0'):
#     """
#     Batched cart-pole simulation with RK4 integration.

#     m_c, m_1, l_1: (batch_size,) tensors on device

#     Returns:
#         t: (num_steps,)
#         ys: (num_steps, batch_size, 4)
#         controls: (num_steps-1, batch_size)
#     """
#     batch_size = m_c.shape[0]
#     num_steps = int(total_time / dt)
#     t = torch.linspace(0, total_time, num_steps, device=device)

#     if y0 is None:
#         y0 = torch.stack([
#             torch.zeros(batch_size, device=device),          # x
#             torch.ones(batch_size, device=device) * np.pi,  # theta1
#             torch.zeros(batch_size, device=device),          # x_dot
#             torch.zeros(batch_size, device=device)           # theta1_dot
#         ], dim=1)
#     else:
#         y0 = y0.to(device)

#     ys = torch.zeros((num_steps, batch_size, 4), dtype=torch.float32, device=device)
#     ys[0] = y0
#     controls = torch.zeros((num_steps - 1, batch_size), dtype=torch.float32, device=device)
#     control_modes = torch.zeros((num_steps - 1, batch_size), dtype=torch.float32, device=device)

#     controller = make_batched_controller(m_c, m_1, l_1, device=device)
#     deriv_func = make_deriv_vectorized(controller, m_c, m_1, l_1, device=device)

#     switched = torch.zeros(batch_size, dtype=torch.bool, device=device)
#     use_sym = torch.zeros(batch_size, dtype=torch.bool, device=device)
#     args = (switched, use_sym)

#     for i in range(num_steps - 1):
#         y_curr = ys[i]
#         t_curr = t[i]
#         y_next, u, args, mode = rk4_step(deriv_func, y_curr, t_curr, dt, args)
#         ys[i+1] = y_next
#         controls[i] = u
#         control_modes[i] = mode

#     return t, ys, controls, control_modes

# def checking(Y0, total_time, m_c=torch.tensor([1.0]), m_1=torch.tensor([1.0]), l_1=torch.tensor([1.0]), method='rk4', dt=0.025, device='cuda:0'):
#     """
#     Check the batched cart-pole system simulation with given parameters.
    
#     Parameters:
#     - Y0: initial state of the system (x, theta1, x_dot, theta1_dot) as a tensor of shape (batch_size, 4)
#     - total_time: total simulation time
#     - m_c: mass of the cart
#     - m_1: mass of the pendulum
#     - l_1: length of the pendulum
#     - method: integration method ('rk4' or 'odeint')
#     - dt: time step for the simulation
#     - device: device to run the simulation on ('cpu' or 'cuda')
#     """
#     if method == 'rk4':
#         t, y, control_logger, control_mode_logger = simulate_cartpole_batch(m_c=m_c, m_1=m_1, l_1=l_1, y0=Y0, total_time=total_time, dt=dt, device=device)
#         y = y.permute(1, 0, 2)  # (batch_size, num_steps, 4)
#         control_logger = control_logger.permute(1, 0)  # (batch_size, num_steps-1)
#         control_mode_logger = control_mode_logger.permute(1, 0) # (batch_size, num_steps-1)
#     else:
#         raise NotImplementedError("Only 'rk4' method is implemented in this example.")

#     return t, y, control_logger, control_mode_logger



