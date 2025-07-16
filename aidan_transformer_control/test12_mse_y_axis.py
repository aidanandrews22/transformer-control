import numpy as np
import torch
import matplotlib.pyplot as plt
import workCon
import os
from eval import get_model_from_run
from tqdm import tqdm
import ipdb
import traceback
import random
import math
import generate_dataset


plot_label = 'multi10_mse_Alternating'
save_results = "blah.txt"
log_info = "zzzzzer.txt"
model_name= "test"
model_run_id= "9b6c9448-02d0-4283-be63-13e5a053254c"
model_checkpoint_step=600
folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}"

    

total_time = 1.5
dt = 0.01
Num_of_context = 5
Num_of_pendulums = 10




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

def mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true states.

    Args:
        theta_model (np.ndarray or list): Model's predicted theta 
        thetadot_model (np.ndarray or list): Model's predicted thetadot 
        theta_rk4 (np.ndarray or list): True theta using RK4
        thetadot_rk4 (np.ndarray or list): True thetadot using RK4

    Returns:
        float: The computed MSE value, representing the average squared difference between 
        the predicted and true states.
    """
    xs_pred = torch.tensor(np.column_stack((theta_model, thetadot_model)), dtype=torch.float32)
    xs_true = torch.tensor(np.column_stack((theta_rk4, thetadot_rk4)), dtype=torch.float32)
    return (xs_true - xs_pred).pow(2).mean().item()

def load_model(run_dir, name, run_id, step):
    """
    Loads a pre-trained model and its configuration from a specified run directory.

    Args:
        run_dir (str): The base directory containing the model runs.
        name (str): The name of the model
        run_id (str): The unique identifier for the specific run to load the model from.
        step (int, optional): The training step at which to load the model checkpoint. 

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model.
            - conf (dict): The configuration dictionary associated with the model run.
    """
    run_path = os.path.join(run_dir, name, run_id)
    model, conf = get_model_from_run(run_path, step=step)
    return model, conf


def generate_random_X0():
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - theta (float): A randomly generated theta, sampled uniformly from range [-π, π].
            - thetadot (float): A randomly generated thetadot, sampled uniformly from the range [-10, 10].
    """
    theta = np.random.uniform(-np.pi, np.pi)
    thetadot = np.random.uniform(-10, 10)
    return [theta, thetadot]


def run_inference_on_model(model, XData, YS, total_time, dt=0.01, context=1, start_index=1, mass = 1, length = 1):
    """
    Runs inference on a trained model to simulate the dynamics of a pendulum system over time, given an initial state and context data.

    Args:
        model (torch.nn.Module): The trained model used for predicting control inputs.
        XData (torch.Tensor): The input state  (e.g., [theta, thetadot])
        YS (torch.Tensor): The ground truth control input data
        total_time (float): Total duration of the simulation in seconds.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
        context (int, optional): The number of previous time steps used as context for the model. Defaults to 1.
        start_index (int, optional): The starting index for inference. Must be at least equal to `context`. Defaults to 1.
        mass (float, optional): The mass of the pendulum. Defaults to 1.
        length (float, optional): The length of the pendulum. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - T (np.ndarray): Array of time steps during the simulation.
            - theta_model (np.ndarray): Array of predicted theta over time.
            - thetadot_model (np.ndarray): Array of predicted thetadot over time.

    Raises:
        AssertionError: If `start_index` is less than `context`.
    """
    assert start_index >= context, "start_index must be at least equal to context"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    T = np.arange(0, total_time + dt, dt)
    n_steps = len(T)

    XData_context = XData[start_index - context:start_index].to(device)
    YS_context = YS[start_index - context:start_index].to(device)
    
    counter = 0
    for i in range(start_index, n_steps):
        with torch.no_grad():
            u_pred = model(XData_context, YS_context, inf = "yes")
            u = u_pred[0][-2].cpu().numpy()  

        theta, thetadot = workCon.single_step_inverted_pendulum_rk4(
            [XData_context[-1][0].cpu().numpy(), XData_context[-1][1].cpu().numpy()],
            u,
            dt, mass = mass, length = length
        )
     
        new_X = torch.tensor([theta, thetadot], dtype=torch.float32, device=device).unsqueeze(0)
        XData_context = torch.cat((XData_context, new_X), dim=0)
   
        new_Y = torch.tensor(u, dtype=torch.float32, device=device).squeeze() 
       
        if counter == 0:
            YS_context = torch.cat((YS_context[:-1], new_Y.unsqueeze(0)), dim=0) 
            counter = 1
        else:
            YS_context = torch.cat((YS_context, new_Y.unsqueeze(0)), dim=0)
  
    theta_model = XData_context[:, 0].cpu().numpy()
    thetadot_model = XData_context[:, 1].cpu().numpy()
    return T, theta_model, thetadot_model

def plot_and_log_results(x_axis, context_lengths, save_results_path, folder_name, plot_label):
    """
    Logs the x-axis and y-axis values to a file and plots the accumulated MSE vs. context length graph.

    Args:
        x_axis (list): The x-axis values (e.g., context lengths).
        context_lengths (list): The y-axis values (e.g., accumulated MSE).
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.

    Returns:
        None
    """
    with open(save_results_path, "a") as f:
        f.write(f"graphs x_axis: {x_axis}\n")
        f.write(f"graphs y_axis: {context_lengths}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, context_lengths, marker='o', label=plot_label)
    plt.xticks(x_axis)
    plt.xlabel('Context Length')
    plt.ylabel('Accumulated MSE')
    plt.title('Accumulated MSE vs Context Length')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"mse_vs_context_length({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')


def main():
    """_summary_
    """
    model, _ = load_model(
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        step=model_checkpoint_step
    )

    os.makedirs(folder_name, exist_ok=True)
    save_results_path = os.path.join(folder_name, save_results)
    log_info_path = os.path.join(folder_name, log_info)
    context_lengths = [0] * Num_of_context
    start_indices = [Num_of_context]

    masses = [generate_dataset.sample_bounded_gaussian() for _ in range(Num_of_pendulums)]
    lengths = [generate_dataset.sample_bounded_gaussian() for _ in range(Num_of_pendulums)]
    with open(log_info_path, "w") as file:
        for mass, length in tqdm(zip(masses, lengths), desc="MultiPendulum", total=len(masses), leave=False):
            X0 = generate_random_X0()
            T1, theta_rk4, thetadot_rk4, control_values_rk4, K_values = workCon.checking(
                    X0, total_time, method='rk4', dt=dt, mass = mass, length = length
                )
            

            file.write(f"Current pendulum mass: {mass}\n")
            file.write(f"Current pendulum length: {length}\n")
            file.write(f"X0: {X0}\n")
            file.write(f"T1 (RK4 Time): {T1}\n")
            file.write(f"K_values (RK4 K_values): {K_values}\n\n")


            xs_dataset = np.column_stack((theta_rk4, thetadot_rk4))
            xs_dataset = torch.tensor(xs_dataset).float().cuda()
            control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()
            for start_index in start_indices:
                file.write(f"  Start Index: {start_index}\n")
                theta_rk4_temp = theta_rk4[start_index - 1:]
                thetadot_rk4_temp = thetadot_rk4[start_index - 1:]
                for context1 in tqdm(range(len(context_lengths)), desc=f"Context Loop (Start Index {start_index})", leave=False):
                    if start_index < context1:
                        continue
                    T_model, theta_model2, thetadot_model2 = run_inference_on_model(
                            model, xs_dataset, control_values_rk4, total_time, dt, context=context1 + 1, start_index=start_index, mass = mass, length = length
                        )
                    theta_model = theta_model2[context1:]
                    thetadot_model = thetadot_model2[context1:]
                    mse_loss = mse(theta_model, thetadot_model, theta_rk4_temp, thetadot_rk4_temp)


                    file.write(f"    Context1: {context1 + 1}\n")
                    file.write(f"    Before Splice Theta Model: {theta_model2.tolist()}\n")
                    file.write(f"    Before Splice Thetadot Model: {thetadot_model2.tolist()}\n")
                    file.write(f"    Before Splice theta_rk4_temp: {theta_rk4.tolist()}\n")
                    file.write(f"    Before Splice thetadot_rk4_temp: {thetadot_rk4.tolist()}\n")
                    file.write(f"    After Splice Theta Model: {theta_model.tolist()}\n")
                    file.write(f"    After Splice theta_rk4_temp: {theta_rk4_temp.tolist()}\n")
                    file.write(f"    After Splice Thetadot Model: {thetadot_model.tolist()}\n")
                    file.write(f"    After Splice thetadot_rk4_temp: {thetadot_rk4_temp.tolist()}\n")
                    file.write(f"    MSE Loss: {mse_loss}\n")
                    file.write(f"    Context Lengths (Accum MSE): {context_lengths}\n\n")


                    context_lengths[context1] += mse_loss

    x_axis = list(range(1, len(context_lengths) + 1))
    plot_and_log_results(x_axis, context_lengths, save_results_path, folder_name, plot_label)

try:
    main()
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()
