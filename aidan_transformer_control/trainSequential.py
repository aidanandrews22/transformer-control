import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import tasks
from curriculum import Curriculum
from schema import schema
from models import build_model
import wandb
import pickle
import random
import numpy as np
import torch
import gc

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True


def calculate_lyapunov_derivative(theta, thetadot, u, m=1, l=1, b=0.5, g=9.81):
    """
    Calculates the time derivative of the Lyapunov function for the inverted pendulum system,
    which is used to evaluate the stability of the system under a given control input.

    Args:
        theta (torch.Tensor): The angular position of the pendulum (in radians).
        thetadot (torch.Tensor): The angular velocity of the pendulum.
        u (torch.Tensor): The control input applied to the system.
        m (float, optional): The mass of the pendulum. Defaults to 1.
        l (float, optional): The length of the pendulum. Defaults to 1.
        b (float, optional): The damping coefficient. Defaults to 0.5.
        g (float, optional): Gravity (in m/s^2). Defaults to 9.81.

    Returns:
        torch.Tensor: The time derivative of the Lyapunov function dV/dt, showing if energy is decreasing at all time steps
    """
    dV_dtheta = m * g * l * torch.sin(theta)
    dV_dthetadot = m * l**2 * thetadot
    ddot_theta = (-b * thetadot + m * g * l * torch.sin(theta) + u) / (m * l**2)
    dV_dt = dV_dtheta * thetadot + dV_dthetadot * ddot_theta
    return dV_dt


def lyapunov_loss(xs, ys, m=1, l=1, b=0.5, g=9.81, lambda_coeff=100.0):
    """
    Getting the Lyapunov loss, which penalizes positive derivatives of the Lyapunov function

    Args:
        xs (torch.Tensor): The state trajectory of the system, with shape (batch_size, timesteps, 2),
            where last dimension is [theta, thetadot].
        ys (torch.Tensor): The control input of system, with shape (batch_size, u value).
        m (float, optional): The mass of the pendulum. Defaults to 1.
        l (float, optional): The length of the pendulum. Defaults to 1.
        b (float, optional): The damping coefficient. Defaults to 0.5.
        g (float, optional): gravity (in m/s^2). Defaults to 9.81.
        lambda_coeff (float, optional): Scaling loss. Defaults to 100.0.

    Returns:
        torch.Tensor: The mean Lyapunov loss, penalizes positive derivatives of the Lyapunov function.
    """
    theta = xs[:, :, 0]  
    thetadot = xs[:, :, 1]  
    u = ys  
    dV_dt = calculate_lyapunov_derivative(theta, thetadot, u, m, l, b, g)
    lyapunov_derivative_loss = torch.clamp(dV_dt, min=0)
    return lambda_coeff * lyapunov_derivative_loss.mean()


def log_training_info(file_path, i, args, xs, ys, output, loss):
    """
    Logs training information to txt file

    Args:
        file_path (str): This will be path of where model is saved.
        i (int): The current training iteration.
        args (Namespace): schema arguments
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): The are ground truth control u values
        output (torch.Tensor): These are predicted control u values.
        loss (torch.Tensor): loss for model update.

    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Iteration {i} - {args.model_logger_textfile}\n")
        f.write(f"xs\n{xs[0].detach().cpu().numpy()}\n\n")
        f.write(f"ys\n{ys[0].detach().cpu().numpy()}\n\n")
        f.write(f"output\n{output[0].detach().cpu().numpy()}\n\n")
        f.write(f"Loss ---- {loss.item()}\n\n\n\n")
        # f.write(f"lyapunov_loss_value ---- {lyapunov_loss_value.item()}\n\n\n\n")


def train_step(model, xs, ys, optimizer, loss_func, i, args):
    """
    Performs a single training step for the model, including forward pass, loss calculation, 
    backpropagation, and optimizer update. Also logs every 500 iteration.

    Args:
        model (torch.nn.Module): GPT2 decoder.
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): These are ground truth control u values
        optimizer (torch.optim.Optimizer): Adam optimizer
        loss_func (callable): loss function that will be used for training loss.
        i (int): current training iteration.
        args (Namespace): schema arguments

    Returns:
        tuple: A tuple containing:
            - total_loss (float): total loss value for the current iteration.
            - output (torch.Tensor): The model's predicted outputs for the input data (don't really use this for anything).
    """
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    # lyapunov_loss_value = lyapunov_loss(xs, output)
    total_loss = loss

    file_path = os.path.join(args.out_dir, args.model_logger_textfile)
    if i % 10 == 0:
        log_training_info(file_path, i, args, xs, ys, output, loss)

    total_loss.backward()
    optimizer.step()
    return total_loss.detach().item(), output.detach()


def count_files_in_folder(folder, prefix, suffix):
    """
    Counts the number of files in dataset folder that matches prefix and suffix.

    Args:
        folder (str): The path to the folder where the files are located.
        prefix (str): The prefix that the file names must start with.
        suffix (str): The file extention that the file must end with.

    Returns:
        int: the number of files in the folder.
    """
    return len([f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix)])


def load_dataset_chunk(pickle_folder, start_idx, end_idx):
    """
    Loads chunk of dataset files from specified folder within a given index range, 
    and returns the data as a list of tuples.

    Args:
        pickle_folder (str): The path to folder containing the pickle files.
        start_idx (int): The starting index of the pickle files to load.
        end_idx (int): The ending index of the pickle files to load.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
    """
    dataset = []
    with tqdm(total=end_idx - start_idx + 1, desc=f"Loading files {start_idx}-{end_idx}", leave=False) as load_pbar:
        for i in range(start_idx, end_idx + 1):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    xs, ys = pickle.load(f)
                    dataset.append((xs, ys))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset

def load_dataset_full(pickle_folder):
    """
    Loads entire dataset from a specified folder

    Args:
        pickle_folder (str): The path to the folder containing the pickle files.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.

    Returns:
        list: A list containing the loaded data from all the pickle files in the folder.
    """
    dataset = []
    total_files = count_files_in_folder(pickle_folder, "multipendulum_", ".pkl")
    with tqdm(total=total_files, desc="Loading all files", leave=False) as load_pbar:
        for i in range(total_files):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    xs, ys = pickle.load(f)
                    dataset.append((xs, ys))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset


def train(model, args):
    """
    Trains a given model on a dataset using the specified arguments and configurations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        args (Namespace): A configuration object containing training parameters and settings, including:
            - args.training.learning_rate (float): The learning rate for the optimizer.
            - args.loss (str): The name of the loss function to use (must be defined in the tasks module).
            - args.out_dir (str): Directory where training states and checkpoints will be saved.
            - args.dataset_filesfolder (str): Path to the folder containing dataset-related files.
            - args.pickle_folder (str): dataset_filesfolder subfolder name containing the pickled dataset files.
            - args.use_chunk (int): Number of chunks to divide the dataset for memory-efficient loading. default 1
            - args.wandb.log_every_steps (int): How often to log metrics
            - args.training.save_every_steps (int): How often to save model checkpoints
            - args.test_run (bool): If True, skips logging and checkpoint saving for debugging

    Raises:
        ValueError: If the specified loss function is not found in the `tasks` module.

    Notes:
        - The function supports chunk-based dataset loading if memory restraints, or loading full dataset into memory
        - Checkpoints and training states are saved here.
        - Wandb logs are done here.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    loss_function_name = args.loss
    loss_function = getattr(tasks, loss_function_name, None)
    if loss_function is None:
        raise ValueError(f"Unknown loss function: {loss_function_name}")

    state_path = os.path.join(args.out_dir, "state.pt")

    dataset_folder = args.dataset_filesfolder
    picklefolder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, picklefolder)
    current_step = 0

    total_files = count_files_in_folder(fullpicklepath, "multipendulum_", ".pkl")
    num_chunks = args.use_chunk  
    files_per_chunk = total_files // num_chunks
    remainder = total_files % num_chunks

    with tqdm(total=num_chunks, desc="Chunk Progress") as chunk_pbar:
        for chunk_idx in range(num_chunks):
            if args.use_chunk == 1:
                dataset = load_dataset_full(fullpicklepath)
            else:
                start_idx = chunk_idx * files_per_chunk
                end_idx = start_idx + files_per_chunk - 1
                if chunk_idx == num_chunks - 1:  
                    end_idx += remainder
                dataset = load_dataset_chunk(fullpicklepath, start_idx, end_idx)

            print(f"loaded chunk {chunk_idx + 1}/{num_chunks}")


            with tqdm(total=len(dataset), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                for xs, ys in dataset:
                    xs = xs.cuda()
                    ys = ys.cuda()


                    loss, output = train_step(model, xs, ys, optimizer, loss_function, current_step, args)
                    print(f"loss: {loss}")
                    current_step += 1
                    pbar.update(1)


                    if current_step % args.wandb.log_every_steps == 0 and not args.test_run:
                        wandb.log(
                            {
                                "step": current_step,
                                "loss": loss,
                            }
                        )

                    curriculum.update()

                    if current_step % args.training.save_every_steps == 0 and not args.test_run:
                        training_state = {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_step": current_step,
                        }
                        torch.save(training_state, state_path)

                        checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{current_step}.pt")
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Checkpoint saved at step {current_step}: {checkpoint_path}")


            print(f"Chunk {chunk_idx + 1}/{num_chunks} finished. unloading dataset from memory...")
            del dataset
            torch.cuda.empty_cache()
            gc.collect()
            chunk_pbar.update(1)

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


    main(args)
