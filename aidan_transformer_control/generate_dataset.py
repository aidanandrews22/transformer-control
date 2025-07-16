import os
import random
from tqdm import tqdm
from samplers import PendulumSampler
from curriculum import Curriculum
from random import randint
import uuid
import ipdb

from quinine import QuinineArgumentParser
import torch
import yaml
from schema import schema
from models import build_model
import math
import random
import numpy as np
import torch
import pickle

seed = [1]
torch.backends.cudnn.benchmark = True

def reseed_all(seed):
    """
    Reseeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value used to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_valid_masses_and_lengths( dt=0.01, mean=3, variance=1, lowerbound=1, upperbound=5):
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a bounded Gaussian distribution.

    Args:
        dt (float, optional): The time step for simulation. Defaults to 0.01.
        mean (int, optional): The mean of the Gaussian distribution used for sampling. Defaults to 3.
        variance (int, optional): The variance of the Gaussian distribution. Defaults to 1.
        lowerbound (int, optional): The lower bound for the sampled values. Defaults to 1.
        upperbound (int, optional): The upper bound for the sampled values. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - masses (float): A valid mass value sampled from the Gaussian distribution.
            - lengths (float): A valid length value sampled from the Gaussian distribution.
    """
    while True:
        masses = sample_bounded_gaussian(mean, variance, lowerbound, upperbound)
        lengths = sample_bounded_gaussian(mean, variance, lowerbound, upperbound)
        if is_valid_mass_length(masses, lengths, dt=dt):
            return masses, lengths
        seed[0] += 1
        reseed_all(seed[0])


def sample_bounded_gaussian(mean=3, stddev=1, lower_bound=1, upper_bound=5):
    """
    Samples a value from a Gaussian distribution within specified bounds.

    Args:
        mean (float): The mean of the Gaussian distribution.
        stddev (float): The standard deviation of the Gaussian distribution.
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled value within the specified bounds.
    """
    while True:
        value = random.gauss(mean, stddev)
        if lower_bound <= value <= upper_bound:
            return value
        seed[0] += 1
        reseed_all(seed[0])


def is_valid_mass_length(mass, length, dt):
    """
    Validates mass and length values to ensure they do not cause issues with rk4 during simulation.

    Args:
        mass (float): The mass of the pendulum.
        length (float): The length of the pendulum.
        dt (float): The time step for the simulation.

    Returns:
        bool: True if the mass and length values are valid, False otherwise.
    """
    g = 9.81 
    natural_frequency = math.sqrt(g / length)  
    moment_of_inertia = mass * length**2
    if moment_of_inertia < 1e-6: 
        return False
    if natural_frequency * dt > 0.1: 
        return False
    return True


def save_pickle(data, pickle_path):
    """
    Saves data to a pickle file.

    Args:
        data (any): The data to save.
        pickle_path (str): The path to the pickle file where the data will be saved.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def append_to_seed_file(seed_file, iteration, seed, masses, lengths, k_values, xs_shape):
    """
    Appends simulation details to a seed file for tracking and reproducibility.

    Args:
        seed_file (str): The path to the seed file.
        iteration (int): The iteration number of the simulation.
        seed (int): The seed value used for the simulation.
        masses (list[float]): The mass used in the simulation.
        lengths (list[float]): The length used in the simulation.
        k_values (list[float]): The gain matrix used in the simulation.
        xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
    """
    with open(seed_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Seed: {seed}\n")
        f.write("Masses: " + str(masses) + "\n")
        f.write("Lengths: " + str(lengths) + "\n")
        f.write("K: " + str(k_values) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Geneates training datasets for an inverted pendulum system simulation and saves them 
    as pickle files along with metadata for reproducibility.

    Args:
        args (Namespace):
            - args.training.train_steps (int): The total number of pickle files.
            - args.training.batch_size (int): batch of each pickle file.
            - args.training.curriculum (dict): Curriculum settings to adjust training parameters dynamically.
            - args.dataset_filesfolder (str): The directory where dataset files and logs are stored.
            - args.dataset_logger_textfile (str): The name of the file for logging dataset info.
            - args.pickle_folder (str): The dataset_filesfolder subfolder where generated dataset pickle files are saved.

    Notes:
        - The `PendulumSampler` is used to generate the dataset based on random valid pendulum parameters (masses and lengths).
        - The generated datasets are saved as pickle files named in the format `multipendulum_{i}.pkl`.
        - Metadata such as the current seed, pendulum parameters, and dataset shape is logged in a separate file for reproducibility.
        - The curriculum dynamically updates training parameters, such as the number of points in each dataset.

    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps)) 

    seed_file = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    os.makedirs(base_data_dir, exist_ok=True)

    for i in pbar:
        reseed_all(seed[0]+i)
        
        masses, lengths = get_valid_masses_and_lengths()
        sampler = PendulumSampler(n_dims=2)
        T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
        pickle_file = f'multipendulum_{i}.pkl'
        pickle_path = os.path.join(base_data_dir, pickle_file)
        save_pickle((xs, control_values), pickle_path)
        append_to_seed_file(seed_file, i, seed[0] + i, masses, lengths, k_values, xs.shape)
        curriculum.update()


def main(args):
    reseed_all(seed[0])
    make_train_data(args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    with open(os.path.join(args.dataset_filesfolder, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

