# CartPole Data Generation for Transformer Control

This directory contains scripts to generate training data from the CartPole swing-up controller in `aidan_gym.py` for training transformer models.

## Overview

The data generation system creates trajectories using the hybrid controller (energy-based swing-up + LQR stabilization) from `aidan_gym.py`. The generated data is compatible with the existing transformer training pipeline.

## Files

- `generate_cartpole_data.py`: Main data generation script
- `test_data_generation.py`: Test script to verify data generation
- `conf/cartpole_aidan.yaml`: Configuration file for training
- `README_cartpole_data.md`: This documentation

## Data Format

The generated data follows the same format as the existing codebase:

- **States (xs)**: `(batch_size, trajectory_length, 4)` tensor
  - `[x, x_dot, theta, theta_dot]` - cart position, velocity, pole angle, angular velocity
- **Controls (ys)**: `(batch_size, trajectory_length, 2)` tensor  
  - `[force, mode_flag]` - control force and controller mode (0=swing-up, 1=LQR)

## Usage

### 1. Generate Training Data

```bash
# Generate default dataset (100 batches of 64 trajectories each)
python generate_cartpole_data.py

# Custom parameters
python generate_cartpole_data.py \
    --output_dir dataset_cartpole_custom \
    --num_batches 200 \
    --batch_size 32 \
    --trajectory_length 500 \
    --max_force 400.0
```

### 2. Test Data Generation

```bash
# Run tests and view sample trajectories
python test_data_generation.py
```

### 3. Train Transformer Model

```bash
# Train using the generated data
python trainSequential.py --config conf/cartpole_aidan.yaml
```

## Configuration Options

### Data Generation Parameters

- `--output_dir`: Directory to save pickle files (default: `dataset_cartpole_aidan`)
- `--num_batches`: Number of batches to generate (default: 100)
- `--batch_size`: Trajectories per batch (default: 64)
- `--trajectory_length`: Timesteps per trajectory (default: 400)
- `--max_force`: Maximum control force (default: 500.0)

### Physical Parameter Randomization

The system automatically randomizes physical parameters for each trajectory:
- **Pole mass**: 0.5x to 1.5x the default mass
- **Cart mass**: 0.8x to 1.2x the default mass  
- **Pole length**: 0.8x to 1.2x the default length

This creates a diverse dataset that should improve model generalization.

## Model Architecture

The transformer model uses the same architecture as the existing codebase:

- **Input**: Interleaved state-control sequences `[x₀, u₀, x₁, u₁, ...]`
- **Output**: Predicted next control `û` and next state `x̂`
- **Attention**: Causal self-attention over the full trajectory history

## Controller Details

The hybrid controller combines:

1. **Energy-based swing-up**: Pumps energy into the pole to reach upright position
2. **LQR stabilization**: Maintains balance once near upright (< 0.5 rad from vertical)

The controller mode flag in the data indicates which controller was active:
- `0`: Swing-up mode (energy pumping)
- `1`: LQR mode (stabilization)

## Expected Results

- **Trajectory diversity**: Wide range of initial conditions and physical parameters
- **Successful control**: Most trajectories should reach and maintain upright position
- **Mode switching**: Clear transitions between swing-up and LQR modes
- **Realistic dynamics**: Physics-based trajectories from Gymnasium CartPole environment

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `aidan_gym.py` is in the same directory
2. **Gymnasium version**: Requires `gymnasium` (not `gym`)
3. **Memory usage**: Large datasets may require chunked loading during training

### Performance Tips

- Use smaller batch sizes if running out of memory
- Reduce trajectory length for faster generation
- Monitor controller success rate in test output

## Integration with Existing Code

The generated data is fully compatible with:
- `trainSequential.py`: Main training script
- `models.py`: Transformer architecture
- `schema.py`: Configuration validation
- `tasks.py`: Loss functions

Simply point the training configuration to your generated dataset directory. 