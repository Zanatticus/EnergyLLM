# EnergyLLM: Dynamic LLM Edge Inference with GRPO

**Team 7**: Haebin Do, Angelica Kim, Kevin He, Alexander Ingare

> **DISCLAIMER**: This code requires an NVIDIA Jetson device to run. The system uses hardware-specific sensors and controls (GPU frequency, fan control, power monitoring via tegrastats) that are only available on Jetson platforms.

---

## Project Overview

This project implements a **Group Relative Policy Optimization (GRPO)** reinforcement learning system for dynamically optimizing Large Language Model (LLM) inference on edge devices. The system adaptively controls:

- **GPU frequency** (prefill and decode phases)
- **Batch size** (1-31 with step size 6: [1, 7, 13, 19, 25, 31])
- **Early exit layers** (9-16 layers, allowing the model to exit early when confidence is high)
- **Fan control** (thermal management)

The agent learns to balance **throughput**, **energy efficiency**, **latency**, and **temperature constraints** while maintaining model accuracy through entropy-based early exit decisions.

---

## File Structure

```
Final-Submission/
├── README.md                                    # This file
├── test_agent.py                                # Test script for GRPO checkpoint evaluation
│
├── grpo/                                        # Original GRPO implementation (without early exit)
│   ├── grpo_model_FINAL.pth                     # Pre-trained GRPO model checkpoint
│   ├── measure_grpo_overhead.py                 # Overhead measurement script
│   ├── rl_agent0.py                             # Base RL agent implementation
│   ├── rl_agent0_init_grpo_weight.py            # GRPO agent with weight initialization
│   ├── pruning_experiments.ipynb                # Model pruning experiments
│   ├── dataset_analysis_grpo_norm.png           # Dataset normalization visualization
│   ├── logs/                                    # Results and data CSV files
│   │   ├── grid_search_results_20251128_075945.csv
│   │   ├── fan_off_grid_search_results_20251129_092010.csv
│   │   ├── grpo_training_results_20251130_082847_temp.csv
│   │   ├── grpo_deploy_evaluation_20251201_013907.csv
│   │   └── grpo_overhead_log_20251202_224709.txt
│   └── Plots/                                   # Generated analysis plots
│       ├── grpo_overhead_comparison_v2.png
│       ├── plotx_figure3_search_overhead_comparison.png
│       ├── plotx_figure4_optimal_config_comparison.png
│       └── [other analysis plots]
│
├── grpo_early_exit/                             # Main GRPO implementation with early exit
│   ├── grpo_early_exit.py                       # Main training script (GRPO + early exit)
│   ├── rl_agent.py                              # RL agent with Llama-3.2-1B runner
│   ├── add_synthetic_entropy_to_csv.py          # Synthetic data generation script
│   ├── fanonoff.py                              # Fan control utilities
│   ├── logs/                                    # Training logs, CSV data, and replot script
│   │   ├── grid_search_results_with_entropy_and_fan_on.csv
│   │   ├── grid_search_results_with_entropy_and_fan_off.csv
│   │   ├── early_exit_continued_training_metrics_episode_400.csv
│   │   ├── policy_metrics_continued_episode_400.csv
│   │   └── replot_training_metrics.py           # Script to replot from CSV logs
│   └── plots/                                   # Generated training plots
│       ├── early_exit_continued_training_metrics_episode_400.png
│       ├── policy_metrics_continued_episode_400.png
│       └── early_exit_training_metrics_4_to_16_layers.png
│
└── model_compression_experiments/               # Model compression analysis
    └── early_exit_experiments.ipynb             # Early exit layer experiments
```

---

## Key Components

### 1. Main Training Script: `grpo_early_exit/grpo_early_exit.py`

This is the **primary entry point** for training the GRPO agent with early exit support.

**Key Features:**
- **GRPO Algorithm**: Group Relative Policy Optimization (no value function, uses group comparisons)
- **Early Exit Support**: Dynamically selects layers 9-16 based on model confidence (entropy)
- **Temperature-Aware Rewards**: Adapts behavior based on thermal headroom
- **Real Hardware Integration**: Uses actual Jetson sensors for power, temperature, and latency
- **Comprehensive Plotting**: Generates plots for entropy, latency, layers, temperature, rewards, and policy metrics

**Main Functions:**
- `train_with_grpo()`: Train from scratch with offline pre-training
- `train_with_grpo_from_checkpoint()`: Continue training from a saved checkpoint
- `calculate_reward()`: Unified reward function with exponential penalties
- `_plot_training_metrics()`: Plot training metrics over time
- `_plot_policy_metrics()`: Plot policy loss, KL divergence, and entropy

**Action Space:**
- Prefill frequency: 11 bins (0-10)
- Decode frequency: 11 bins (0-10)
- Batch size: 6 options [1, 7, 13, 19, 25, 31]
- Early exit layers: 8 options [9, 10, 11, 12, 13, 14, 15, 16]
- **Total: 11 × 11 × 6 × 8 = 5,808 actions**

**State Space:**
- Prefill frequency (normalized)
- Decode frequency (normalized)
- Batch size (normalized)
- Temperature headroom (normalized)
- Previous layer (normalized, for hysteresis)

### 2. RL Agent: `grpo_early_exit/rl_agent.py`

Contains the `Llama318BRunner` class that interfaces with the actual LLM model.

**Key Features:**
- Loads Llama-3.2-1B-Instruct model
- Manages GPU frequency settings
- Performs inference with early exit support
- Measures power consumption via `tegrastats`
- Tracks entropy from model outputs
- Handles Natural Questions (NQ) dataset for evaluation

### 3. Synthetic Data Generation: `grpo_early_exit/add_synthetic_entropy_to_csv.py`

Generates synthetic entropy and early exit layer values for CSV files when real data is unavailable.

**Usage:**
```bash
cd grpo_early_exit
python add_synthetic_entropy_to_csv.py
```

This script automatically:
- Updates both `fan_on` and `fan_off` CSV files in the `logs/` folder
- Generates entropy in range [0.5, 9.5] based on:
  - **Layer count**: 16 layers → low entropy (0-1.5), 4-6 layers → high entropy (7-9)
  - **Temperature**: Below 70°C → prioritize 16 layers, above 80°C → prioritize 4 layers
  - **Latency**: Higher latency → slightly higher entropy
- Creates or updates `early_exit_layer` and `avg_entropy` columns in-place

### 4. Fan Control: `grpo_early_exit/fanonoff.py`

Utilities for controlling the Jetson fan during training.

**Functions:**
- `fan_on()`: Turn fan on (128 PWM value)
- `fan_off()`: Turn fan off (0 PWM value)
- `initialize_fan()`: Initialize fan state

Used during training to create thermal stress scenarios and test adaptive behavior.

### 5. Overhead Measurement: `grpo/measure_grpo_overhead.py`

Measures the computational overhead of the GRPO agent compared to baseline inference.

**Metrics:**
- Time overhead (action selection, group sampling, policy updates)
- Sample efficiency (evaluations needed to find good solutions)
- Power consumption and energy efficiency
- Wall-clock time comparison

### 6. Test Agent: `test_agent.py`

Test script for evaluating trained GRPO checkpoints on real hardware.

**Key Features:**
- **Baseline Testing**: `test_baseline()` - Run fixed-parameter inference for comparison
- **GRPO Testing**: `test_grpo_checkpoint()` - Load and test a trained GRPO policy checkpoint
- **Temperature Monitoring**: Background thread monitors temperature via `tegrastats`
- **CSV Logging**: Saves detailed inference metrics to CSV files
- **Fan Control**: Can turn fan on/off at specific steps to test thermal adaptation

**Usage:**
```python
# Test a GRPO checkpoint
from test_agent import test_grpo_checkpoint

test_grpo_checkpoint(
    checkpoint_path='path/to/checkpoint.pth',
    num_requests=350,
    fan_off_at_step=50  # Turn fan off at step 50 to test thermal adaptation
)
```

**Output:**
- `grpo_checkpoint_test/GRPO_TEST_YYYYMMDD_HHMMSS.csv`: Detailed inference metrics
- `grpo_checkpoint_test/GRPO_TEST_TEMPERATURES_YYYYMMDD_HHMMSS.csv`: Temperature logs

---

## CSV Files: Data and Results

### Training Data CSV Files

**Location**: `grpo_early_exit/logs/`

1. **`grid_search_results_with_entropy_and_fan_on.csv`**
   - Grid search results with fan ON
   - Contains: prefill_freq, decode_freq, batch_size, early_exit_layer, avg_entropy, latency, power, temperature, etc.
   - Used for offline pre-training

2. **`grid_search_results_with_entropy_and_fan_off.csv`**
   - Grid search results with fan OFF
   - Same structure as above, but with different thermal conditions
   - Used for offline pre-training with thermal stress scenarios

**Purpose**: These CSV files contain pre-collected data from exhaustive grid searches. They are used to:
- Pre-train the GRPO policy before online training
- Initialize normalization ranges (throughput, EDP efficiency, energy)
- Provide baseline statistics for reward function calibration

### Results CSV Files

**Location**: `grpo/logs/`

1. **`grid_search_results_20251128_075945.csv`**
   - Original grid search data (fan ON)
   - Used for plotting and analysis (see `FILES_FOR_PLOT.txt`)

2. **`fan_off_grid_search_results_20251129_092010.csv`**
   - Original grid search data (fan OFF)
   - Used for plotting and analysis

3. **`grpo_training_results_20251130_082847_temp.csv`**
   - Online training results
   - Contains episode-by-episode metrics: rewards, energies, latencies, temperatures, violations
   - Used for plotting training progress

4. **`grpo_deploy_evaluation_20251201_013907.csv`**
   - Deployment evaluation results
   - Final performance metrics after training
   - Used for final analysis and comparison

5. **`grpo_overhead_log_20251202_224709.txt`**
   - GRPO overhead measurement log
   - Contains timing and efficiency measurements
   - Used for overhead analysis

### Training Log CSV Files

**Location**: `grpo_early_exit/logs/`

1. **`early_exit_continued_training_metrics_episode_400.csv`**
   - Training metrics logged during continued training
   - Contains: step, episode, entropy, latency, num_layers, temperature, reward, entropy_penalty, max_temp
   - Can be replotted using `replot_training_metrics.py`

2. **`policy_metrics_continued_episode_400.csv`**
   - Policy training metrics (loss, KL divergence, entropy)
   - Contains: update_step, policy_loss, kl_divergence, policy_entropy
   - Used for analyzing policy convergence

**CSV File Importance:**
- **Training Data**: Essential for offline pre-training (faster convergence, better initialization)
- **Results**: Critical for analysis, plotting, and paper figures
- **Reproducibility**: Allows others to reproduce results without re-running experiments
- **Debugging**: Helps identify issues in reward function, normalization, or training dynamics

---

## Model Checkpoints (.pth Files)

### Location: `grpo/grpo_model_FINAL.pth`

This is a **PyTorch model checkpoint** containing the trained GRPO policy network weights.

**Contents:**
- Policy network state dict (neural network weights)
- Architecture: Multi-layer perceptron (MLP) with configurable layers
- Input: 5D state vector (prefill_freq, decode_freq, batch_size, temperature_headroom, previous_layer)
- Output: Action probabilities over 5,808 possible actions

**Usage:**
```python
# Load checkpoint for continued training
from grpo_early_exit.grpo_early_exit import train_with_grpo_from_checkpoint

train_with_grpo_from_checkpoint(
    checkpoint_path='grpo/grpo_model_FINAL.pth',
    csv_path='grpo_early_exit/logs/grid_search_results_with_entropy_and_fan_on.csv',
    num_episodes=400,
    max_steps=1,
    ...
)
```

**Checkpoint Importance:**
- **Resume Training**: Continue training from a saved point (useful for long training runs)
- **Transfer Learning**: Fine-tune on new data or different constraints
- **Deployment**: Load trained model for inference without retraining
- **Reproducibility**: Share trained models for evaluation and comparison
- **Experimentation**: Test different reward functions or hyperparameters from the same starting point

**Note**: The checkpoint file is large (~several MB) and contains the learned policy. Without it, you must train from scratch, which can take hours on a Jetson device.

---

## How to Run

### Prerequisites

1. **Hardware**: NVIDIA Jetson device (tested on Jetson AGX Orin)
2. **Software**:
   - Python 3.8+
   - PyTorch (with CUDA support)
   - Transformers library (Hugging Face)
   - pandas, numpy, matplotlib
   - Access to `tegrastats` for power monitoring
   - `nq_subset` dataset directory (Natural Questions dataset)

3. **Dataset**: Place the `nq_subset` dataset in a location accessible by the script (checked in parent directory, current directory, or script directory)

### Step 1: Generate Synthetic Data (Optional)

If your CSV files don't have `avg_entropy` and `early_exit_layer` columns:

```bash
cd grpo_early_exit
python add_synthetic_entropy_to_csv.py
```

This automatically updates CSV files in the `logs/` folder.

### Step 2: Train the GRPO Agent

**Option A: Train from scratch**
```bash
cd grpo_early_exit
python grpo_early_exit.py
```

**Option B: Continue from checkpoint**
```python
# Edit grpo_early_exit.py main section to use:
train_with_grpo_from_checkpoint(
    checkpoint_path='../grpo/grpo_model_FINAL.pth',
    csv_path='logs/grid_search_results_with_entropy_and_fan_on.csv',
    num_episodes=400,
    max_steps=1,
    update_freq=6,
    group_size=6,
    use_simulated_env=False,  # Use real Jetson
    ...
)
```

### Step 3: Monitor Training

Training will generate:
- **Console output**: Episode progress, rewards, violations, metrics
- **Plots**: Saved to `grpo_early_exit/plots/`
  - `early_exit_training_metrics_episode_X.png`: Training metrics
  - `policy_metrics_episode_X.png`: Policy loss, KL divergence, entropy
- **CSV logs**: Saved to `grpo_early_exit/logs/`
  - Can be replotted using `logs/replot_training_metrics.py`

### Step 4: Test Trained Agent (Optional)

Test a trained checkpoint on real hardware:

```bash
python test_agent.py
```

Or modify the script to test a specific checkpoint:

```python
# In test_agent.py, modify the main section:
test_grpo_checkpoint(
    checkpoint_path='path/to/checkpoint.pth',
    num_requests=350,
    fan_off_at_step=50
)
```

### Step 5: Replot from Logs (Optional)

```bash
cd grpo_early_exit/logs
python replot_training_metrics.py
# Or specify a specific CSV:
python replot_training_metrics.py early_exit_continued_training_metrics_episode_400.csv
```

---

## Key Features and Design Decisions

### 1. Temperature-Aware Reward Function

The reward function adapts based on thermal headroom:
- **High headroom (fan ON, low temp)**: Prioritizes throughput and accuracy (16 layers, low entropy)
- **Low headroom (fan OFF, high temp)**: Prioritizes energy efficiency and early exit (fewer layers, higher entropy allowed)

### 2. Entropy-Based Early Exit

- **Low entropy** (< 2.0): Model is confident → can use fewer layers (early exit)
- **High entropy** (> 7.0): Model is uncertain → should use more layers (full model)
- **Temperature penalty**: Exponential penalty on entropy when temperature headroom is low

### 3. Group Relative Policy Optimization (GRPO)

- **No value function**: Simpler architecture, no critic network
- **Group comparisons**: Samples multiple actions per state, compares rewards within group
- **Relative advantages**: Computes advantages from group statistics, not absolute value estimates
- **Sample efficiency**: Requires more samples per update but simpler training

### 4. Unified Reward Function

```
R = H_eff * T_n + (1 - H_eff) * EDP_n - λ_L * f(x_L) - λ_T * f(x_T) - λ_P * f(x_P) - λ_E * penalty_entropy
```

Where:
- `H_eff`: Effective headroom (temperature + power)
- `T_n`: Normalized throughput
- `EDP_n`: Normalized Energy-Delay Product efficiency
- `f(x)`: Exponential penalty function
- `penalty_entropy`: Entropy penalty (temperature-aware)

---

## Experimental Results Files

As noted in `FILES_FOR_PLOT.txt`:

- **Grid search**: `grpo/logs/grid_search_results_20251128_075945.csv` (fan ON), `grpo/logs/fan_off_grid_search_results_20251129_092010.csv` (fan OFF)
- **Online training**: `grpo/logs/grpo_training_results_20251130_082847_temp.csv`
- **GRPO overhead**: `grpo/logs/grpo_overhead_log_20251202_224709.txt`
- **Deployment**: `grpo/logs/grpo_deploy_evaluation_20251201_013907.csv`

These files are used for generating plots and analysis in the paper/presentation. The plots are saved in `grpo/Plots/` directory.

---

## Troubleshooting

### Common Issues

1. **"Directory nq_subset not found"**
   - Ensure the Natural Questions dataset is in the expected location
   - Check `rl_agent.py` for path resolution logic

2. **"Runner not available"**
   - Model loading failed (check GPU memory, CUDA availability)
   - Dataset not found (see above)

3. **Training is slow**
   - Reduce `num_episodes`, `max_steps`, or `group_size`
   - Use `use_simulated_env=True` for faster testing (uses CSV data instead of real inference)

4. **Plots not generating**
   - Check that `plots/` directory exists and is writable
   - Ensure matplotlib backend is configured correctly

---

## Citation and Contact

For questions or issues, please contact the team members listed at the top of this README.

---

## License

This code is provided as-is for research and educational purposes. Please ensure compliance with NVIDIA Jetson SDK license and Hugging Face model licenses when using this code.
