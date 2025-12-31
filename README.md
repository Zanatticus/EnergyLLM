# EnergyLLM: Dynamic LLM Edge Inference with GRPO

A reinforcement learning system for dynamically optimizing Large Language Model (LLM) inference on edge devices using Group Relative Policy Optimization (GRPO). The system adaptively controls GPU frequency, batch size, early exit layers, and fan control to balance throughput, energy efficiency, latency, and temperature constraints.

**Team:** Haebin Do, Angelica Kim, Kevin He, Alexander Ingare

> **⚠️ DISCLAIMER**: This code requires an NVIDIA Jetson device to run. The system uses hardware-specific sensors and controls (GPU frequency, fan control, power monitoring via tegrastats) that are only available on Jetson platforms.

---

## Repository Structure

```
EnergyLLM/
├── README.md                               # This file
├── LICENSE                                 # MIT License
├── Code/                                   # Complete GRPO implementation
│   ├── grpo_early_exit/                    # Main GRPO with early exit support
│   │   ├── grpo_early_exit.py              # Main training script
│   │   ├── rl_agent.py                     # RL agent with Llama-3.2-1B runner
│   │   ├── fanonoff.py                     # Fan control utilities
│   │   ├── add_synthetic_entropy_to_csv.py # Synthetic data generation
│   │   ├── logs/                           # Training logs and CSV data
│   │   └── plots/                          # Generated training plots
│   ├── grpo/                               # Original GRPO implementation
│   │   ├── grpo_model_FINAL.pth            # Pre-trained model checkpoint
│   │   ├── measure_grpo_overhead.py        # Overhead measurement script
│   │   ├── rl_agent0.py                    # Base RL agent
│   │   ├── rl_agent0_init_grpo_weight.py   # GRPO agent with initialization
│   │   ├── pruning_experiments.ipynb       # Model pruning experiments
│   │   ├── logs/                           # Results and data CSV files
│   │   └── Plots/                          # Analysis plots
│   ├── model_compression_experiments/      # Model compression analysis
│   │   └── early_exit_experiments.ipynb    # Early exit layer experiments
│   ├── test_agent.py                       # Evaluation script for checkpoints
│   └── README.md                           # Detailed project documentation
├── Final Report.pdf                        # Complete project report
└── Final Project Presentation.pdf          # Final presentation slides
```

---

## Project Overview

EnergyLLM implements a **Group Relative Policy Optimization (GRPO)** reinforcement learning system that dynamically optimizes LLM inference on edge devices. The system learns to adaptively control multiple hardware and model parameters to optimize performance under thermal and energy constraints.

### Key Features

- **GRPO Algorithm** - Group Relative Policy Optimization for adaptive control (no value function, uses group comparisons)
- **Early Exit** - Entropy-based layer selection (9-16 layers) for efficiency
- **Temperature-Aware** - Adapts behavior based on thermal headroom
- **Hardware Integration** - Real-time control on NVIDIA Jetson devices
- **Multi-Objective Optimization** - Balances throughput, energy, latency, and temperature

### Controlled Parameters

- **GPU Frequency** - Prefill and decode phases (11 bins each: 0-10)
- **Batch Size** - 6 options: [1, 7, 13, 19, 25, 31]
- **Early Exit Layers** - 8 options: [9, 10, 11, 12, 13, 14, 15, 16]
- **Fan Control** - Thermal management

**Total Action Space:** 11 × 11 × 6 × 8 = **5,808 actions**

---

## Key Topics

- Reinforcement learning for system optimization
- Edge computing and energy-efficient inference
- Dynamic voltage and frequency scaling (DVFS)
- Early exit strategies for neural networks
- Thermal management in edge devices
- Group Relative Policy Optimization (GRPO)

---

## Hardware Requirements

- **NVIDIA Jetson device** (tested on Jetson AGX Orin)
- Hardware-specific sensors and controls:
  - GPU frequency control
  - Fan control
  - Power monitoring via `tegrastats`
- Access to `nq_subset` dataset (Natural Questions dataset)

---

## Getting Started

### Prerequisites

1. **Hardware**: NVIDIA Jetson device (tested on Jetson AGX Orin)
2. **Software**:
   - Python 3.8+
   - PyTorch (with CUDA support)
   - Transformers library (Hugging Face)
   - pandas, numpy, matplotlib
   - Access to `tegrastats` for power monitoring
   - `nq_subset` dataset directory

### Quick Start

1. **Navigate to the code directory:**
   ```bash
   cd Code
   ```

2. **Train the GRPO agent:**
   ```bash
   cd grpo_early_exit
   python grpo_early_exit.py
   ```

3. **Test a trained checkpoint:**
   ```bash
   cd ..
   python test_agent.py
   ```

### Detailed Instructions

For comprehensive setup instructions, usage examples, and troubleshooting, see:
- **`Code/README.md`** - Complete project documentation with:
  - Detailed component descriptions
  - Training procedures
  - Evaluation methods
  - CSV file documentation
  - Troubleshooting guide

---

## Components

### Main Training Script
- **`Code/grpo_early_exit/grpo_early_exit.py`** - Primary entry point for training GRPO agent with early exit support

### RL Agent
- **`Code/grpo_early_exit/rl_agent.py`** - Interfaces with Llama-3.2-1B model, manages GPU frequency, performs inference with early exit

### Evaluation
- **`Code/test_agent.py`** - Test script for evaluating trained GRPO checkpoints on real hardware

### Analysis Tools
- **`Code/grpo/measure_grpo_overhead.py`** - Measures computational overhead of GRPO agent
- **`Code/grpo/pruning_experiments.ipynb`** - Model pruning experiments
- **`Code/model_compression_experiments/early_exit_experiments.ipynb`** - Early exit layer analysis

---

## Documentation

- **`Code/README.md`** - Comprehensive technical documentation
- **`Final Report.pdf`** - Complete project report with methodology, results, and analysis
- **`Final Project Presentation.pdf`** - Presentation slides

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```
EnergyLLM: Dynamic LLM Edge Inference with GRPO
Team: Haebin Do, Angelica Kim, Kevin He, Alexander Ingare
```

---

## Contact

For questions or issues, please refer to the detailed documentation in `Code/README.md` or contact the team members listed above.
