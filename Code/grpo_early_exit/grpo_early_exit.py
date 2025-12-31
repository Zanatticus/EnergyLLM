import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import os
import subprocess
import re
import time
import multiprocessing
import csv
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from rl_agent import Llama318BRunner, set_gpu_freq_bin, set_gpu_frequency, get_gpu_frequency, measure_power_tegrastats
from fanonoff import fan_on, fan_off, initialize_fan

GPU_FREQ_BINS = [
    306000000, 408000000, 510000000, 612000000, 714000000,
    816000000, 918000000, 1020000000, 1122000000, 1224000000, 1300500000
]

# Global power constraint (Watts)
MAX_POWER = 60.0

# ============================================================
# SHARED REWARD CALCULATION FUNCTION
# ============================================================

def calculate_reward(throughput_raw, edp_efficiency_raw, power_headroom,
                     end_to_end_latency, avg_power, avg_temp,
                     slo_target, max_temp, max_power,
                     throughput_min=None, throughput_max=None,
                     edp_efficiency_min=None, edp_efficiency_max=None,
                     energy_total=None, energy_min = None, energy_max = None,
                     temp_min=40.0, power_min=10.0,
                     slo_max_over=None, temp_max_over=None, power_max_over=None,
                     w_temp=1.0, w_power=0.0,
                     k_penalty=5.0, lambda_latency=1.0, lambda_temp=5.0, lambda_power=1.0,
                     entropy=None, entropy_max=9.0, lambda_entropy=1.5, num_layers_used=None, max_layers=16,
                     curr_prefill=None, curr_decode=None, curr_batch=None, curr_layer=None,
                     prev_prefill=None, prev_decode=None, prev_batch=None, prev_layer=None,
                     lambda_action_change=0.0, freq_bins=11, max_batch_size=32):  # Added hysteresis parameters
                      
    """
    Calculate unified reward function with exponential penalties and hysteresis.
    
    Unified reward: R = H_eff * T_n + (1 - H_eff) * EDP_n - λ_L * f(x_L) - λ_T * f(x_T) - λ_P * f(x_P) - λ_AC * penalty_AC
    
    Where:
    - H_eff = w_T * H_T + w_P * H_P (effective headroom for throughput vs EDP weighting)
    - H_T = clip((T_max - T) / (T_max - T_min), 0, 1) (temperature headroom)
    - H_P = clip((P_max - P) / (P_max - P_min), 0, 1) (power headroom)
    - x_L = max(0, (latency - SLO) / (SLO_max_over - SLO)) (latency overage)
    - x_T = max(0, (T - T_max) / (T_max_over - T_max)) (temperature overage)
    - x_P = max(0, (P - P_max) / (P_max_over - P_max)) (power overage)
    - penalty_AC = normalized action change magnitude (0 to 1)
    - f(x) = (e^(kx) - 1) / (e^k - 1) (exponential penalty function)
    
    This is the shared reward function used by both OfflineDataset and RealJetsonEnv.
    
    Args:
        throughput_raw: Raw throughput value (tokens_per_second = total_tokens / total_time)
        edp_efficiency_raw: Raw EDP efficiency (1.0 / (energy_total * end_to_end_latency))
            where end_to_end_latency is per-request latency (total time), not per-token latency
        power_headroom: Power headroom ratio (0 to 1, where 1 = lots of headroom) - DEPRECATED, calculated internally
        end_to_end_latency: End-to-end latency in seconds
        avg_power: Average power consumption in Watts
        avg_temp: Average temperature in Celsius
        slo_target: SLO target for end-to-end latency
        max_temp: Maximum temperature constraint
        max_power: Maximum power constraint
        throughput_min, throughput_max: Normalization ranges for throughput (optional)
        edp_efficiency_min, edp_efficiency_max: Normalization ranges for EDP efficiency (optional)
        energy_total: Total energy in Joules
        energy_min, energy_max: Normalization ranges for energy (optional)
        temp_min: Minimum temperature for headroom calculation (default: 40.0°C)
        power_min: Minimum power for headroom calculation (default: 10.0W)
        slo_max_over: Maximum SLO overage for normalization (default: 1.5 * slo_target)
        temp_max_over: Maximum temperature overage for normalization (default: 1.2 * max_temp)
        power_max_over: Maximum power overage for normalization (default: 1.3 * max_power)
        w_temp: Weight for temperature headroom in H_eff (default: 1.0)
        w_power: Weight for power headroom in H_eff (default: 0.0)
        k_penalty: Exponential penalty steepness parameter (default: 5.0)
        lambda_latency: Penalty coefficient for latency violations (default: 1.0)
        lambda_temp: Penalty coefficient for temperature violations (default: 1.5)
        lambda_power: Penalty coefficient for power violations (default: 1.0)
        entropy: Average entropy from model outputs (higher = less confident, worse accuracy) (default: None)
        entropy_max: Maximum acceptable entropy threshold (default: 10.0)
        lambda_entropy: Penalty coefficient for entropy violations (default: 1.5)
        num_layers_used: Number of early exit layers used in current step (default: None)
        max_layers: Maximum number of layers in model (default: 16)
        prev_prefill: Previous prefill frequency bin (for hysteresis penalty, default: None)
        prev_decode: Previous decode frequency bin (for hysteresis penalty, default: None)
        prev_batch: Previous batch size (for hysteresis penalty, default: None)
        prev_layer: Previous early exit layer (for hysteresis penalty, default: None)
        lambda_action_change: Penalty coefficient for action changes (default: 0.1)
        freq_bins: Number of frequency bins for normalization (default: 11)
        max_batch_size: Maximum batch size for normalization (default: 32)
    
    Returns:
        reward: Calculated reward value
        performance_metric: Base performance metric (H_eff * T_n + (1 - H_eff) * EDP_n)
        constraint_penalty: Total constraint penalty (λ_L * f(x_L) + λ_T * f(x_T) + λ_P * f(x_P) + λ_AC)
        throughput_normalized: Normalized throughput [0, 1]
        edp_efficiency_normalized: Normalized EDP efficiency [0, 1]
        inverse_energy_normalized: Normalized inverse energy [0, 1]
        headroom_threshold_met: Boolean indicating if headroom is below threshold
        penalty_entropy: Entropy penalty component
        penalty_action_change: Action change (hysteresis) penalty component
    """
    
    # Normalize throughput and EDP efficiency to [0, 1] range
    if throughput_min is not None and throughput_max is not None and throughput_max > throughput_min:
        throughput_normalized = (throughput_raw - throughput_min) / (throughput_max - throughput_min)
    else:
        # Rough normalization if ranges not available (assume throughput range 0-200)
        throughput_normalized = min(1.0, max(0.0, throughput_raw / 200.0))
    
    if edp_efficiency_min is not None and edp_efficiency_max is not None and edp_efficiency_max > edp_efficiency_min:
        edp_efficiency_normalized = (edp_efficiency_raw - edp_efficiency_min) / (edp_efficiency_max - edp_efficiency_min)
    else:
        # Rough normalization if ranges not available (assume EDP efficiency range 0-0.001)
        edp_efficiency_normalized = min(1.0, max(0.0, edp_efficiency_raw / 0.001))
    
    # Clip normalized values to [0, 1]
    throughput_normalized_before_clip = throughput_normalized
    edp_normalized_before_clip = edp_efficiency_normalized
    
    throughput_normalized = np.clip(throughput_normalized, 0.0, 1.0)
    edp_efficiency_normalized = np.clip(edp_efficiency_normalized, 0.0, 1.0)
    
    # Log clipping details (debug: set to False to disable)
    print(f"[calculate_reward] Normalization clipping:")
    print(f"  Throughput: {throughput_normalized_before_clip:.4f} → {throughput_normalized:.4f} (clipped: {throughput_normalized_before_clip != throughput_normalized})")
    print(f"  EDP efficiency: {edp_normalized_before_clip:.6f} → {edp_efficiency_normalized:.4f} (clipped: {edp_normalized_before_clip != edp_efficiency_normalized})")
    
    # Calculate headroom (H_T and H_P)
    # H_T = clip((T_max - T) / (T_max - T_min), 0, 1)
    if max_temp > temp_min:
        H_T_linear = np.clip((max_temp - avg_temp) / (max_temp - temp_min), 0.0, 1.0)
        H_T = H_T_linear
    else:
        H_T = 1.0 if avg_temp < max_temp else 0.0
    
    # H_P = clip((P_max - P) / (P_max - P_min), 0, 1)
    H_P = 0.0  # Fixed to 0: power headroom not used, only temperature matters
    
    # Effective headroom: H_eff = w_T * H_T + w_P * H_P
    H_eff = w_temp * H_T + w_power * H_P
    H_eff = np.clip(H_eff, 0.0, 1.0)
    
    # Calculate overages (x_L, x_T, x_P)
    # Set default max_over values if not provided
    if slo_max_over is None:
        slo_max_over = 1.5 * slo_target
    if temp_max_over is None:
        temp_max_over = 1.05 * max_temp
    if power_max_over is None:
        power_max_over = 1.3 * max_power
    
    # x_L = max(0, (latency - SLO) / (SLO_max_over - SLO))
    if slo_max_over > slo_target:
        x_L = max(0.0, (end_to_end_latency - slo_target) / (slo_max_over - slo_target))
    else:
        x_L = 1.0 if end_to_end_latency > slo_target else 0.0
    x_L = np.clip(x_L, 0.0, 1.0)
    
    # x_T = max(0, (T - T_max) / (T_max_over - T_max))
    # avg_temp is already validated above
    if temp_max_over > max_temp:
        x_T = max(0.0, (avg_temp - max_temp) / (temp_max_over - max_temp))
    else:
        x_T = 1.0 if avg_temp > max_temp else 0.0
    x_T = np.clip(x_T, 0.0, 1.0)
    
    # x_P = max(0, (P - P_max) / (P_max_over - P_max))
    if power_max_over > max_power:
        x_P = max(0.0, (avg_power - max_power) / (power_max_over - max_power))
    else:
        x_P = 1.0 if avg_power > max_power else 0.0
    x_P = np.clip(x_P, 0.0, 1.0)
    
    # Exponential penalty function: f(x) = (e^(kx) - 1) / (e^k - 1)
    def exp_penalty(x, k):
        """Exponential penalty function for overage x ∈ [0, 1]."""
        if k <= 0:
            return x  # Linear fallback
        exp_k = np.exp(k)
        if x <= 0:
            return 0.0
        return (np.exp(k * x) - 1.0) / (exp_k - 1.0)
    
    # Calculate penalties using exponential function
    penalty_latency = lambda_latency * exp_penalty(x_L, k_penalty)
    
    # Increase temperature penalty when headroom is low
    TEMP_SOFT_THRESHOLD = 80.0
    if avg_temp > TEMP_SOFT_THRESHOLD:
        soft_temp_penalty = 2.0 * (avg_temp - TEMP_SOFT_THRESHOLD) / (max_temp - TEMP_SOFT_THRESHOLD)
        soft_temp_penalty = min(soft_temp_penalty, 1.0)
    else:
        soft_temp_penalty = 0.0

    # Scale lambda_temp by (1 - H_T) to penalize more when temp headroom is low
    temp_penalty_scale = 1.0 + 2.0 * (1.0 - H_T)  # 1.0x when H_T=1.0, 3.0x when H_T=0.0
    penalty_temp = lambda_temp * temp_penalty_scale * exp_penalty(x_T, k_penalty)
    
    penalty_power = lambda_power * exp_penalty(x_P, k_penalty)
    
    # Unified reward: R = H_eff * T_n + (1 - H_eff) * Efficiency_n - λ_L * f(x_L) - λ_T * f(x_T) - λ_P * f(x_P)
    # 
    # When headroom is low, we want to prioritize power reduction (low batch/frequency).
    # EDP efficiency = 1/(energy × latency) = 1/(power × latency²) includes latency in the denominator,
    # which could work against our goal if latency increases. So we use pure power efficiency instead.
    #
    # Strategy:
    # - When headroom is very low: Use power efficiency (directly rewards low power, ignores latency)
    # - When headroom is moderate/high: Use EDP efficiency (balances energy and latency)
    
    # Calculate power efficiency (inverse of normalized power)
    # This directly rewards low power consumption, independent of latency
    # Power is what we directly control via batch/frequency
    # power_normalized_for_efficiency = np.clip((avg_power - power_min) / (max_power - power_min), 0.0, 1.0)
    # power_efficiency = 1.0 - power_normalized_for_efficiency  # Higher power → lower efficiency
    inverse_energy_before_clip = (energy_max - energy_total) / (energy_max - energy_min)
    inverse_energy_normalized = np.clip(inverse_energy_before_clip, 0.0, 1.0)
    
    # Log energy normalization details (debug: set to False to disable)
    print(f"[calculate_reward] Energy normalization:")
    print(f"  Raw: {energy_total:.6f}, Range: [{energy_min}, {energy_max}], Before clip: {inverse_energy_before_clip:.4f}, After clip: {inverse_energy_normalized:.4f}")
      
    # When headroom is very low, use pure power efficiency (no EDP blending)
    headroom_threshold = 0.33 # not currently used in reward function
    # if H_eff <= headroom_threshold:
    #     # efficiency_metric = inverse_energy_normalized
    #     # performance_metric = inverse_energy_normalized * throughput_normalized
    #     performance_metric = inverse_energy_normalized
    # else:
    #     # efficiency_metric = edp_efficiency_normalized
    #     performance_metric = H_eff * throughput_normalized + (1.0 - H_eff) * edp_efficiency_normalized * throughput_normalized

    # Continuous performance blending based on H_eff
    # energy_weight = (1.0 - H_eff) ** 3      # Cubic decay - energy fades quickly
    # throughput_weight = H_eff ** 1          # Linear - throughput stays strong
    # edp_weight = 0.3 * H_eff * (1.0 - H_eff)  # Reduced mid-range EDP influence

    # total_weight = energy_weight + throughput_weight + edp_weight + 1e-8

    # performance_metric = (
    #     energy_weight * inverse_energy_normalized +
    #     throughput_weight * throughput_normalized +
    #     edp_weight * edp_efficiency_normalized
    # ) / total_weight


    # Sigmoid Blending
    def smooth_blend(H_eff, center=0.33, steepness=10):
        """Returns blend weight from 0 (low headroom) to 1 (high headroom)"""
        return 1 / (1 + np.exp(-steepness * (H_eff - center)))

    # Compute both metrics
    low_headroom_metric = inverse_energy_normalized
    high_headroom_metric = H_eff * throughput_normalized + (1 - H_eff) * edp_efficiency_normalized * throughput_normalized

    # Smooth blend
    alpha = smooth_blend(H_eff)
    performance_metric = alpha * high_headroom_metric + (1 - alpha) * low_headroom_metric
    
    
    # Additional power penalty when temperature headroom is low
    # NOTE: Disabled aggressive energy penalty for simplicity and consistency with CSV reward calculation.
    # We keep the parameter for backward compatibility but do not apply any extra penalty here.
    
    # Entropy penalty: penalize high entropy (low confidence, worse accuracy)
    # Entropy is directly related to accuracy - higher entropy means model is less confident
    # 
    # Strategy: Use exponential penalty on temperature headroom to control entropy
    # - High headroom (H_T close to 1.0, fan on, low temp): Strong penalty on entropy → use more layers (16) → lower entropy
    # - Low headroom (H_T close to 0.0, approaching thermal limits): Weak penalty on entropy → allow early exit → higher entropy OK
    #
    # The exponential penalty on headroom ensures:
    # - When headroom is high: exp_penalty(1.0 - H_T) is small, but we multiply by exp_penalty(H_T) which is large → strong penalty
    # - When headroom is low: exp_penalty(1.0 - H_T) is large, but we multiply by exp_penalty(H_T) which is small → weak penalty
    #
    # Note: Latency is not used here because it stays stable until temperature reaches a high threshold.
    # The SLO target is loose, so latency violations don't impact policy updates.
    # Temperature is the primary driver for early exit decisions.
    # penalty_entropy = 0.0
    # if entropy is not None and entropy_max > 0:
    #     x_E = np.clip(entropy / entropy_max, 0.0, 1.0)
    #     entropy_violation_penalty = exp_penalty(x_E, k_penalty)
        
    #     if H_T < 0.3:  # Hot (>82°C) - allow early exit
    #         penalty_scale = 0.2 + 0.5 * H_T  # Range 0.2-0.35 (soft)
    #     else:  # Cool - force 16 layers
    #         penalty_scale = 20.0  # Very strict, basically forces low entropy
        
    #     penalty_entropy = lambda_entropy * entropy_violation_penalty * penalty_scale



    # Action change (hysteresis) penalty: penalize large changes from previous action
    # This encourages the policy to make small incremental adjustments rather than jumping around
    # penalty_action_change = 0.0
    # if prev_prefill is not None or prev_decode is not None or prev_batch is not None or prev_layer is not None:
    #     # Calculate normalized action change magnitude
    #     # Strategy: Use MAX norm to penalize ANY large change, not average
    #     # This ensures that changing multiple parameters is penalized more
    #     component_changes = []
        
    #     # Frequency changes (prefill and decode): normalize by freq_bins - 1
    #     if prev_prefill is not None and curr_prefill is not None and freq_bins > 1:
    #         freq_change_norm = abs(int(curr_prefill) - int(prev_prefill)) / (freq_bins - 1)
    #         component_changes.append(freq_change_norm)
        
    #     if prev_decode is not None and curr_    penalty_entropy = 0.0


    # penalty_entropy = 0.0

    # if entropy is not None and entropy_max > 0:
    #     x_E = np.clip(entropy / entropy_max, 0.0, 1.0)
        
    #     # Normalized: 0 = 9 layers, 1 = 16 layers
    #     layer_norm = (num_layers_used - 9) / 7
        
    #     COOL_THRESHOLD = 0.33
        
    #     if H_T >= COOL_THRESHOLD:
    #         # COOL: Enforce 16 layers via entropy penalty
    #         penalty_scale = 5.0
    #         layer_bonus = 0.0
    #     else:
    #         # HOT: Reduce penalty AND reward fewer layers
    #         hot_intensity = 1.0 - (H_T / COOL_THRESHOLD)
    #         penalty_scale = 5.0 * np.exp(-4.0 * hot_intensity)
            
    #         # KEY: Bonus for using fewer layers when hot
    #         # Max bonus of 2.0 for 9 layers at H_T=0
    #         layer_bonus = 2.0 * hot_intensity * (1.0 - layer_norm)
        
    #     entropy_penalty = lambda_entropy * penalty_scale * (np.exp(2.0 * x_E) - 1.0)
    #     penalty_entropy = entropy_penalty - layer_bonus

    penalty_entropy = 0.0
    layer_efficiency_reward = 0.0

    if entropy is not None and entropy_max > 0:
        x_E = np.clip(entropy / entropy_max, 0.0, 1.0)
        layer_norm = (num_layers_used - 9) / 7
        
        COOL_THRESHOLD = 0.33
        
        if H_T >= COOL_THRESHOLD:
            # COOL: Strong penalty to enforce 16 layers
            penalty_scale = 2.5
        else:
            # HOT: Reduce penalty exponentially
            hot_intensity = 1.0 - (H_T / COOL_THRESHOLD)
            penalty_scale = 2.5 * np.exp(-2.0 * hot_intensity)
            
            # Reward using fewer layers when hot (separate term)
            layer_efficiency_reward = 5 * hot_intensity * (1.0 - layer_norm)
            print(f"Layer efficiency reward: H_T={H_T:.3f}, hot_intensity={hot_intensity:.3f}, layer_norm={layer_norm:.3f}, reward={layer_efficiency_reward:.3f}")
        
        # Entropy penalty (always positive, just smaller when hot)
        penalty_entropy = lambda_entropy * penalty_scale * (np.exp(2.0 * x_E) - 1.0) - layer_efficiency_reward
        print("Entropy penalty:", lambda_entropy * penalty_scale * (np.exp(2.0 * x_E) - 1.0))
        print("Entropy penalty - layer_reward :", penalty_entropy)
    #         freq_change_norm = abs(int(curr_decode) - int(prev_decode)) / (freq_bins - 1)
    #         component_changes.append(freq_change_norm)
        
    #     # Batch size change: normalize by max_batch_size - 1
    #     if prev_batch is not None and curr_batch is not None and max_batch_size > 1:
    #         batch_change_norm = abs(int(curr_batch) - int(prev_batch)) / (max_batch_size - 1)
    #         component_changes.append(batch_change_norm)
        
    #     # Layer change: normalize by max_layers - 9 (valid range is 9 to 16)
    #     if prev_layer is not None and curr_layer is not None and max_layers > 9:
    #         layer_change_norm = abs(int(curr_layer) - int(prev_layer)) / (max_layers - 9)
    #         component_changes.append(layer_change_norm)
        
    #     # Use SUM of normalized changes (not average)
    #     # This way: 1 component changing by max = 1.0 penalty input
    #     #          4 components changing by 0.25 each = 1.0 penalty input (same)
    #     #          4 components changing by max = 4.0 penalty input (much worse, as intended)
    #     if len(component_changes) > 0:
    #         change_magnitude = sum(component_changes)
    #         # Clip to [0, 1] for consistent penalty scaling
    #         # The exponential penalty will still work well with values > 1
    #         change_magnitude = np.clip(change_magnitude, 0.0, 1.0)
            
    #         # Apply exponential penalty to action change magnitude
    #         # Higher change → stronger penalty
    #         penalty_action_change = lambda_action_change * exp_penalty(change_magnitude, k_penalty)
    #         print(f"Action change penalty: components={[f'{c:.3f}' for c in component_changes]}, sum={sum(component_changes):.3f}, clipped_magnitude={change_magnitude:.3f}, penalty={penalty_action_change:.3f}")
    penalty_action_change = 0.0  # Disabled for simplicity
    constraint_penalty = penalty_latency + penalty_temp + penalty_power + penalty_entropy + penalty_action_change
    reward = performance_metric - constraint_penalty
    
    return reward, performance_metric, constraint_penalty, throughput_normalized, edp_efficiency_normalized, inverse_energy_normalized, (H_eff <= headroom_threshold), penalty_entropy, penalty_action_change


# ============================================================
# OFFLINE DATASET FOR PRE-TRAINING (DUAL FREQUENCY)
# ============================================================

class OfflineDataset(Dataset):
    """
    Dataset for offline pre-training with unified reward function.
    """
    
    def __init__(self, csv_path, freq_bins=11, max_batch=1, alpha=0.8, 
                 slo_target=15.0, max_temp=95.0, max_power=MAX_POWER):
        """
        Args:
            csv_path: Path to CSV data, or list/tuple of CSV paths to combine
            freq_bins: Number of frequency bins
            max_batch: Maximum batch size
            alpha: Legacy parameter (kept for compatibility)
            slo_target: Maximum end-to-end latency in seconds
            max_temp: Maximum temperature in Celsius
            max_power: Maximum power in Watts
        """
        self.freq_bins = freq_bins
        self.max_batch = max_batch
        self.alpha = alpha
        self.slo_target = slo_target
        self.max_temp = max_temp
        self.max_power = max_power
        print(f"Max power constraint: {self.max_power}W")
        
        # Load CSV(s) - support single file or list of files
        if isinstance(csv_path, (list, tuple)):
            # Multiple CSV files - combine them
            dfs = []
            for path in csv_path:
                df = pd.read_csv(path)
                # Add source file identifier
                df['_source_file'] = path
                # Check if avg_entropy column exists
                if 'avg_entropy' not in df.columns:
                    print(f"WARNING: CSV file {path} does not contain 'avg_entropy' column. Adding default values (None).")
                    df['avg_entropy'] = None
                dfs.append(df)
                print(f"Loaded {len(df)} rows from {path}")
            self.df = pd.concat(dfs, ignore_index=True)
            print(f"Combined total: {len(self.df)} rows from {len(csv_path)} CSV files")
            self.csv_path = csv_path  # Store list of paths
        else:
            # Single CSV file
            self.df = pd.read_csv(csv_path)
            self.df['_source_file'] = csv_path
            # Check if avg_entropy column exists
            if 'avg_entropy' not in self.df.columns:
                print(f"WARNING: CSV file {csv_path} does not contain 'avg_entropy' column. Adding default values (None).")
                self.df['avg_entropy'] = None
            print(f"Loaded {len(self.df)} rows from {csv_path}")
            self.csv_path = csv_path
        
        # Verify avg_entropy column exists
        if 'avg_entropy' not in self.df.columns:
            print("ERROR: 'avg_entropy' column is missing from CSV. Adding column with None values.")
            self.df['avg_entropy'] = None
        else:
            # Check how many rows have valid entropy values
            valid_entropy_count = self.df['avg_entropy'].notna().sum()
            total_rows = len(self.df)
            print(f"CSV entropy data: {valid_entropy_count}/{total_rows} rows have valid entropy values")
        
        self.GPU_FREQ_BINS = GPU_FREQ_BINS
        self.freq_to_bin = {freq: i for i, freq in enumerate(self.GPU_FREQ_BINS)}
        
        self.batch_sizes = list(range(1, 32, 6))  # Steps of 6: [1, 7, 13, 19, 25, 31]
        self.num_batch_sizes = len(self.batch_sizes)
        
        self.actions = []
        for prefill_freq in range(0, freq_bins):
            for decode_freq in range(0, freq_bins):
                for batch in self.batch_sizes:
                    for early_exit_layer in range(9, 17, 1):
                        self.actions.append((prefill_freq, decode_freq, batch, early_exit_layer))
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        self.action_dim = len(self.actions)
        print(f"Action space: {freq_bins} x {freq_bins} x {self.num_batch_sizes}(batch sizes: {self.batch_sizes}) x 8 (early exit layers: 9, 10, 11, 12, 13, 14, 15, 16) = {self.action_dim} actions")
        
        raw_data = self._process_data_raw()
        print(f"Collected {len(raw_data)} raw data points")
        
        if raw_data:
            throughputs = [d['throughput_raw'] for d in raw_data]
            edp_efficiencies = [d['edp_efficiency_raw'] for d in raw_data]
            energies = [d['energy_total'] for d in raw_data]
            
            self.throughput_min = np.min(throughputs)
            self.throughput_max = np.max(throughputs)
            self.edp_efficiency_min = np.min(edp_efficiencies)
            self.edp_efficiency_max = np.max(edp_efficiencies)
            self.energy_min = np.min(energies)
            self.energy_max = np.max(energies)
            
            print(f"Normalization ranges:")
            print(f"  Throughput: [{self.throughput_min:.4f}, {self.throughput_max:.4f}]")
            print(f"  EDP efficiency: [{self.edp_efficiency_min:.6f}, {self.edp_efficiency_max:.6f}]")
            print(f"  Energy: [{self.energy_min:.6f}, {self.energy_max:.6f}]")
            self.processed_data = self._process_data_normalized(raw_data)
            print(f"Processed {len(self.processed_data)} normalized data points")
            
            rewards = [d['reward'] for d in self.processed_data]
            self.reward_mean = np.mean(rewards)
            self.reward_std = np.std(rewards) + 1e-8
            print(f"Reward stats - Mean: {self.reward_mean:.4f}, Std: {self.reward_std:.4f}")
        else:
            self.reward_mean = 0
            self.reward_std = 1
            self.throughput_min = 0
            self.throughput_max = 1
            self.edp_efficiency_min = 0
            self.edp_efficiency_max = 1
            self.energy_min = 0
            self.energy_max = 1
            self.processed_data = []
    
    def _normalize(self, value, min_val, max_val):
        """Normalize value to [0, 1] range."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def _process_data_raw(self):
        """Process CSV data and collect raw metrics for normalization."""
        processed = []
        avg_power_values = []
        
        for idx, row in self.df.iterrows():
            # Get source file if available
            source_file = row.get('_source_file', 'unknown')
            try:
                prefill_freq_bin = int(row['prefill_freq_bin'])
                decode_freq_bin = int(row['decode_freq_bin'])
            except (KeyError, ValueError) as e:
                print(f"WARNING: Row {idx} has invalid frequency bins: {e}")
                continue
            
            try:
                batch_size = int(row['effective_batch_size'])
            except (KeyError, ValueError) as e:
                print(f"WARNING: Row {idx} has invalid batch size: {e}")
                continue
            
            if batch_size not in self.batch_sizes:
                continue
            
            # Extract early_exit_layer from CSV if available, otherwise use default (16 = full model)
            early_exit_layer = 16  # Default to full model
            if 'early_exit_layer' in row:
                try:
                    early_exit_val = row['early_exit_layer']
                    if pd.notna(early_exit_val) and early_exit_val != '':
                        early_exit_layer = int(early_exit_val)
                        # Ensure it's in valid range [9, 16]
                        if early_exit_layer < 9:
                            early_exit_layer = 9
                        elif early_exit_layer > 16:
                            early_exit_layer = 16
                        # Round to nearest valid early exit layer (9, 10, 11, 12, 13, 14, 15, 16)
                        valid_layers = list(range(9, 17, 1))
                        early_exit_layer = min(valid_layers, key=lambda x: abs(x - early_exit_layer))
                except (ValueError, TypeError):
                    early_exit_layer = 16  # Default to full model
            
            # Action now includes early_exit_layer (4 elements)
            action = (prefill_freq_bin, decode_freq_bin, batch_size, early_exit_layer)
            if action not in self.action_to_idx:
                continue
            action_idx = self.action_to_idx[action]
            
            try:
                latency_prefill = float(row['prefill_time'])
                latency_decode = float(row['decode_time'])
                tokens_prefill = int(row['num_prompt_tokens'])
                tokens_decode = int(row['num_decode_tokens'])
                
                power_prefill = float(row['prefill_TOTAL_POWER'])
                power_decode = float(row['decode_TOTAL_POWER'])
                avg_temp = float(row['temperature'])
            except (KeyError, ValueError) as e:
                print(f"WARNING: Row {idx} has invalid metric values: {e}")
                continue
            
            total_time = latency_prefill + latency_decode
            # Match CSV generation: use simple average, not weighted average
            avg_power = (power_prefill + power_decode) / 2.0
            avg_power_values.append(avg_power)
            
            end_to_end_latency = latency_prefill + latency_decode
            
            # Calculate throughput: decode tokens per second only (matches RealJetsonEnv)
            # Only tracking decode phase efficiency, not total request throughput
            throughput_raw = tokens_decode / (latency_decode + 1e-6) if latency_decode > 0 else 0.0
            
            energy_total = power_prefill * latency_prefill + power_decode * latency_decode
            # Use total tokens for energy efficiency calculation (different from throughput_raw)
            total_tokens = tokens_prefill + tokens_decode
            energy_efficiency = total_tokens / (energy_total + 1e-6) if energy_total > 0 else 0.0
            
            power_headroom = max(0, (self.max_power - avg_power) / self.max_power)
            
            edp = energy_total * end_to_end_latency
            edp_efficiency_raw = 1.0 / (edp + 1e-6)
            
            # Use reward from CSV if available, otherwise will be recalculated in _process_data_normalized
            csv_reward = row.get('reward', None)
            
            # Extract avg_entropy from CSV if available
            avg_entropy_raw = None
            if 'avg_entropy' in row:
                try:
                    entropy_val = row['avg_entropy']
                    if pd.notna(entropy_val) and entropy_val != '':
                        avg_entropy_raw = float(entropy_val)
                except (ValueError, TypeError):
                    avg_entropy_raw = None
            
            processed.append({
                'prefill_freq_bin': prefill_freq_bin,
                'decode_freq_bin': decode_freq_bin,
                'batch_size': batch_size,
                'early_exit_layer': early_exit_layer,  # Store early exit layer
                'action_idx': action_idx,
                'throughput_raw': throughput_raw,
                'edp_efficiency_raw': edp_efficiency_raw,
                'power_headroom': power_headroom,
                'energy_total': energy_total,
                'end_to_end_latency': end_to_end_latency,
                'latency_prefill': latency_prefill,
                'latency_decode': latency_decode,
                'tokens_prefill': tokens_prefill,
                'tokens_decode': tokens_decode,
                'power_prefill': power_prefill,
                'power_decode': power_decode,
                'avg_power': avg_power,
                'avg_temp': avg_temp,
                'avg_entropy': avg_entropy_raw,  # Store entropy from CSV
                'total_tokens': total_tokens,
                'total_time': total_time,
                'energy_efficiency': energy_efficiency,
                'source_file': source_file,  # Track which CSV file this came from
                'csv_reward': csv_reward  # Store CSV reward if available
            })
        
        # Print avg_power statistics
        if avg_power_values:
            print(f"\nAverage power statistics:")
            print(f"  Min avg_power: {np.min(avg_power_values):.2f}W")
            print(f"  Max avg_power: {np.max(avg_power_values):.2f}W")
            print(f"  Mean avg_power: {np.mean(avg_power_values):.2f}W")
            print(f"  Std avg_power: {np.std(avg_power_values):.2f}W")
            print(f"  Max power constraint: {self.max_power}W")
            violations = sum(1 for p in avg_power_values if p > self.max_power)
            print(f"  Power violations: {violations}/{len(avg_power_values)} ({violations/len(avg_power_values)*100:.1f}%)")
        
        return processed
    
    def _process_data_normalized(self, raw_data):
        """Normalize metrics to [0, 1] range and calculate final rewards."""
        processed = []
        reward_list = []
        for raw in raw_data:
            prefill_freq_bin = raw['prefill_freq_bin']
            decode_freq_bin = raw['decode_freq_bin']
            batch_size = raw['batch_size']
            action_idx = raw['action_idx']
            throughput_raw = raw['throughput_raw']
            edp_efficiency_raw = raw['edp_efficiency_raw']
            power_headroom = raw['power_headroom']
            energy_total = raw['energy_total']
            end_to_end_latency = raw['end_to_end_latency']
            avg_power = raw['avg_power']
            avg_temp = raw['avg_temp']
            tokens_prefill = raw['tokens_prefill']
            tokens_decode = raw['tokens_decode']
            latency_prefill = raw['latency_prefill']
            latency_decode = raw['latency_decode']
            power_prefill = raw['power_prefill']
            power_decode = raw['power_decode']
            total_tokens = raw['total_tokens']
            total_time = raw['total_time']
            energy_efficiency = raw['energy_efficiency']
            source_file = raw.get('source_file', 'unknown')  # Preserve source file
            early_exit_layer = raw['early_exit_layer']  # Get early exit layer used
            csv_reward = raw.get('csv_reward', None)  # Get CSV reward if available
            
            # Use CSV reward if available, otherwise recalculate
            if csv_reward is not None and not np.isnan(csv_reward):
                # Use reward directly from CSV (no recalculation)
                reward = float(csv_reward)
                # Still calculate normalized values for other purposes, but use CSV reward
                temp_min = 40.0
                power_min = 10.0
                w_temp = 1
                w_power = 0
                slo_max_over = 1.5 * self.slo_target if self.slo_target > 0 else 15.0
                temp_max_over = 1.2 * self.max_temp
                power_max_over = 1.3 * self.max_power
                
                # Calculate normalized values for compatibility (but reward comes from CSV)
                if self.throughput_min is not None and self.throughput_max is not None and self.throughput_max > self.throughput_min:
                    throughput_normalized = (throughput_raw - self.throughput_min) / (self.throughput_max - self.throughput_min)
                else:
                    throughput_normalized = min(1.0, max(0.0, throughput_raw / 200.0))
                
                if self.edp_efficiency_min is not None and self.edp_efficiency_max is not None and self.edp_efficiency_max > self.edp_efficiency_min:
                    edp_efficiency_normalized = (edp_efficiency_raw - self.edp_efficiency_min) / (self.edp_efficiency_max - self.edp_efficiency_min)
                else:
                    edp_efficiency_normalized = min(1.0, max(0.0, edp_efficiency_raw / 0.001))
                
                if self.energy_min is not None and self.energy_max is not None and self.energy_max > self.energy_min:
                    inverse_energy_normalized = (self.energy_max - energy_total) / (self.energy_max - self.energy_min)
                else:
                    inverse_energy_normalized = 0.5
                
                inverse_energy_normalized = np.clip(inverse_energy_normalized, 0.0, 1.0)
                
                # Calculate headroom for headroom_threshold_met
                if self.max_temp > temp_min:
                    H_T = max(0.0, min(1.0, (self.max_temp - avg_temp) / (self.max_temp - temp_min)))
                else:
                    H_T = 1.0 if avg_temp < self.max_temp else 0.0
                H_eff = w_temp * H_T + w_power * 0.0
                H_eff = max(0.0, min(1.0, H_eff))
                headroom_threshold_met = (H_eff <= 0.33)
                
                performance_metric = 0.0  # Not used when using CSV reward
                constraint_penalty = 0.0  # Not used when using CSV reward
            else:
                # No CSV reward - recalculate using calculate_reward function
                temp_min = 40.0
                power_min = 10.0
                w_temp = 1
                w_power = 0
                
                # Calculate max_over values (matching measure_grpo_overhead.py defaults)
                slo_max_over = 1.5 * self.slo_target if self.slo_target > 0 else 15.0
                temp_max_over = 1.2 * self.max_temp
                power_max_over = 1.3 * self.max_power
                
                if self.max_temp > temp_min:
                    H_T = max(0.0, min(1.0, (self.max_temp - avg_temp) / (self.max_temp - temp_min)))
                else:
                    H_T = 1.0 if avg_temp < self.max_temp else 0.0
                
                if self.max_power > power_min:
                    H_P = max(0.0, min(1.0, (self.max_power - avg_power) / (self.max_power - power_min)))
                else:
                    H_P = 1.0 if avg_power < self.max_power else 0.0
                
                H_eff = w_temp * H_T + w_power * H_P
                H_eff = max(0.0, min(1.0, H_eff))
                
                # Extract entropy from processed data (already extracted in _process_data_raw)
                avg_entropy = d.get('avg_entropy', None)
                if avg_entropy is not None:
                    try:
                        avg_entropy = float(avg_entropy)
                    except (ValueError, TypeError):
                        avg_entropy = None
                
                # Use shared reward calculation function with correct parameters
                # Match parameters used in CSV generation (measure_grpo_overhead.py)
                reward, performance_metric, constraint_penalty, throughput_normalized, \
                edp_efficiency_normalized, inverse_energy_normalized, headroom_threshold_met, penalty_entropy, penalty_action_change = \
                calculate_reward(
                    throughput_raw=throughput_raw,
                    edp_efficiency_raw=edp_efficiency_raw,
                    power_headroom=power_headroom,
                    end_to_end_latency=end_to_end_latency,
                    avg_power=avg_power,
                    avg_temp=avg_temp,
                    slo_target=self.slo_target,
                    max_temp=self.max_temp,
                    max_power=self.max_power,
                    throughput_min=self.throughput_min,
                    throughput_max=self.throughput_max,
                    edp_efficiency_min=self.edp_efficiency_min,
                    edp_efficiency_max=self.edp_efficiency_max,
                    energy_total=energy_total,
                    energy_min=self.energy_min,
                    energy_max=self.energy_max,
                    temp_min=temp_min,
                    power_min=power_min,
                    slo_max_over=slo_max_over,
                    temp_max_over=temp_max_over,
                    power_max_over=power_max_over,
                    w_temp=w_temp,
                    w_power=w_power,
                    k_penalty=5.0,
                    lambda_latency=1.0,
                    lambda_temp=5.0,  # Default value (CSV generation doesn't pass this, so uses default)
                    lambda_power=1.0,
                    entropy=avg_entropy,  # Pass entropy from CSV if available
                    entropy_max=10.0,  # Maximum entropy value in the system (actual max is 10)
                    lambda_entropy=1.5,  # Penalty coefficient for entropy violations (increased for better early exit)
                    num_layers_used=early_exit_layer,  # Pass early exit layer for layer-dependent penalty
                    max_layers=16,  # Maximum layers in the model
                    curr_prefill=prefill_freq_bin,  # Current action: prefill frequency
                    curr_decode=decode_freq_bin,  # Current action: decode frequency
                    curr_batch=batch_size,  # Current action: batch size
                    curr_layer=early_exit_layer  # Current action: early exit layer
                )
            reward_list.append(reward)
            latency_component = throughput_normalized
            energy_efficiency_normalized = energy_efficiency
            energy_component = energy_efficiency_normalized / 20.0
            
            adaptive_alpha_latency = H_eff
            adaptive_alpha_energy = 1.0 - H_eff
            
            energy = energy_total
            latency = end_to_end_latency / total_tokens
            total_power = avg_power
            temperature = avg_temp
            
            batch_idx = self.batch_sizes.index(batch_size)
            temperature_headroom = max(0, (self.max_temp - avg_temp) / (self.max_temp - temp_min))
    
            previous_layer_normalized = (early_exit_layer - 9) / 7.0
            state = np.array([
                prefill_freq_bin / (self.freq_bins - 1),
                decode_freq_bin / (self.freq_bins - 1),
                batch_idx / (self.num_batch_sizes - 1) if self.num_batch_sizes > 1 else 0.0,
                temperature_headroom, 
                previous_layer_normalized
            ], dtype=np.float32)
            
            total_tokens = tokens_prefill + tokens_decode
            total_time = latency_prefill + latency_decode
            throughput_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            processed.append({
                'state': state,
                'action_idx': action_idx,
                'reward': reward,
                'energy': energy,
                'latency': latency,
                'end_to_end_latency': end_to_end_latency,
                'power': total_power,
                'temperature': temperature,
                'power_headroom': power_headroom,
                'temperature_headroom': temperature_headroom,
                'slo_violation': end_to_end_latency > self.slo_target,
                'temp_violation': temperature > self.max_temp,
                'power_violation': total_power > self.max_power,
                'latency_component': latency_component,
                'energy_component': energy_component,
                'adaptive_alpha_energy': adaptive_alpha_energy,
                'adaptive_alpha_latency': adaptive_alpha_latency,
                'throughput_raw': throughput_raw,
                'throughput_normalized': throughput_normalized,
                'throughput_tokens_per_sec': throughput_tokens_per_sec,
                'tokens_prefill': tokens_prefill,
                'tokens_decode': tokens_decode,
                'latency_prefill': latency_prefill,
                'latency_decode': latency_decode,
                'power_prefill': power_prefill,
                'power_decode': power_decode,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'prefill_freq_bin': prefill_freq_bin,
                'decode_freq_bin': decode_freq_bin,
                'batch_size': batch_size,
                'source_file': source_file  # Preserve source file for tracking
            })
        print(min(reward_list), max(reward_list))
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        return {
            'state': torch.FloatTensor(data['state']),
            'action_idx': torch.LongTensor([data['action_idx']]),
            'reward': torch.FloatTensor([data['reward']]),
        }
    
    def get_best_actions(self, top_k=10):
        """Get the top-k best actions based on reward."""
        sorted_data = sorted(self.processed_data, key=lambda x: x['reward'], reverse=True)
        
        # Count constraint violations
        total = len(self.processed_data)
        slo_viol = sum(1 for d in self.processed_data if d['slo_violation'])
        temp_viol = sum(1 for d in self.processed_data if d['temp_violation'])
        power_viol = sum(1 for d in self.processed_data if d['power_violation'])
        
        print(f"\nDataset Constraint Analysis:")
        print(f"  Total samples: {total}")
        print(f"  SLO violations: {slo_viol} ({slo_viol/total*100:.1f}%)")
        print(f"  Temp violations: {temp_viol} ({temp_viol/total*100:.1f}%)")
        print(f"  Power violations: {power_viol} ({power_viol/total*100:.1f}%)")
        
        # Find actions that meet ALL constraints
        valid_actions = [d for d in self.processed_data 
                        if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']]
        print(f"  Valid (all constraints met): {len(valid_actions)} ({len(valid_actions)/total*100:.1f}%)")
        
        print("\nTop actions by reward:")
        for i, d in enumerate(sorted_data[:top_k]):
            violations = []
            if d['slo_violation']: violations.append('SLO')
            if d['temp_violation']: violations.append('TEMP')
            if d['power_violation']: violations.append('POWER')
            viol_str = f" [{', '.join(violations)}]" if violations else " [✓ All OK]"
            
            print(f"  {i+1}. Prefill={d['prefill_freq_bin']}, Decode={d['decode_freq_bin']}, "
                  f"Batch={d['batch_size']}, Reward={d['reward']:.4f}{viol_str}")
            
            # Calculate throughput if not stored
            if 'throughput_tokens_per_sec' in d:
                throughput_tps = d['throughput_tokens_per_sec']
                throughput_norm = d.get('throughput_normalized', 0.0)
            else:
                # Fallback: calculate from available data
#               total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
                total_tokens = d.get('total_tokens', 0)
                total_time = d['end_to_end_latency']
                throughput_tps = total_tokens / total_time if total_time > 0 else 0
                throughput_norm = d.get('throughput_normalized', 0.0)
            
            print(f"     E2E={d['end_to_end_latency']:.3f}s, Throughput={throughput_tps:.1f} tok/s (norm={throughput_norm:.3f}), "
                  f"Temp={d['temperature']:.1f}°C, Power={d['power']:.1f}W, Power_Headroom={d['power_headroom']:.2f}, Temp_Headroom={d['temperature_headroom']:.2f}")
        
        return sorted_data[:top_k]
    
    def plot_top_actions(self, top_k=10, save_path='plots/early_exit_top_actions_plot.png'):
        """Plot visualization of top-k actions with their metrics."""
        # Ensure plots directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
        sorted_data = sorted(self.processed_data, key=lambda x: x['reward'], reverse=True)
        top_actions = sorted_data[:top_k]
        
        if not top_actions:
            print("No actions to plot!")
            return
        
        # Extract metrics for plotting
        action_labels = []
        rewards = []
        e2e_latencies = []
        throughputs = []
        powers = []
        temperatures = []
        
        for i, d in enumerate(top_actions):
            # Create action label
            action_label = f"P{d['prefill_freq_bin']}-D{d['decode_freq_bin']}-B{d['batch_size']}"
            action_labels.append(action_label)
            
            rewards.append(d['reward'])
            e2e_latencies.append(d['end_to_end_latency'])
            
            # Calculate throughput if not stored
            if 'throughput_tokens_per_sec' in d:
                throughput_tps = d['throughput_tokens_per_sec']
            else:
#                total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
                total_tokens = d.get('total_tokens', 0)
                total_time = d['end_to_end_latency']
                throughput_tps = total_tokens / total_time if total_time > 0 else 0
            throughputs.append(throughput_tps)
            
            powers.append(d['power'])
            temperatures.append(d['temperature'])
        
        # Create figure with subplots - optimized size and spacing
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
        fig.suptitle(f'Top {top_k} Actions by Reward - Performance Metrics', fontsize=13, fontweight='bold', y=0.98)
        
        x_pos = np.arange(len(action_labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(action_labels)))
        
        # 1. Reward (bar chart)
        axes[0, 0].barh(x_pos, rewards, color=colors, height=0.8)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(action_labels, fontsize=7)
        axes[0, 0].set_xlabel('Reward', fontweight='bold', fontsize=8)
        axes[0, 0].set_title('Reward Score', fontweight='bold', fontsize=9)
        axes[0, 0].grid(axis='x', alpha=0.3, linestyle=':')
        axes[0, 0].invert_yaxis()  # Highest reward at top
        
        # 2. E2E Latency (bar chart)
        axes[0, 1].barh(x_pos, e2e_latencies, color=colors, height=0.8)
        axes[0, 1].set_yticks(x_pos)
        axes[0, 1].set_yticklabels(action_labels, fontsize=7)
        axes[0, 1].set_xlabel('End-to-End Latency (s)', fontweight='bold', fontsize=8)
        axes[0, 1].set_title('E2E Latency (Lower is Better)', fontweight='bold', fontsize=9)
        axes[0, 1].grid(axis='x', alpha=0.3, linestyle=':')
        axes[0, 1].axvline(x=self.slo_target, color='r', linestyle='--', linewidth=1.5, 
                          label=f'SLO: {self.slo_target}s', zorder=10)
        axes[0, 1].legend(loc='lower right', fontsize=7, framealpha=0.95, edgecolor='gray', frameon=True)
        axes[0, 1].invert_yaxis()
        
        # 3. Throughput (bar chart)
        axes[0, 2].barh(x_pos, throughputs, color=colors, height=0.8)
        axes[0, 2].set_yticks(x_pos)
        axes[0, 2].set_yticklabels(action_labels, fontsize=7)
        axes[0, 2].set_xlabel('Throughput (tokens/s)', fontweight='bold', fontsize=8)
        axes[0, 2].set_title('Throughput (Higher is Better)', fontweight='bold', fontsize=9)
        axes[0, 2].grid(axis='x', alpha=0.3, linestyle=':')
        axes[0, 2].invert_yaxis()
        
        # 4. Power (bar chart)
        axes[1, 0].barh(x_pos, powers, color=colors, height=0.8)
        axes[1, 0].set_yticks(x_pos)
        axes[1, 0].set_yticklabels(action_labels, fontsize=7)
        axes[1, 0].set_xlabel('Power (W)', fontweight='bold', fontsize=8)
        axes[1, 0].set_title('Power Consumption', fontweight='bold', fontsize=9)
        axes[1, 0].grid(axis='x', alpha=0.3, linestyle=':')
        axes[1, 0].axvline(x=self.max_power, color='r', linestyle='--', linewidth=1.5, 
                          label=f'Max: {self.max_power}W', zorder=10)
        axes[1, 0].legend(loc='lower right', fontsize=7, framealpha=0.95, edgecolor='gray', frameon=True)
        axes[1, 0].invert_yaxis()
        
        # 5. Temperature (bar chart)
        axes[1, 1].barh(x_pos, temperatures, color=colors, height=0.8)
        axes[1, 1].set_yticks(x_pos)
        axes[1, 1].set_yticklabels(action_labels, fontsize=7)
        axes[1, 1].set_xlabel('Temperature (°C)', fontweight='bold', fontsize=8)
        axes[1, 1].set_title('Temperature', fontweight='bold', fontsize=9)
        axes[1, 1].grid(axis='x', alpha=0.3, linestyle=':')
        axes[1, 1].axvline(x=self.max_temp, color='r', linestyle='--', linewidth=1.5, 
                          label=f'Max: {self.max_temp}°C', zorder=10)
        axes[1, 1].legend(loc='lower right', fontsize=7, framealpha=0.95, edgecolor='gray', frameon=True)
        axes[1, 1].invert_yaxis()
        
        # 6. Reward vs Throughput (scatter plot)
        scatter = axes[1, 2].scatter(throughputs, rewards, c=range(len(throughputs)), 
                                    cmap='viridis', s=120, alpha=0.75, edgecolors='black', linewidths=1.5, zorder=5)
        
        # Only label top 3 actions to avoid clutter
        top_3_indices = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:3]
        for i, idx in enumerate(top_3_indices):
            x, y = throughputs[idx], rewards[idx]
            # Use data coordinates with small offsets
            x_range = max(throughputs) - min(throughputs)
            y_range = max(rewards) - min(rewards)
            x_offset = x_range * 0.03
            y_offset = y_range * 0.03 * (i + 1)
            axes[1, 2].text(x + x_offset, y + y_offset, action_labels[idx],
                           fontsize=6, alpha=0.9, zorder=10,
                           bbox=dict(boxstyle='round,pad=0.25', facecolor='yellow', 
                                    alpha=0.85, edgecolor='black', linewidth=0.8))
        
        axes[1, 2].set_xlabel('Throughput (tokens/s)', fontweight='bold', fontsize=8)
        axes[1, 2].set_ylabel('Reward', fontweight='bold', fontsize=8)
        axes[1, 2].set_title('Reward vs Throughput', fontweight='bold', fontsize=9)
        axes[1, 2].grid(alpha=0.3, linestyle=':')
        cbar = plt.colorbar(scatter, ax=axes[1, 2], label='Rank', pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Rank (lower is better)', fontsize=7)
        
        # Adjust spacing - minimize whitespace
        plt.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.35, wspace=0.35)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        print(f"\nTop actions plot saved to '{save_path}'")
        plt.close()
    
    def get_valid_actions(self, top_k=10):
        """Get top-k actions that meet ALL constraints."""
        # Filter for actions meeting all constraints
        valid_data = [d for d in self.processed_data 
                     if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']]
        
        if not valid_data:
            print("\nWARNING: No actions meet all constraints!")
            return []
        
        sorted_valid = sorted(valid_data, key=lambda x: x['reward'], reverse=True)
        
        print(f"\nTop {min(top_k, len(sorted_valid))} VALID actions (all constraints met):")
        for i, d in enumerate(sorted_valid[:top_k]):
            print(f"  {i+1}. Prefill={d['prefill_freq_bin']}, Decode={d['decode_freq_bin']}, "
                  f"Batch={d['batch_size']}, Reward={d['reward']:.4f}")
            
            # Calculate throughput if not stored
            if 'throughput_tokens_per_sec' in d:
                throughput_tps = d['throughput_tokens_per_sec']
                throughput_norm = d.get('throughput_normalized', 0.0)
            else:
                # Fallback: calculate from available data
                total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
                total_time = d['end_to_end_latency']
                throughput_tps = total_tokens / total_time if total_time > 0 else 0
                throughput_norm = d.get('throughput_normalized', 0.0)
            
            print(f"     E2E={d['end_to_end_latency']:.3f}s, Throughput={throughput_tps:.1f} tok/s (norm={throughput_norm:.3f}), "
                  f"Temp={d['temperature']:.1f}°C, Power={d['power']:.1f}W, Headroom={d['power_headroom']:.2f}")
        
        return sorted_valid[:top_k]
    
    def get_action_statistics(self):
        """Compute statistics per action."""
        action_stats = {}
        for d in self.processed_data:
            action = (d['prefill_freq_bin'], d['decode_freq_bin'], d['batch_size'])
            if action not in action_stats:
                action_stats[action] = {
                    'rewards': [], 'energies': [], 'latencies': [], 
                    'end_to_end_latencies': [], 'temperatures': [], 
                    'powers': [], 'power_headrooms': []
                }
            action_stats[action]['rewards'].append(d['reward'])
            action_stats[action]['energies'].append(d['energy'])
            action_stats[action]['latencies'].append(d['latency'])
            action_stats[action]['end_to_end_latencies'].append(d['end_to_end_latency'])
            action_stats[action]['temperatures'].append(d['temperature'])
            action_stats[action]['powers'].append(d['power'])
            action_stats[action]['power_headrooms'].append(d['power_headroom'])
        
        for action in action_stats:
            stats = action_stats[action]
            action_stats[action] = {
                'mean_reward': np.mean(stats['rewards']),
                'std_reward': np.std(stats['rewards']),
                'mean_energy': np.mean(stats['energies']),
                'mean_latency': np.mean(stats['latencies']),
                'mean_end_to_end_latency': np.mean(stats['end_to_end_latencies']),
                'std_end_to_end_latency': np.std(stats['end_to_end_latencies']),
                'mean_temperature': np.mean(stats['temperatures']),
                'mean_power': np.mean(stats['powers']),
                'mean_power_headroom': np.mean(stats['power_headrooms']),
                'count': len(stats['rewards'])
            }
        
        return action_stats


# ============================================================
# GRPO POLICY NETWORK (NO VALUE HEAD - KEY DIFFERENCE FROM PPO)
# ============================================================

class GRPOPolicy(nn.Module):
    """
    GRPO Policy Network - Only actor, no critic.
    GRPO computes advantages from group comparisons, not learned values.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32, num_layers=4,
                 freq_bins=11, max_batch=32):
        super(GRPOPolicy, self).__init__()
        
        self.action_dim = action_dim
        self.num_layers = num_layers  # Store for later retrieval
        self.hidden_dim = hidden_dim
        self.freq_bins = freq_bins
        self.max_batch = max_batch
        
        # Simple MLP feature extractor
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)
        
        # Standard: Full linear layer
        self.actor = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        features = self.features(state)
        
        # Standard: single logits vector
        logits = self.actor(features)
        
        # Check for NaN/Inf in logits before softmax
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            print("WARNING: NaN/Inf in action logits, resetting to zeros")
            logits = torch.zeros_like(logits)
        
        action_probs = torch.softmax(logits, dim=-1)
        
        # Final check for NaN/Inf in probabilities
        if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
            print("WARNING: NaN/Inf in action probabilities after softmax, using uniform")
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        return action_probs, logits
    
    def get_log_probs(self, state, actions):
        """Get log probabilities for given actions."""
        # Standard method
        action_probs, logits = self.forward(state)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather log probs for the taken actions
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs, action_probs
    
    def get_action_probs(self, state):
        """Get action probabilities."""
        action_probs, _ = self.forward(state)
        return action_probs
    
    def act(self, state, deterministic=False):
        """Sample an action from the policy."""
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # Ensure action_log_prob is a scalar tensor (squeeze if needed)
        if action_log_prob.dim() > 0:
            action_log_prob = action_log_prob.squeeze()
        return action.item(), action_log_prob


# ============================================================
# OFFLINE PRE-TRAINING FOR GRPO
# ============================================================

def pretrain_grpo_from_offline_data(csv_path, freq_bins=11, max_batch=1,
                                     epochs=100, batch_size=32, lr=5e-3, alpha=0.8,
                                     slo_target=15.0, max_temp=100.0, max_power=60.0,
                                     num_layers=4):
    """Pre-train the GRPO policy using offline data.
    
    Args:
        alpha: Weight for energy vs latency (0 = latency only, 1 = energy only)
        slo_target: Maximum end-to-end latency (SLO)
        max_temp: Maximum temperature constraint
        max_power: Maximum power constraint
    """
    
    state_dim = 5  # prefill_freq, decode_freq, batch_size, temperature_headroom, layers


    # Load dataset with constraints
    dataset = OfflineDataset(csv_path, freq_bins=freq_bins, max_batch=max_batch, alpha=alpha,
                             slo_target=slo_target, max_temp=max_temp, max_power=max_power)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nReward weighting: Unified reward function with adaptive H_eff")
    print(f"  R = H_eff * T_n + (1 - H_eff) * EDP_n - penalties")
    print(f"  H_eff = w_T * H_T + w_P * H_P (combines temperature and power headroom)")
    
    action_dim = dataset.action_dim
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize GRPO policy (no value head)
    # Use dataset's action_dim (117) directly - policy will output 117 actions
    policy = GRPOPolicy(state_dim, action_dim, num_layers=num_layers,
                       freq_bins=freq_bins, max_batch=max_batch)

    # Use constant learning rate (no scheduler for faster convergence)
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    
    # Create target distribution based on rewards
    # Option 1: Use only valid actions (meet all constraints)
    # Option 2: Use average reward from valid samples only (ignoring violations)
    # We'll use Option 2 to be more robust to dataset noise
    action_rewards = np.zeros(action_dim)
    action_counts = np.zeros(action_dim)
    
    # Collect all valid rewards per action (NO SPLITTING BY HEADROOM)
    action_valid_rewards = {}  # {action_idx: [rewards]}
    
    for d in dataset.processed_data:
        # Only count samples that meet ALL constraints
        if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
            action_idx = d['action_idx']
            if action_idx not in action_valid_rewards:
                action_valid_rewards[action_idx] = []
            action_valid_rewards[action_idx].append(d['reward'])
    
    # Helper function to create target distribution from action rewards
    def create_target_distribution(action_valid_rewards_dict, action_dim, name, prefer_small_batches=False):
        """Create target probability distribution from action rewards.
        Args:
            action_valid_rewards_dict: Dictionary mapping action_idx to list of rewards
            action_dim: Total number of actions
            name: Name for debugging
            prefer_small_batches: If True, add bonus to smaller batch sizes (for low headroom)
        """
        avg_action_rewards = np.zeros(action_dim)
        action_counts = np.zeros(action_dim)
        
        for action_idx in range(action_dim):
            if action_idx in action_valid_rewards_dict and len(action_valid_rewards_dict[action_idx]) > 0:
                # Use MEAN reward from valid samples
                avg_reward = np.mean(action_valid_rewards_dict[action_idx])
                
                # No batch bonus - use raw rewards from CSV as-is
                # (prefer_small_batches parameter kept for compatibility but not used)
                avg_action_rewards[action_idx] = avg_reward
                action_counts[action_idx] = len(action_valid_rewards_dict[action_idx])
            else:
                # No valid samples - use average from all samples (including violations)
                # But weight it down so it won't rank highly
                action_rewards_all = []
                for d in dataset.processed_data:
                    if d['action_idx'] == action_idx:
                        action_rewards_all.append(d['reward'])
                if action_rewards_all:
                    avg_action_rewards[action_idx] = np.mean(action_rewards_all) - 10.0  # Penalty
                    action_counts[action_idx] = len(action_rewards_all)
                else:
                    avg_action_rewards[action_idx] = -100.0  # Very low for actions never seen
                    action_counts[action_idx] = 1
        
        # Create target distribution using softmax over all actions based on mean rewards
        # Use lower temperature to create sharper distribution (focus on high-reward actions)
        # Lower temperature = sharper distribution (more probability on top actions)
        temperature = 0.001  # Reduced from 0.1 to make distribution sharper
        scaled_rewards = (avg_action_rewards - avg_action_rewards.mean()) / (avg_action_rewards.std() + 1e-8)
        target_probs = torch.softmax(torch.FloatTensor(scaled_rewards) / temperature, dim=0)
        
        return target_probs, avg_action_rewards, action_counts
    
    # Calculate batch sizes list (needed for action decoding in print statements)
    batch_sizes = list(range(1, max_batch, 6))  # Steps of 6: 1, 7, 13, 19, 25, 31
    num_batch_sizes = len(batch_sizes)
    num_freq_values = freq_bins
    
    # Analyze dataset headroom distribution
    # Low headroom should ONLY come from fan_off CSV (high temperature samples)
    # High headroom comes from grid_search CSV (normal temperature samples)
    headroom_threshold = 0.33
    low_headroom_samples = []
    high_headroom_samples = []
    # Analyze ALL samples (including violations) to see full headroom distribution
    for d in dataset.processed_data:
        state = d['state']
        temperature_headroom = state[3] if len(state) > 3 else 0.5
        source_file = d.get('source_file', 'unknown')
        
        # Low headroom: only samples from fan_off CSV (high temperature, fan off)
        # High headroom: samples from grid_search CSV (normal temperature, fan on)
        if 'fan_off' in source_file.lower():
            # All fan_off samples are low headroom (high temperature)
            low_headroom_samples.append(d)
        else:
            # All grid_search samples are high headroom (normal temperature)
            high_headroom_samples.append(d)
    
    print(f"\nDataset headroom analysis (all samples, including violations):")
    print(f"  Low headroom (fan_off CSV, high temperature): {len(low_headroom_samples)} total samples")
    print(f"  High headroom (grid_search CSV, normal temperature): {len(high_headroom_samples)} total samples")
    
    # Analyze source file distribution
    if len(low_headroom_samples) > 0:
        low_headroom_sources = {}
        for d in low_headroom_samples:
            source = d.get('source_file', 'unknown')
            low_headroom_sources[source] = low_headroom_sources.get(source, 0) + 1
        print(f"  Low headroom samples by source file:")
        for source, count in low_headroom_sources.items():
            print(f"    {source}: {count} samples")
        
        # Analyze batch sizes in low headroom samples
        low_headroom_batches = [d['batch_size'] for d in low_headroom_samples]
        print(f"  Low headroom batch sizes: min={min(low_headroom_batches)}, max={max(low_headroom_batches)}, "
              f"avg={np.mean(low_headroom_batches):.2f}")
    
    if len(high_headroom_samples) > 0:
        high_headroom_sources = {}
        for d in high_headroom_samples:
            source = d.get('source_file', 'unknown')
            high_headroom_sources[source] = high_headroom_sources.get(source, 0) + 1
        print(f"  High headroom samples by source file:")
        for source, count in high_headroom_sources.items():
            print(f"    {source}: {count} samples")
        
        high_headroom_batches = [d['batch_size'] for d in high_headroom_samples]
        print(f"  High headroom batch sizes: min={min(high_headroom_batches)}, max={max(high_headroom_batches)}, "
              f"avg={np.mean(high_headroom_batches):.2f}")
    
    # Create separate target distributions for analysis (to show optimal actions per headroom)
    # Collect valid rewards per action, split by headroom
    action_valid_rewards_low = {}
    action_valid_rewards_high = {}
    
    # Debug: Track batch size distribution for valid low headroom samples
    valid_low_batches = []
    valid_high_batches = []
    
    for d in dataset.processed_data:
        if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
            action_idx = d['action_idx']
            source_file = d.get('source_file', 'unknown')
            
            # Categorize by source file, not just headroom value
            # Low headroom = fan_off CSV (high temperature)
            # High headroom = grid_search CSV (normal temperature)
            if 'fan_off' in source_file.lower():
                if action_idx not in action_valid_rewards_low:
                    action_valid_rewards_low[action_idx] = []
                action_valid_rewards_low[action_idx].append(d['reward'])
                valid_low_batches.append(d['batch_size'])
            else:
                if action_idx not in action_valid_rewards_high:
                    action_valid_rewards_high[action_idx] = []
                action_valid_rewards_high[action_idx].append(d['reward'])
                valid_high_batches.append(d['batch_size'])
    
    # Debug output
    if valid_low_batches:
        print(f"\nDEBUG: Valid low headroom (≤{headroom_threshold}) samples batch size distribution:")
        print(f"  Count: {len(valid_low_batches)}")
        print(f"  Min: {min(valid_low_batches)}, Max: {max(valid_low_batches)}, Avg: {np.mean(valid_low_batches):.2f}")
        batch_counts = {}
        for b in valid_low_batches:
            batch_counts[b] = batch_counts.get(b, 0) + 1
        print(f"  Top 5 batch sizes: {sorted(batch_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    if valid_high_batches:
        print(f"\nDEBUG: Valid high headroom (>{headroom_threshold}) samples batch size distribution:")
        print(f"  Count: {len(valid_high_batches)}")
        print(f"  Min: {min(valid_high_batches)}, Max: {max(valid_high_batches)}, Avg: {np.mean(valid_high_batches):.2f}")
        batch_counts = {}
        for b in valid_high_batches:
            batch_counts[b] = batch_counts.get(b, 0) + 1
        print(f"  Top 5 batch sizes: {sorted(batch_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # Create target distributions for each headroom condition
    # No batch bonus - use raw rewards from CSV as-is
    target_probs_low, avg_action_rewards_low, action_counts_low = create_target_distribution(
        action_valid_rewards_low, action_dim, "Low headroom target distribution", prefer_small_batches=False
    )
    target_probs_high, avg_action_rewards_high, action_counts_high = create_target_distribution(
        action_valid_rewards_high, action_dim, "High headroom target distribution", prefer_small_batches=False
    )
    
    # Print target distributions for each headroom condition
    print(f"\nTarget probability distribution - LOW headroom (fan_off CSV, high temperature) [top 10 actions]:")
    print(f"  NOTE: Optimal actions when temperature headroom is low (need to reduce batch size)")
    print(f"        Used for training states with low headroom (from fan_off CSV)")
    top_actions_low = torch.topk(target_probs_low, 10)
    for i, (prob, idx) in enumerate(zip(top_actions_low.values, top_actions_low.indices), 1):
        prefill_freq = idx.item() // (num_freq_values * num_batch_sizes)
        remainder = idx.item() % (num_freq_values * num_batch_sizes)
        decode_freq = remainder // num_batch_sizes
        batch = batch_sizes[remainder % num_batch_sizes]
        mean_reward = avg_action_rewards_low[idx.item()]
        valid_count = int(action_counts_low[idx.item()]) if action_counts_low[idx.item()] > 0 else 0
        print(f"  {i}. Action {idx.item()}: prefill={prefill_freq}, decode={decode_freq}, batch={batch}, "
              f"prob={prob.item():.4f}, mean_reward={mean_reward:.4f} ({valid_count} valid samples)")
    
    print(f"\nTarget probability distribution - HIGH headroom (grid_search CSV, normal temperature) [top 10 actions]:")
    print(f"  NOTE: Optimal actions when temperature headroom is high (can use larger batches)")
    print(f"        Used for training states with high headroom (from grid_search CSV)")
    top_actions_high = torch.topk(target_probs_high, 10)
    for i, (prob, idx) in enumerate(zip(top_actions_high.values, top_actions_high.indices), 1):
        prefill_freq = idx.item() // (num_freq_values * num_batch_sizes)
        remainder = idx.item() % (num_freq_values * num_batch_sizes)
        decode_freq = remainder // num_batch_sizes
        batch = batch_sizes[remainder % num_batch_sizes]
        mean_reward = avg_action_rewards_high[idx.item()]
        valid_count = int(action_counts_high[idx.item()]) if action_counts_high[idx.item()] > 0 else 0
        print(f"  {i}. Action {idx.item()}: prefill={prefill_freq}, decode={decode_freq}, batch={batch}, "
              f"prob={prob.item():.4f}, mean_reward={mean_reward:.4f} ({valid_count} valid samples)")
    
    print(f"\nTraining will use CONDITIONAL target distributions:")
    print(f"  - Low headroom states (from fan_off CSV, high temperature) → use low headroom target distribution")
    print(f"  - High headroom states (from grid_search CSV, normal temperature) → use high headroom target distribution")
    print(f"  This ensures the policy learns different behaviors for different headroom conditions")
    
    # Pre-training loop
    losses = []
    loss_changes = []
    print(f"\nStarting GRPO pre-training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_data in dataloader:
            states = batch_data['state']
            # Ensure actions is 1D tensor (batch_size,)
            actions = batch_data['action_idx']
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            elif actions.dim() == 0:
                actions = actions.unsqueeze(0)
            
            rewards = batch_data['reward']
            if rewards.dim() > 1:
                rewards = rewards.squeeze(-1)
            elif rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            
            # Use get_log_probs to handle standard method
            action_log_probs, action_probs = policy.get_log_probs(states, actions)
            
            # Reward-weighted policy gradient (behavioral cloning with reward weighting)
            # Normalize rewards to have mean 0, std 1, then shift to make weights positive
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            weights = torch.clamp(normalized_rewards + 1, min=0.1, max=2.0)
            
            # action_log_probs is already the log probability for the taken actions
            actor_loss = -torch.mean(weights * action_log_probs)
            
            # For KL divergence and entropy, we need the full probability distribution
            # action_probs from get_log_probs is already the full flattened distribution
            log_probs = torch.log(action_probs + 1e-10)
            
            # KL divergence to CONDITIONAL target distribution based on headroom
            # Extract temperature headroom from states (4th dimension)
            # Low headroom = fan_off CSV (headroom ≤ 0.15, high temperature)
            # High headroom = grid_search CSV (headroom > 0.15, normal temperature)
            temperature_headrooms = states[:, 3]  # Extract headroom for each state in batch
            headroom_threshold = 0.33  # Stricter threshold: fan_off samples have headroom 0.006-0.15
            
            # Split by headroom threshold (proxy for source file)
            low_headroom_mask = (temperature_headrooms <= headroom_threshold)
            high_headroom_mask = (temperature_headrooms > headroom_threshold)
            
            # Compute KL loss separately for each headroom group
            kl_losses = []
            
            if low_headroom_mask.sum() > 0:
                # Use low headroom target distribution for low headroom states
                mean_log_probs_low = log_probs[low_headroom_mask].mean(dim=0)
                kl_loss_low = torch.sum(target_probs_low * (torch.log(target_probs_low + 1e-10) - mean_log_probs_low))
                kl_losses.append(kl_loss_low)
            
            if high_headroom_mask.sum() > 0:
                # Use high headroom target distribution for high headroom states
                mean_log_probs_high = log_probs[high_headroom_mask].mean(dim=0)
                kl_loss_high = torch.sum(target_probs_high * (torch.log(target_probs_high + 1e-10) - mean_log_probs_high))
                kl_losses.append(kl_loss_high)
            
            # Combine KL losses - if both exist, use equal weighting; otherwise use the one that exists
            if len(kl_losses) == 2:
                kl_loss = 0.5 * kl_losses[0] + 0.5 * kl_losses[1]
            elif len(kl_losses) == 1:
                kl_loss = kl_losses[0]
            else:
                # Fallback: should not happen (both headroom groups empty)
                # Use uniform distribution as fallback
                mean_log_probs = log_probs.mean(dim=0)
                uniform_target = torch.ones(action_dim) / action_dim
                kl_loss = torch.sum(uniform_target * (torch.log(uniform_target + 1e-10) - mean_log_probs))
            
            # Entropy bonus for exploration (entropy is positive, so we subtract it)
            entropy = -torch.mean(torch.sum(action_probs * log_probs, dim=-1))
            
            # Scale loss components appropriately
            # Combine reward-weighted policy gradient with conditional KL divergence
            loss = actor_loss + 2.5 * kl_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent instability
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        
        # Calculate loss change from previous epoch (for CSV tracking)
        if len(losses) > 1:
            loss_change = abs(losses[-1] - losses[-2])
            loss_changes.append(loss_change)
        else:
            loss_changes.append(None)  # No change for first epoch
        
        # Print epoch summary (every 10 epochs to reduce log spam)
        # Note: Loss can be negative when entropy bonus exceeds KL penalty
        # This is normal - it means the policy has high exploration (high entropy)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, LR: {lr:.6f}")
    
    # Evaluate final policy
    print("\nEvaluating pre-trained GRPO policy...")
    policy.eval()
    with torch.no_grad():
        # First: Evaluate on test states (arbitrary states for general behavior)
        print("\n1. Evaluation on test states:")
        print("   Testing conditional behavior: low headroom (≤0.33) should prefer batch=1, high headroom (>0.33) should prefer larger batches")
        print("   Using prefill_norm=0.5, decode_norm=0.5 (middle frequencies) to isolate temperature headroom effect")
        headroom_threshold = 0.33
        test_states = [
            # Low headroom cases (≤0.33) - should prefer batch=1, energy-efficient configs
            np.array([0.5, 0.5, 0.5, 0.10, 4], dtype=np.float32),  # Very low headroom (0.10) - should use batch=1
            np.array([0.5, 0.5, 0.5, 0.20, 8], dtype=np.float32),  # Low headroom (0.20) - should use batch=1
            np.array([0.5, 0.5, 0.5, 0.30, 12], dtype=np.float32),  # At threshold (0.30) - should use batch=1
            np.array([0.5, 0.5, 0.5, 0.33, 16], dtype=np.float32),  # At threshold (0.33) - should use batch=1
            
            # High headroom cases (>0.33) - should prefer larger batches, throughput-focused
            np.array([0.5, 0.5, 0.5, 0.50, 4], dtype=np.float32),  # Medium headroom (0.50) - can use larger batches
            np.array([0.5, 0.5, 0.5, 0.70, 8], dtype=np.float32),  # High headroom (0.70) - can use larger batches
            np.array([0.5, 0.5, 0.5, 0.90, 12], dtype=np.float32),  # Very high headroom (0.90) - can use larger batches
        ]
        
        batch_sizes = list(range(1, max_batch, 6))  # Steps of 6: 1, 7, 13, 19, 25, 31
        num_batch_sizes = len(batch_sizes)
        num_freq_values = freq_bins
        
        for state in test_states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action probabilities
            action_probs = policy.get_action_probs(state_tensor)
            action_probs = action_probs.squeeze()
            
            top_actions_result = torch.topk(action_probs, 5)
            
            print(f"\nState: prefill_norm={state[0]:.2f}, decode_norm={state[1]:.2f}, batch_norm={state[2]:.2f}, temp_headroom={state[3]:.2f}")
            print(f"  Top 5 actions:")
            
            for prob, idx in zip(top_actions_result.values, top_actions_result.indices):
                prefill_freq = idx.item() // (num_freq_values * num_batch_sizes)
                remainder = idx.item() % (num_freq_values * num_batch_sizes)
                decode_freq = remainder // num_batch_sizes
                batch = batch_sizes[remainder % num_batch_sizes]
                print(f"    prefill_freq={prefill_freq}, decode_freq={decode_freq}, batch={batch}, prob={prob.item():.4f}")
        
        # Second: Compare against target distributions (conditional)
        print("\n2. Comparison with conditional target distributions:")
        # Note: We use conditional target distributions, so evaluation is done per headroom condition
        # Get top actions from both low and high headroom target distributions
        top_target_actions_low = torch.topk(target_probs_low, 5).indices
        top_target_actions_high = torch.topk(target_probs_high, 5).indices
        # Combine for evaluation (we'll evaluate on both conditions)
        top_target_actions = torch.cat([top_target_actions_low, top_target_actions_high])
        
        # Collect states from dataset for these top actions
        test_states_dict = {}  # action_idx -> list of states
        for d in dataset.processed_data:
            action_idx = d['action_idx']
            if action_idx in top_target_actions.tolist():
                state = d['state']
                if action_idx not in test_states_dict:
                    test_states_dict[action_idx] = []
                test_states_dict[action_idx].append(state)
        
        # Evaluate on states corresponding to top 5 target actions
        print(f"\nComparing policy probabilities to target distribution:")
        print(f"  (Evaluating on states from dataset that correspond to top target actions)")
        
        batch_sizes = list(range(1, max_batch, 6))  # Steps of 6: 1, 7, 13, 19, 25, 31
        num_batch_sizes = len(batch_sizes)
        num_freq_values = freq_bins
        
        # Evaluate low headroom actions
        print(f"\n  LOW headroom condition (≤{headroom_threshold}):")
        for target_action_idx in top_target_actions_low[:5]:
            target_action_idx = target_action_idx.item()
            target_prob = target_probs_low[target_action_idx].item()
            
            # Decode target action
            prefill_freq_target = target_action_idx // (num_freq_values * num_batch_sizes)
            remainder = target_action_idx % (num_freq_values * num_batch_sizes)
            decode_freq_target = remainder // num_batch_sizes
            batch_target = batch_sizes[remainder % num_batch_sizes]
            
            # Get states that led to this action (low headroom)
            if target_action_idx in test_states_dict and len(test_states_dict[target_action_idx]) > 0:
                # Find a low headroom state for this action
                test_state = None
                for state in test_states_dict[target_action_idx]:
                    if state[3] <= headroom_threshold:
                        test_state = state
                        break
                if test_state is None:
                    continue
                    
                state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
                
                # Get policy's action probabilities
                policy_probs = policy.get_action_probs(state_tensor).squeeze()
                policy_prob_for_target = policy_probs[target_action_idx].item()
                
                # Calculate KL divergence between policy and low headroom target
                kl_div = torch.sum(target_probs_low * (torch.log(target_probs_low + 1e-10) - 
                                                   torch.log(policy_probs + 1e-10)))
                
                print(f"    Action {target_action_idx}: prefill={prefill_freq_target}, decode={decode_freq_target}, batch={batch_target}")
                print(f"      Target prob: {target_prob:.4f}, Policy prob: {policy_prob_for_target:.4f}, KL: {kl_div.item():.4f}")
        
        # Evaluate high headroom actions
        print(f"\n  HIGH headroom condition (>{headroom_threshold}):")
        for target_action_idx in top_target_actions_high[:5]:
            target_action_idx = target_action_idx.item()
            target_prob = target_probs_high[target_action_idx].item()
            
            # Decode target action
            prefill_freq_target = target_action_idx // (num_freq_values * num_batch_sizes)
            remainder = target_action_idx % (num_freq_values * num_batch_sizes)
            decode_freq_target = remainder // num_batch_sizes
            batch_target = batch_sizes[remainder % num_batch_sizes]
            
            # Get states that led to this action (high headroom)
            if target_action_idx in test_states_dict and len(test_states_dict[target_action_idx]) > 0:
                # Find a high headroom state for this action
                test_state = None
                for state in test_states_dict[target_action_idx]:
                    if state[3] > headroom_threshold:
                        test_state = state
                        break
                if test_state is None:
                    continue
                    
                state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
                
                # Get policy's action probabilities
                policy_probs = policy.get_action_probs(state_tensor).squeeze()
                policy_prob_for_target = policy_probs[target_action_idx].item()
                
                # Calculate KL divergence between policy and high headroom target
                kl_div = torch.sum(target_probs_high * (torch.log(target_probs_high + 1e-10) - 
                                                        torch.log(policy_probs + 1e-10)))
                
                print(f"    Action {target_action_idx}: prefill={prefill_freq_target}, decode={decode_freq_target}, batch={batch_target}")
                print(f"      Target prob: {target_prob:.4f}, Policy prob: {policy_prob_for_target:.4f}, KL: {kl_div.item():.4f}")
        
        # Note: We can't compute a single "average" KL divergence since we use conditional targets
        # The evaluation above shows per-condition comparisons
        print(f"\n  Note: Using conditional target distributions - see per-condition evaluations above")
    
    # # Close CSV file
    # csv_file.close()
    # print(f"\nCSV logging completed. Results saved to: {csv_filename}")
    
    policy.train()
    return policy, losses, dataset



# ============================================================
# REAL JETSON ENVIRONMENT (FOR ACTUAL HARDWARE)
# ============================================================

class RealJetsonEnv:
    """
    Real Jetson environment that reads actual sensor values from hardware.
    
    Usage for online training on real Jetson:
    1. Set USE_REAL_JETSON = True in main section
    2. Implement _run_actual_inference() with your LLM
    3. Run script on Jetson device
    
 This environment measures ACTUAL hardware metrics during online training.
    """
    
    def __init__(self, freq_bins=11, max_batch=1, 
                 slo_target=15.0, max_temp=95.0, min_temp=40.0, max_power=15.0,
                 model_id="meta-llama/Llama-3.2-1B-Instruct"):
        """
        Args:
            freq_bins: Number of frequency bins
            max_batch: Maximum batch size
            slo_target: Maximum end-to-end latency (SLO)
            max_temp: Maximum temperature constraint
            max_power: Maximum power constraint
            model_id: HuggingFace model ID for Llama model
        """
        self.freq_bins = freq_bins
        self.max_batch = max_batch
        self.state_dim = 5  # prefill_freq, decode_freq, batch_size, temperature_headroom
        self.slo_target = slo_target
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.max_power = max_power
        # Initialize Llama runner for actual inference
        print(f"\n[RealJetsonEnv] Loading Llama model: {model_id}")
        try:
            # Llama318BRunner should already be imported at module level
            self.runner = Llama318BRunner(model_id=model_id)
            print(f"[RealJetsonEnv] Model loaded successfully!")
        except Exception as e:
            print(f"[RealJetsonEnv] ERROR: Failed to load model: {e}")
            print(f"[RealJetsonEnv] This is likely because the 'nq_subset' dataset directory is missing.")
            print(f"[RealJetsonEnv] Please ensure the dataset is available before running online training.")
            import traceback
            print(f"[RealJetsonEnv] Full traceback:")
            traceback.print_exc()
            print(f"[RealJetsonEnv] Falling back to placeholder inference (training will not work properly!)")
            self.runner = None
        
        # Verify runner is available
        if self.runner is None:
            print(f"\n[RealJetsonEnv] WARNING: Runner is None! Training will use placeholder values.")
            print(f"[RealJetsonEnv] This indicates a model loading failure. Check the error messages above.")
        
        # GPU frequency bins (Hz) - use shared constant
        self.GPU_FREQ_BINS = GPU_FREQ_BINS
        
        # Batch sizes: 0 to 32 (inclusive)
        # Must match OfflineDataset batch size range to ensure action space compatibility!
        
        # Initialize tracking lists for metrics
        self.inference_times = []
        self.batching_times = []
        self.compute_times = []
        self.power_measurements = []
        self.batch_sizes = list(range(1, 32, 6))  # Steps of 6: [1, 7, 13, 19, 25, 31]
        self.num_batch_sizes = len(self.batch_sizes)
        
        # Initialize previous action tracking for hysteresis penalty
        self.prev_prefill_bin = None
        self.prev_decode_bin = None
        self.prev_batch_size = None
        self.prev_early_exit_layer = None
        
        # Build action list: (prefill_freq, decode_freq, batch)
        # Must match OfflineDataset action space structure!
        # freq_bins represents the number of bins, so range(0, freq_bins) gives indices 0 to (freq_bins-1)
        self.actions = []
        for prefill_freq in range(0, freq_bins):  # 0 to (freq_bins-1) inclusive
            for decode_freq in range(0, freq_bins):  # 0 to (freq_bins-1) inclusive
                for batch in self.batch_sizes:  # 1 to 32 (inclusive)
                    for early_exit_layer in range(9, 17, 1):  # Layers: 9, 10, 11, 12, 13, 14, 15, 16
                        self.actions.append((prefill_freq, decode_freq, batch, early_exit_layer))
        
        self.action_dim = len(self.actions)
        print(f"Real Jetson environment initialized with {self.action_dim} actions")
        print(f"  Action space: {freq_bins} x {freq_bins} x {self.num_batch_sizes} (batch sizes: {self.batch_sizes}) x 8 (early exit layers: 9, 10, 11, 12, 13, 14, 15, 16) = {self.action_dim} actions")
                
        self.reset()
    
    def reset(self):
        self.prefill_freq = self.freq_bins // 2
        self.decode_freq = self.freq_bins // 2
        # Use middle batch size from batch_sizes list (matches OfflineDataset)
        self.batch_size = self.batch_sizes[len(self.batch_sizes) // 2]
        # Initialize temperature to a reasonable default (70% of max)
        self.current_temp = self.max_temp * 0.7
        self.last_early_exit_layer = 16
        return self._get_state()
    
    def _get_state(self):
        # Find batch index in batch_sizes list
        batch_idx = self.batch_sizes.index(self.batch_size)
        batch_norm = batch_idx / (len(self.batch_sizes) - 1) if len(self.batch_sizes) > 1 else 0.0
        
        # Calculate temperature headroom
        current_temp = getattr(self, 'current_temp', self.max_temp * 0.7)
        temperature_headroom = max(0, (self.max_temp - current_temp) / (self.max_temp - self.min_temp))
        
        # Current state (4D)
        current_state = np.array([
            self.prefill_freq / (self.freq_bins - 1) if self.freq_bins > 1 else 0.0,
            self.decode_freq / (self.freq_bins - 1) if self.freq_bins > 1 else 0.0,
            batch_norm,
            temperature_headroom
        ], dtype=np.float32)
        
        # Previous layer (1D) - normalized to [0, 1]
        previous_layer = np.array([
            (self.last_early_exit_layer - 9) / 7.0  # Normalize 9-16 → 0-1
        ], dtype=np.float32)
        
        # Concatenate: [current(4D) | previous_layer(1D)] = 5D total
        full_state = np.concatenate([current_state, previous_layer])
        return full_state
    
    def step(self, action_idx):
        """
        Execute action on real Jetson hardware and measure actual metrics.
        Uses Llama318BRunner to run actual LLM inference with specified batch size, frequencies, and early exit layer.
        The early exit layer is selected by the RL agent, which learns to balance latency (fewer layers) 
        vs entropy/accuracy (more layers) through the reward function.
        """
        prefill_freq_bin, decode_freq_bin, batch_size, early_exit_layer = self.actions[action_idx]
        
        # Use the early_exit_layer selected by the RL agent
        # The RL agent learns to balance latency (fewer layers = faster) vs entropy (more layers = better accuracy)
        # We trust the agent's choice rather than overriding it with adaptive logic
        # The reward function already penalizes high entropy, so the agent will learn to avoid too few layers
        
        # Validate early_exit_layer is in valid range
        num_layers = getattr(self.runner, 'num_layers', 16) if self.runner is not None else 16
        if early_exit_layer < 4 or early_exit_layer > num_layers:
            # Clamp to valid range
            early_exit_layer = max(4, min(early_exit_layer, num_layers))
        
        # Store early_exit_layer for later retrieval in info dict
        self.early_exit_layer = early_exit_layer
        
        # # Convert to None if it's the full model (for compatibility with step_env)
        # if early_exit_layer >= num_layers:
        #     early_exit_layer = None  # None means use full model
        
        # Run actual inference if runner is available
        if self.runner is not None:
            # Time the inference
            start_inference = time.time()
            # Run actual LLM inference with the specified configuration and early exit
            results = self.runner.step_env(
                prefill_freq_bin=prefill_freq_bin,
                decode_freq_bin=decode_freq_bin,
                batch_size=batch_size,
                logging=False,  # Set to True for verbose logging
                max_new_tokens=100,  # Adjust as needed
                early_exit_layer=early_exit_layer  # Pass early exit layer
            )
            inference_time = time.time() - start_inference
            self.inference_times.append(inference_time)
            
            # Extract metrics from inference results
            latency_prefill = results['prefill_time']
            latency_decode = results['decode_time']
            tokens_prefill = results['num_prompt_tokens']
            tokens_decode = results['num_decode_tokens']
            end_to_end_latency = latency_prefill + latency_decode
            
            # Track batching and compute times separately
            tokenize_time = results.get('tokenize_time', 0.0)
            prefill_compute_time = results.get('prefill_compute_time', latency_prefill)
            self.batching_times.append(tokenize_time)
            self.compute_times.append(prefill_compute_time)
            
            # Extract power metrics (from step_env power measurements)
            power_prefill = results['prefill_TOTAL_POWER']  
            power_decode = results['decode_TOTAL_POWER']
            
            # Average power across prefill and decode phases
            total_power = (power_prefill + power_decode) / 2.0
            
            # Track power measurements
            total_energy = (power_prefill * latency_prefill) + (power_decode * latency_decode)
            self.power_measurements.append({
                'avg_power_w': total_power,
                'total_energy_j': total_energy,
                'prefill_power_w': power_prefill,
                'decode_power_w': power_decode
            })
            
            # Read temperature from sensors (step_env may also provide this)
            # Use TJ_TEMP only (junction temp, hottest spot), ignore GPU_TEMP
            temp_prefill = results.get('prefill_TJ_TEMP', 0)
            temp_decode = results.get('decode_TJ_TEMP', 0)
            if temp_prefill > 0 and temp_decode > 0:
                temperature = (temp_prefill + temp_decode) / 2.0
            else:
                # Fallback: read from sensors if not in results
                temperature = self._read_jetson_temperature()
            
            # Extract entropy from results (calculated in decode_batch)
            avg_entropy = results.get('avg_entropy', None)
            if avg_entropy is not None:
                # Convert to float if it's a tensor
                if hasattr(avg_entropy, 'item'):
                    avg_entropy = float(avg_entropy.item())
                else:
                    avg_entropy = float(avg_entropy)
            
            # Store current temperature for state calculation
            self.current_temp = temperature
            
            # Calculate energy per token
            energy_prefill_per_token = (power_prefill * latency_prefill) / tokens_prefill if tokens_prefill > 0 else 0
            energy_decode_per_token = (power_decode * latency_decode) / tokens_decode if tokens_decode > 0 else 0
            
            # Total energy
            energy_total = (power_prefill * latency_prefill) + (power_decode * latency_decode)
            
            # Per-token latency
            per_token_latency = end_to_end_latency / (tokens_prefill + tokens_decode) if (tokens_prefill + tokens_decode) > 0 else 0
            
            # Power headroom
            power_headroom = max(0, (self.max_power - total_power) / self.max_power)
            
            # Debug: Print inference results (first time only)
            if not hasattr(self, '_inference_debug_logged'):
                print(f"\n[DEBUG] RealJetsonEnv: Running ACTUAL LLM inference")
                print(f"  Action: prefill_freq={prefill_freq_bin}, decode_freq={decode_freq_bin}, batch={batch_size}, early_exit_layer={early_exit_layer}")
                print(f"  Prefill: {latency_prefill:.4f}s, {tokens_prefill} tokens, {power_prefill:.2f}W")
                print(f"  Decode: {latency_decode:.4f}s, {tokens_decode} tokens, {power_decode:.2f}W")
                print(f"  E2E: {end_to_end_latency:.4f}s, Total Power: {total_power:.2f}W, Temp: {temperature:.1f}°C")
                if avg_entropy is not None:
                    print(f"  Entropy: {avg_entropy:.4f}")
                self._inference_debug_logged = True
        else:
            # Fallback if runner is not available (shouldn't happen in real usage)
            # Initialize default values for all required variables
            avg_entropy = None
            tokens_prefill = 0
            tokens_decode = 0
            end_to_end_latency = 0.0
            energy_total = 0.0
            total_energy = 0.0  # Same as energy_total, used in calculate_reward
            temperature = self._read_jetson_temperature() if hasattr(self, '_read_jetson_temperature') else 50.0
            total_power = 0.0
            power_headroom = 1.0
            latency_prefill = 0.0
            latency_decode = 0.0
            per_token_latency = 0.0  # Initialize per_token_latency for fallback case
            print("[WARNING] RealJetsonEnv: Runner not available, using placeholder values")
        
        # Calculate reward using shared reward function (same as offline training)
        # Calculate throughput: decode tokens per second only
        # Only tracking decode phase efficiency, not total request throughput
        throughput_raw = tokens_decode / (latency_decode + 1e-6) if latency_decode > 0 else 0.0  # tokens/sec
        
        # Calculate EDP efficiency
        # EDP = Energy * Latency, where latency is per-request latency (total time)
        # end_to_end_latency is per-request latency (seconds), not per-token latency
        edp = energy_total * end_to_end_latency
        edp_efficiency_raw = 1.0 / (edp + 1e-6)  # Invert so higher is better
        
        # Store raw values for normalization range updates (if tracking is enabled)
        # This allows normalization to include both offline and online data
        if hasattr(self, '_track_normalization') and self._track_normalization:
            if not hasattr(self, '_online_throughput_raw_values'):
                self._online_throughput_raw_values = []
                self._online_edp_efficiency_raw_values = []
                self._online_energy_raw_values = []
            self._online_throughput_raw_values.append(throughput_raw)
            self._online_edp_efficiency_raw_values.append(edp_efficiency_raw)
            self._online_energy_raw_values.append(energy_total)
            
            # Update normalization ranges to include new data
            # Combine offline dataset ranges with online data
            # print(f"self.throughput_min: {self.throughput_min}, self.throughput_max: {self.throughput_max}")
            all_throughput_values = [self.throughput_min, self.throughput_max] + self._online_throughput_raw_values
            all_edp_values = [self.edp_efficiency_min, self.edp_efficiency_max] + self._online_edp_efficiency_raw_values
            all_energy_values = [self.energy_min, self.energy_max] + self._online_energy_raw_values
           
            self.throughput_min = min(all_throughput_values)
            self.throughput_max = max(all_throughput_values)
            self.edp_efficiency_min = min(all_edp_values)
            self.edp_efficiency_max = max(all_edp_values)
            self.energy_min = min(all_energy_values)
            self.energy_max = max(all_energy_values)
        # Use shared reward calculation function
        # Get normalization ranges if available (from dataset)
        throughput_min = getattr(self, 'throughput_min', None)
        throughput_max = getattr(self, 'throughput_max', None)
        edp_efficiency_min = getattr(self, 'edp_efficiency_min', None)
        edp_efficiency_max = getattr(self, 'edp_efficiency_max', None)
        energy_min = getattr(self, 'energy_min', None)
        energy_max = getattr(self, 'energy_max', None)
        
        
        # Calculate headroom parameters for reward calculation (matching SimulatedEnergyLatencyEnv)
        temp_min = 40.0
        power_min = 10.0
        w_temp = 1
        w_power = 0
        
        # Calculate max_over values (matching measure_grpo_overhead.py defaults)
        slo_max_over = 1.5 * self.slo_target if self.slo_target > 0 else 15.0
        temp_max_over = 1.2 * self.max_temp
        power_max_over = 1.3 * self.max_power
        reward, performance_metric, constraint_penalty, \
        throughput_normalized, edp_efficiency_normalized, \
        inverse_energy_normalized, headroom_threshold_met, penalty_entropy, penalty_action_change = \
        calculate_reward(
            throughput_raw=throughput_raw,
            edp_efficiency_raw=edp_efficiency_raw,
            power_headroom=power_headroom,
            end_to_end_latency=end_to_end_latency,
            avg_power=total_power,
            avg_temp=temperature,
            slo_target=self.slo_target,
            max_temp=self.max_temp,
            max_power=self.max_power,
            throughput_min=throughput_min,
            throughput_max=throughput_max,
            edp_efficiency_min=edp_efficiency_min,
            edp_efficiency_max=edp_efficiency_max,
            energy_total=total_energy,
            energy_min=energy_min,
            energy_max=energy_max,
            temp_min=temp_min,
            power_min=power_min,
            slo_max_over=slo_max_over,
            temp_max_over=temp_max_over,
            power_max_over=power_max_over,
            w_temp=w_temp,
            w_power=w_power,
            entropy=avg_entropy,  # Pass entropy from inference results
            entropy_max=10.0,  # Maximum entropy value in the system (actual max is 10)
            lambda_entropy=1.5,  # Penalty coefficient for entropy violations (increased for better early exit)
            num_layers_used=early_exit_layer,  # Pass early exit layer for layer-dependent penalty
            max_layers=num_layers,  # Maximum layers in the model (from runner)
            curr_prefill=prefill_freq_bin,  # Current action: prefill frequency
            curr_decode=decode_freq_bin,  # Current action: decode frequency
            curr_batch=batch_size,  # Current action: batch size
            curr_layer=early_exit_layer,  # Current action: early exit layer
            prev_prefill=self.prev_prefill_bin,  # Previous action: prefill frequency (for hysteresis)
            prev_decode=self.prev_decode_bin,  # Previous action: decode frequency (for hysteresis)
            prev_batch=self.prev_batch_size,  # Previous action: batch size (for hysteresis)
            prev_layer=self.prev_early_exit_layer  # Previous action: early exit layer (for hysteresis)
        )
        
        # Update previous action tracking for next step
        self.prev_prefill_bin = prefill_freq_bin
        self.prev_decode_bin = decode_freq_bin
        self.prev_batch_size = batch_size
        self.prev_early_exit_layer = early_exit_layer
        # Check constraint violations
        slo_violation = end_to_end_latency > self.slo_target
        temp_violation = temperature > self.max_temp
        power_violation = total_power > self.max_power
        
        # Update state
        self.last_early_exit_layer = early_exit_layer
        self.prefill_freq = prefill_freq_bin
        self.decode_freq = decode_freq_bin
        self.batch_size = batch_size
        
        next_state = self._get_state()
        done = False
        info = {
            'energy': energy_total,
            'latency': end_to_end_latency,  # Total latency (alias for end_to_end_latency)
            'per_token_latency': per_token_latency,  # per-token latency
            'end_to_end_latency': end_to_end_latency,
            'prefill_latency': latency_prefill,
            'decode_latency': latency_decode,
            'tokens_prefill': tokens_prefill,  # Number of prefill tokens
            'tokens_decode': tokens_decode,  # Number of decode tokens
            'temperature': temperature,
            'power': total_power,
            'power_headroom': power_headroom,
            'slo_violation': slo_violation,
            'temp_violation': temp_violation,
            'power_violation': power_violation,
            'reward': reward,  # Store reward magnitude for debugging
            'penalty_entropy': penalty_entropy,  # Store entropy penalty magnitude for debugging
            'constraint_penalty': constraint_penalty,  # Store total constraint penalty for debugging
            'performance_metric': performance_metric,  # Store performance metric for debugging
            'throughput_raw': throughput_raw,
            'edp_efficiency_raw': edp_efficiency_raw,
            'throughput_normalized': throughput_normalized,
            'edp_efficiency_normalized': edp_efficiency_normalized,
            'inverse_energy_normalized': inverse_energy_normalized,
            'headroom_threshold_met': headroom_threshold_met,
            'throughput_min': throughput_min,
            'throughput_max': throughput_max,
            'edp_efficiency_min': edp_efficiency_min,
            'edp_efficiency_max': edp_efficiency_max,
            'energy_min': energy_min,
            'energy_max': energy_max,
            'avg_entropy': avg_entropy,  # Add entropy to info dict for tracking
            'early_exit_layer': getattr(self, 'early_exit_layer', early_exit_layer),  # Add early exit layer used during inference for tracking
        }
        
        return next_state, reward, done, info
    
    def get_action_description(self, action_idx):
        prefill_freq, decode_freq, batch, early_exit_layer = self.actions[action_idx]
        return f"Set prefill_freq={prefill_freq}, decode_freq={decode_freq}, batch={batch}, early_exit_layer={early_exit_layer}"


# ============================================================
# GRPO AGENT - Group Relative Policy Optimization
# ============================================================

class GRPOAgent:
    """
    GRPO Agent - Group Relative Policy Optimization
    
    Key differences from PPO:
    1. No value function/critic - advantages computed from group comparisons
    2. Samples multiple actions per state and compares within group
    3. Uses relative rewards within each group for advantage estimation
    4. Simpler architecture but requires more samples per update
    """
    
    def __init__(self, state_dim, action_dim, policy=None, lr=4e-4, 
                 group_size=8, epsilon_clip=0.2, epochs=10, 
                 entropy_coef=0.02, kl_coef=0.1, beta=0.1,
                 num_layers=4, freq_bins=11, max_batch=32, ema_alpha=None):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            policy: Pre-trained policy (optional)
            lr: Learning rate
            group_size: Number of samples per state for group comparison
            epsilon_clip: Clipping parameter (similar to PPO)
            epochs: Number of update epochs per batch
            entropy_coef: Entropy bonus coefficient
            kl_coef: KL penalty coefficient for reference policy
            beta: Temperature for advantage computation
            ema_alpha: Exponential moving average factor for action smoothing (0.0-1.0)
                      Higher values give more weight to recent actions
        """
        self.group_size = group_size
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.beta = beta
        self.ema_alpha = ema_alpha  # EMA smoothing factor for action probabilities
        
        if policy is not None:
            self.policy = policy
            # Get architecture parameters from the provided policy
            # Use the policy's action_dim instead of the environment's action_dim
            actual_action_dim = getattr(policy, 'action_dim', action_dim)
            # Try to detect num_layers from state_dict if not available as attribute
            if hasattr(policy, 'num_layers'):
                actual_num_layers = policy.num_layers
            else:
                # Count linear layers in features to infer num_layers
                # features has: Linear(state_dim, hidden_dim), ReLU, then (num_layers-1) × (Linear, ReLU)
                # So total linear layers = 1 + (num_layers - 1) = num_layers
                feature_layers = [m for m in policy.features if isinstance(m, nn.Linear)]
                actual_num_layers = len(feature_layers) if feature_layers else num_layers
            
            # Get parameters from policy if available
            actual_freq_bins = getattr(policy, 'freq_bins', freq_bins)
            actual_max_batch = getattr(policy, 'max_batch', max_batch)
        else:
            self.policy = GRPOPolicy(state_dim, action_dim, num_layers=num_layers,
                                    freq_bins=freq_bins, max_batch=max_batch)
            actual_action_dim = action_dim
            actual_num_layers = num_layers
            actual_freq_bins = freq_bins
            actual_max_batch = max_batch
        
        # Reference policy for KL constraint (frozen copy)
        # Must match the policy's architecture exactly, including action_dim
        self.ref_policy = GRPOPolicy(state_dim, actual_action_dim, num_layers=actual_num_layers,
                                    freq_bins=actual_freq_bins, max_batch=actual_max_batch)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory for batch collection
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'group_ids': []  # To track which samples belong to same group
        }
        self.current_group_id = 0
        
        # Action smoothing via exponential moving average
        self.smoothed_action_probs = None  # Will store EMA of action probabilities
        self.action_dim = action_dim
    
    def _apply_ema_smoothing(self, action_probs):
        """
        Apply exponential moving average smoothing to action probabilities.
        Smooths out jagged changes in action selection for more stable behavior.
        
        Args:
            action_probs: Current action probabilities (unnormalized logits or probabilities)
        
        Returns:
            Smoothed action probabilities
        """
        if self.smoothed_action_probs is None:
            # First time: initialize with current probabilities
            self.smoothed_action_probs = action_probs.clone().detach()
        else:
            # Apply EMA: smoothed = alpha * current + (1 - alpha) * previous
            self.smoothed_action_probs = (
                self.ema_alpha * action_probs.detach() + 
                (1 - self.ema_alpha) * self.smoothed_action_probs
            )
        
        # Normalize to ensure valid probability distribution
        smoothed_probs = torch.softmax(self.smoothed_action_probs, dim=-1)
        return smoothed_probs
    
    def select_action(self, state, deterministic=False, use_smoothing=False):
        """Select a single action.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            use_smoothing: Whether to apply EMA smoothing to action probabilities
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy.get_action_probs(state_tensor)
            
            # Apply smoothing if requested
            if use_smoothing:
                action_probs = self._apply_ema_smoothing(action_probs.squeeze())
                action_probs = action_probs.unsqueeze(0)
            
            action, log_prob = self.policy.act(state_tensor, deterministic)
        return action, log_prob
    
    def sample_group(self, state, use_smoothing=False):
        """
        Sample a group of actions for the same state.
        This is the key GRPO mechanism - comparing multiple samples.
        
        Args:
            state: Current state
            use_smoothing: Whether to apply EMA smoothing to action probabilities
        
        Returns:
            actions: List of sampled action indices
            log_probs: List of log probabilities for sampled actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        actions = []
        log_probs = []
        
        with torch.no_grad():
            # Get action probabilities
            action_probs = self.policy.get_action_probs(state_tensor)
            
            # Apply smoothing if requested
            if use_smoothing:
                action_probs = self._apply_ema_smoothing(action_probs.squeeze())
                action_probs = action_probs.unsqueeze(0)
            
            dist = Categorical(action_probs)
            
            for _ in range(self.group_size):
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # Ensure log_prob is a scalar tensor (squeeze if needed)
                if log_prob.dim() > 0:
                    log_prob = log_prob.squeeze()
                actions.append(action.item())
                log_probs.append(log_prob)
        
        return actions, log_probs
    
    def store_group_transitions(self, state, actions, log_probs, rewards):
        """Store a group of transitions."""
        for action, log_prob, reward in zip(actions, log_probs, rewards):
            self.memory['states'].append(state)
            self.memory['actions'].append(action)
            self.memory['log_probs'].append(log_prob)
            self.memory['rewards'].append(reward)
            self.memory['group_ids'].append(self.current_group_id)
        
        self.current_group_id += 1
    
    def store_transition(self, state, action, log_prob, reward, group_id=None):
        """Store a single transition (for compatibility)."""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['group_ids'].append(group_id if group_id is not None else self.current_group_id)
    
    def compute_group_advantages(self):
        """
        Compute advantages using group relative comparisons.
        This is the core GRPO innovation - advantages come from 
        comparing rewards within each group, not from a value function.
        """
        rewards = np.array(self.memory['rewards'])
        group_ids = np.array(self.memory['group_ids'])
        
        advantages = np.zeros_like(rewards)
        
        unique_groups = np.unique(group_ids)
        for group_id in unique_groups:
            mask = group_ids == group_id
            group_rewards = rewards[mask]
            
            # Normalize within group (group-relative advantage)
            if len(group_rewards) > 1:
                group_mean = np.mean(group_rewards)
                group_std = np.std(group_rewards) + 1e-8
                group_advantages = (group_rewards - group_mean) / group_std
            else:
                # Single sample in group - use global normalization
                group_advantages = np.array([0.0])
            
            advantages[mask] = group_advantages
        
        return advantages
    
    def update(self):
        """
        Update policy using GRPO objective.
        """
        if len(self.memory['states']) == 0:
            return 0.0
        
        # Compute group-relative advantages
        advantages = self.compute_group_advantages()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.stack(self.memory['log_probs']).detach()
        advantages = torch.FloatTensor(advantages)
        rewards = torch.FloatTensor(self.memory['rewards'])
        
        # Normalize advantages globally as well
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        for _ in range(self.epochs):
            # Get current policy log probs
            action_log_probs, action_probs = self.policy.get_log_probs(states, actions)
            
            # Get reference policy log probs for KL constraint
            with torch.no_grad():
                ref_log_probs, _ = self.ref_policy.get_log_probs(states, actions)
            
            # Policy ratio
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            # Standard GRPO policy loss (no conditional split)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty from reference policy (FIXED: correct formula)
            kl_penalty = (action_log_probs - ref_log_probs).mean()
            
            # Entropy bonus
            entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1))
            
            # Total loss
            loss = policy_loss + self.kl_coef * kl_penalty - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'group_ids': []
        }
        self.current_group_id = 0
        
        # Return detailed loss components for tracking
        return {
            'total_loss': total_loss / self.epochs,
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_penalty.item(),
            'entropy': entropy.item()
        }
    
    def update_reference_policy(self):
        """Update reference policy to current policy (for KL constraint)."""
        self.ref_policy.load_state_dict(self.policy.state_dict())
    
    def reset_action_smoothing(self):
        """Reset the action smoothing EMA state.
        
        Useful when starting a new episode or training phase.
        """
        self.smoothed_action_probs = None
    
    def get_smoothing_stats(self):
        """
        Get statistics about current action smoothing state.
        
        Returns:
            dict with smoothing statistics or None if no smoothing data
        """
        if self.smoothed_action_probs is None:
            return None
        
        with torch.no_grad():
            probs = torch.softmax(self.smoothed_action_probs, dim=-1)
            return {
                'max_prob': probs.max().item(),
                'min_prob': probs.min().item(),
                'mean_prob': probs.mean().item(),
                'entropy': -(probs * torch.log(probs + 1e-10)).sum().item()
            }


# ============================================================
# INFERENCE CLASS FOR GRPO POLICY
# ============================================================

# class GRPOInference:
#     """Inference wrapper for trained GRPO policy."""
    
#     def __init__(self, policy, freq_bins=11, max_batch=1):
#         """
#         Args:
#             policy: Trained GRPOPolicy model
#             freq_bins: Number of frequency bins
#             max_batch: Maximum batch size
#         """
#         self.policy = policy
#         self.policy.eval()
#         self.freq_bins = freq_bins
#         self.max_batch = max_batch
        
#         # GPU frequency bins (Hz) - use shared constant
#         self.GPU_FREQ_BINS = GPU_FREQ_BINS
        
#         # Batch sizes: Must match OfflineDataset and environment action space!
#         # 1 to 32 (inclusive)
#         self.batch_sizes = list(range(1, max_batch + 1))  # 1, 2, 3, ..., 32
#         self.num_batch_sizes = len(self.batch_sizes)
        
#         # Build action mapping: Must match OfflineDataset structure!
#         # freq_bins represents the number of bins, so range(0, freq_bins) gives indices 0 to (freq_bins-1)
#         self.actions = []
#         for prefill_freq in range(0, freq_bins):  # 0 to (freq_bins-1) inclusive
#             for decode_freq in range(0, freq_bins):  # 0 to (freq_bins-1) inclusive
#                 for batch in self.batch_sizes:  # 1 to 32 (inclusive)
#                     self.actions.append((prefill_freq, decode_freq, batch))
    
#     def get_optimal_action(self, state=None):
#         """
#         Get the optimal action (highest probability) for a given state.
        
#         Args:
#             state: Optional state [prefill_freq_norm, decode_freq_norm, batch_norm, temp_headroom]
#                    If None, uses default state [0.5, 0.5, 0.5, 0.3]
        
#         Returns:
#             dict with keys: prefill_freq_bin, prefill_freq_hz, prefill_freq_mhz,
#                            decode_freq_bin, decode_freq_hz, decode_freq_mhz,
#                            batch_size, action_probability
#         """
#         if state is None:
#             state = np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32)  # Default temp_headroom=0.3
        
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
#         with torch.no_grad():
#             action_probs = self.policy.get_action_probs(state_tensor)
#             action_idx = torch.argmax(action_probs, dim=-1).item()
#             action_prob = action_probs[0, action_idx].item()
        
#         prefill_freq, decode_freq, batch = self.actions[action_idx]
        
#         return {
#             'prefill_freq_bin': prefill_freq,
#             'prefill_freq_hz': self.GPU_FREQ_BINS[prefill_freq],
#             'prefill_freq_mhz': self.GPU_FREQ_BINS[prefill_freq] / 1e6,
#             'decode_freq_bin': decode_freq,
#             'decode_freq_hz': self.GPU_FREQ_BINS[decode_freq],
#             'decode_freq_mhz': self.GPU_FREQ_BINS[decode_freq] / 1e6,
#             'batch_size': batch,
#             'action_probability': action_prob
#         }
    
#     def get_top_k_actions(self, state=None, top_k=5):
#         """
#         Get top-k actions with highest probabilities.
        
#         Args:
#             state: Optional state
#             top_k: Number of top actions to return
        
#         Returns:
#             List of dicts with action information
#         """
#         if state is None:
#             state = np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32)  # Default temp_headroom=0.3
        
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
#         with torch.no_grad():
#             action_probs = self.policy.get_action_probs(state_tensor)
#             top_probs, top_indices = torch.topk(action_probs.squeeze(), top_k)
        
#         results = []
#         for prob, idx in zip(top_probs, top_indices):
#             prefill_freq, decode_freq, batch = self.actions[idx.item()]
#             results.append({
#                 'prefill_freq_bin': prefill_freq,
#                 'prefill_freq_hz': self.GPU_FREQ_BINS[prefill_freq],
#                 'prefill_freq_mhz': self.GPU_FREQ_BINS[prefill_freq] / 1e6,
#                 'decode_freq_bin': decode_freq,
#                 'decode_freq_hz': self.GPU_FREQ_BINS[decode_freq],
#                 'decode_freq_mhz': self.GPU_FREQ_BINS[decode_freq] / 1e6,
#                 'batch_size': batch,
#                 'action_probability': prob.item()
#             })
        
#         return results
    
#     def print_recommendation(self, state=None, slo_target=None):
#         """Print and return optimal recommendation."""
#         if state is None:
#             state = np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32)  # Default temp_headroom=0.3
        
#         # Show what state is being used
#         freq_bins = 11  # Should match self.freq_bins
#         batch_sizes = self.batch_sizes #list(range(1, 2))  # Should match self.batch_sizes (only batch size 1)
        
#         # Decode state to human-readable format
#         prefill_bin_approx = int(state[0] * (freq_bins - 1))
#         decode_bin_approx = int(state[1] * (freq_bins - 1))
#         batch_idx_approx = int(state[2] * (len(batch_sizes) - 1))
#         batch_size_approx = batch_sizes[batch_idx_approx] if batch_idx_approx < len(batch_sizes) else batch_sizes[-1]
        
#         optimal = self.get_optimal_action(state)
        
#         print(f"\n>>> State used for prediction:")
#         print(f"    State vector: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}, {state[3]:.2f}]")
#         temp_headroom_pct = state[3] * 100
#         print(f"    Corresponds to: Prefill~bin{prefill_bin_approx}, Decode~bin{decode_bin_approx}, Batch~{batch_size_approx}, TempHeadroom~{temp_headroom_pct:.0f}%")
#         print(f"\nOptimal Configuration (for this state):")
#         print(f"  Prefill Frequency: {optimal['prefill_freq_mhz']:.0f} MHz (bin {optimal['prefill_freq_bin']})")
#         print(f"  Decode Frequency:  {optimal['decode_freq_mhz']:.0f} MHz (bin {optimal['decode_freq_bin']})")
#         print(f"  Batch Size:        {optimal['batch_size']}")
#         print(f"  Confidence:        {optimal['action_probability']*100:.2f}%")
        
#         # Show top predictions to understand model behavior
#         top_k = self.get_top_k_actions(state, top_k=3)
#         print(f"\n  Top 3 predictions for this state:")
#         for i, action in enumerate(top_k, 1):
#             print(f"    {i}. Prefill={action['prefill_freq_bin']}, Decode={action['decode_freq_bin']}, "
#                   f"Batch={action['batch_size']}, Prob={action['action_probability']*100:.2f}%")
        
#         if slo_target:
#             print(f"\nSLO Target: {slo_target:.3f}s for end-to-end latency")
#             print(f"Note: Use this configuration to meet chatbot response time requirements")
#         return optimal


# def get_grpo_recommendation(model_path, state=None, freq_bins=11, max_batch=1,
#                             num_layers=2):
#     """
#     Standalone function to load a saved GRPO model and get recommendations.
    
#     Args:
#         model_path: Path to saved model (.pth file)
#         state: Optional state for inference
#         freq_bins: Number of frequency bins (default: 11)
#         max_batch: Maximum batch size (default: 16)
    
#     Returns:
#         dict with optimal action information
    
#     Example:
#         >>> optimal = get_grpo_recommendation('grpo_dual_freq_model.pth')
#         >>> print(f"Use prefill freq: {optimal['prefill_freq_mhz']:.0f} MHz")
#         >>> print(f"Use decode freq: {optimal['decode_freq_mhz']:.0f} MHz")
#         >>> print(f"Use batch size: {optimal['batch_size']}")
#     """
#     # Calculate dimensions - must match OfflineDataset action space structure!
#     state_dim = 4  # prefill_freq, decode_freq, batch_size, temperature_headroom
#     # Batch sizes: [1] only for max_batch=1
#     num_batch_sizes = 1  # Only batch size 1 for max_batch=1
#     action_dim = freq_bins * freq_bins * num_batch_sizes  # 11 * 11 * 17 = 2057
    
#     # Load policy
#     policy = GRPOPolicy(state_dim, action_dim, num_layers=num_layers,
#                        freq_bins=freq_bins, max_batch=max_batch)
#     policy.load_state_dict(torch.load(model_path))
#     policy.eval()
    
#     # Create inference object
#     inference = GRPOInference(policy, freq_bins=freq_bins, max_batch=max_batch)
    
#     # Get optimal action
#     return inference.get_optimal_action(state)


# ============================================================
# PLOTTING FUNCTIONS FOR TRAINING METRICS
# ============================================================

def _plot_training_metrics(entropies, latencies, num_layers, current_episode, save_path='plots/early_exit_training_metrics.png',
                          rewards=None, entropy_penalties=None, temperatures=None, max_temp=None,
                          prefill_freq=None, decode_freq=None, batch_size=None, per_token_latency=None, decode_latency=None, decode_throughput=None):
    """
    Plot entropy, latency, number of layers, temperature, reward magnitude, entropy penalty, 
    prefill_freq, decode_freq, batch_size, per-token latency, decode latency, and decode throughput over time during training.
    
    Args:
        entropies: List of entropy values (can contain None values)
        latencies: List of latency values (end-to-end latency in seconds)
        num_layers: List of number of layers used
        current_episode: Current episode number (for title)
        save_path: Path to save the plot
        rewards: List of reward values (optional, for debugging)
        entropy_penalties: List of entropy penalty values (optional, for debugging)
        temperatures: List of temperature values (optional, can contain None values)
        max_temp: Maximum temperature threshold (optional, for plotting threshold line)
        prefill_freq: List of prefill frequency values (optional, can contain None values)
        decode_freq: List of decode frequency values (optional, can contain None values)
        batch_size: List of batch size values (optional, can contain None values)
        per_token_latency: List of per-token latency values (optional, can contain None values)
        decode_latency: List of decode latency values (optional, can contain None values)
        decode_throughput: List of decode throughput values in tokens/second (optional, can contain None values)
    """
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
    # Filter out None values for entropy
    valid_entropies = [(i, e) for i, e in enumerate(entropies) if e is not None]
    entropy_indices = [i for i, e in valid_entropies]
    entropy_values = [e for i, e in valid_entropies]
    
    # Determine number of subplots based on available data
    num_plots = 3  # Base: entropy, latency, layers
    if temperatures is not None:
        num_plots += 1
    if rewards is not None:
        num_plots += 1
    if entropy_penalties is not None:
        num_plots += 1
    if prefill_freq is not None:
        num_plots += 1
    if decode_freq is not None:
        num_plots += 1
    if batch_size is not None:
        num_plots += 1
    if per_token_latency is not None:
        num_plots += 1
    if decode_latency is not None:
        num_plots += 1
    if decode_throughput is not None:
        num_plots += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    # Episode numbers should start at 1, not 0
    episode_display = current_episode if current_episode > 0 else 1
    fig.suptitle(f'Training Metrics Over Time (Episode {episode_display})', fontsize=14, fontweight='bold', y=0.995)
    
    # Plot 1: Entropy over time
    if entropy_values:
        axes[0].plot(entropy_indices, entropy_values, 'b-', alpha=0.6, linewidth=1)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Entropy', fontweight='bold')
        axes[0].set_title('Model Entropy (Lower = More Confident/Better Accuracy)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        # Optional: Add entropy threshold line (70% of max entropy = 7.0, or use a custom threshold)
        # entropy_threshold = 7.0  # 70% of entropy_max (10.0)
        # axes[0].axhline(y=entropy_threshold, color='r', linestyle='--', linewidth=1, label=f'High Entropy Threshold ({entropy_threshold})')
        # axes[0].legend()
        # Add moving average
        if len(entropy_values) > 10:
            window = min(50, len(entropy_values) // 10)
            moving_avg = np.convolve(entropy_values, np.ones(window)/window, mode='valid')
            axes[0].plot(entropy_indices[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (w={window})')
            axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No entropy data available', ha='center', va='center', 
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('Model Entropy', fontweight='bold')
    
    # Plot 2: Latency over time
    if latencies:
        axes[1].plot(latencies, 'g-', alpha=0.6, linewidth=1)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('End-to-End Latency (s)', fontweight='bold')
        axes[1].set_title('Latency Over Time (Lower is Better)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        # Add moving average
        if len(latencies) > 10:
            window = min(50, len(latencies) // 10)
            moving_avg = np.convolve(latencies, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(latencies)), moving_avg, 'orange', linewidth=2, 
                        label=f'Moving Avg (w={window})')
            axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No latency data available', ha='center', va='center', 
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Latency Over Time', fontweight='bold')
    
    # Plot 3: Number of layers over time
    plot_idx = 2
    if num_layers:
        axes[plot_idx].plot(num_layers, 'm-', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Early Exit Layer')
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Number of Layers', fontweight='bold')
        axes[plot_idx].set_title('Early Exit Layers Used (More Layers = Better Accuracy, Higher Latency)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_ylim([3, 17])  # Expected range for layers 4-16
        axes[plot_idx].set_yticks(range(4, 17))
        # Add moving average
        if len(num_layers) > 10:
            window = min(50, len(num_layers) // 10)
            moving_avg = np.convolve(num_layers, np.ones(window)/window, mode='valid')
            axes[plot_idx].plot(range(window-1, len(num_layers)), moving_avg, 'orange', linewidth=2, 
                        label=f'Moving Avg (w={window})')
        axes[plot_idx].legend()
    else:
        axes[plot_idx].text(0.5, 0.5, 'No layer data available', ha='center', va='center', 
                    transform=axes[plot_idx].transAxes, fontsize=12)
        axes[plot_idx].set_title('Number of Layers Over Time', fontweight='bold')
    
    # Plot 4: Temperature over time (if available)
    if temperatures is not None:
        plot_idx += 1
        # Filter out None values for temperature
        valid_temperatures = [(i, t) for i, t in enumerate(temperatures) if t is not None]
        temp_indices = [i for i, t in valid_temperatures]
        temp_values = [t for i, t in valid_temperatures]
        
        if temp_values:
            axes[plot_idx].plot(temp_indices, temp_values, 'orange', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Device Temperature')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Temperature (°C)', fontweight='bold')
            axes[plot_idx].set_title('Device Temperature Over Time', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add max temperature threshold line if provided
            if max_temp is not None:
                axes[plot_idx].axhline(y=max_temp, color='r', linestyle='--', linewidth=2, label=f'Max Temp ({max_temp}°C)')
            # Add moving average
            if len(temp_values) > 10:
                window = min(50, len(temp_values) // 10)
                moving_avg = np.convolve(temp_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(temp_indices[window-1:], moving_avg, 'r-', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No temperature data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Temperature Over Time', fontweight='bold')
    
    # Plot 5: Reward magnitude (if available)
    if rewards is not None:
        plot_idx += 1
        axes[plot_idx].plot(rewards, 'g-', alpha=0.6, linewidth=1, marker='o', markersize=2)
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Reward Magnitude', fontweight='bold')
        axes[plot_idx].set_title('Reward Over Time (Higher is Better)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero Reward')
        # Add moving average
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[plot_idx].plot(range(window-1, len(rewards)), moving_avg, 'orange', linewidth=2, 
                        label=f'Moving Avg (w={window})')
        axes[plot_idx].legend()
    
    # Plot 6: Entropy penalty magnitude (if available)
    if entropy_penalties is not None:
        plot_idx += 1
        axes[plot_idx].plot(entropy_penalties, 'r-', alpha=0.6, linewidth=1, marker='o', markersize=2)
        axes[plot_idx].set_xlabel('Step')
        axes[plot_idx].set_ylabel('Entropy Penalty Magnitude', fontweight='bold')
        axes[plot_idx].set_title('Entropy Penalty Over Time (Lower is Better)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        # Add moving average
        if len(entropy_penalties) > 10:
            window = min(50, len(entropy_penalties) // 10)
            moving_avg = np.convolve(entropy_penalties, np.ones(window)/window, mode='valid')
            axes[plot_idx].plot(range(window-1, len(entropy_penalties)), moving_avg, 'orange', linewidth=2, 
                        label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
    
    # Plot 7: Prefill frequency over time (if available)
    if prefill_freq is not None:
        plot_idx += 1
        # Filter out None values for prefill_freq
        valid_prefill_freq = [(i, f) for i, f in enumerate(prefill_freq) if f is not None]
        prefill_indices = [i for i, f in valid_prefill_freq]
        prefill_values = [f for i, f in valid_prefill_freq]
        
        if prefill_values:
            axes[plot_idx].plot(prefill_indices, prefill_values, 'c-', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Prefill Frequency')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Prefill Frequency (bin)', fontweight='bold')
            axes[plot_idx].set_title('Prefill Frequency Over Time', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(prefill_values) > 10:
                window = min(50, len(prefill_values) // 10)
                moving_avg = np.convolve(prefill_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(prefill_indices[window-1:], moving_avg, 'orange', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No prefill frequency data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Prefill Frequency Over Time', fontweight='bold')
    
    # Plot 8: Decode frequency over time (if available)
    if decode_freq is not None:
        plot_idx += 1
        # Filter out None values for decode_freq
        valid_decode_freq = [(i, f) for i, f in enumerate(decode_freq) if f is not None]
        decode_indices = [i for i, f in valid_decode_freq]
        decode_values = [f for i, f in valid_decode_freq]
        
        if decode_values:
            axes[plot_idx].plot(decode_indices, decode_values, 'y-', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Decode Frequency')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Decode Frequency (bin)', fontweight='bold')
            axes[plot_idx].set_title('Decode Frequency Over Time', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(decode_values) > 10:
                window = min(50, len(decode_values) // 10)
                moving_avg = np.convolve(decode_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(decode_indices[window-1:], moving_avg, 'orange', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No decode frequency data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Decode Frequency Over Time', fontweight='bold')
    
    # Plot 9: Batch size over time (if available)
    if batch_size is not None:
        plot_idx += 1
        # Filter out None values for batch_size
        valid_batch_size = [(i, b) for i, b in enumerate(batch_size) if b is not None]
        batch_indices = [i for i, b in valid_batch_size]
        batch_values = [b for i, b in valid_batch_size]
        
        if batch_values:
            axes[plot_idx].plot(batch_indices, batch_values, 'purple', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Batch Size')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Batch Size', fontweight='bold')
            axes[plot_idx].set_title('Batch Size Over Time', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(batch_values) > 10:
                window = min(50, len(batch_values) // 10)
                moving_avg = np.convolve(batch_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(batch_indices[window-1:], moving_avg, 'orange', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No batch size data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Batch Size Over Time', fontweight='bold')
    
    # Plot 10: Per-token latency over time (if available)
    if per_token_latency is not None:
        plot_idx += 1
        # Filter out None values for per_token_latency
        valid_per_token_latency = [(i, l) for i, l in enumerate(per_token_latency) if l is not None]
        latency_indices = [i for i, l in valid_per_token_latency]
        latency_values = [l for i, l in valid_per_token_latency]
        
        if latency_values:
            axes[plot_idx].plot(latency_indices, latency_values, 'brown', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Per-Token Latency')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Latency (s/token)', fontweight='bold')
            axes[plot_idx].set_title('Average Token Latency Over Time (Lower is Better)', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(latency_values) > 10:
                window = min(50, len(latency_values) // 10)
                moving_avg = np.convolve(latency_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(latency_indices[window-1:], moving_avg, 'orange', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No per-token latency data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Average Token Latency Over Time', fontweight='bold')
    
    # Plot 11: Decode latency over time (if available)
    if decode_latency is not None:
        plot_idx += 1
        # Filter out None values for decode_latency
        valid_decode_latency = [(i, l) for i, l in enumerate(decode_latency) if l is not None]
        decode_latency_indices = [i for i, l in valid_decode_latency]
        decode_latency_values = [l for i, l in valid_decode_latency]
        
        if decode_latency_values:
            axes[plot_idx].plot(decode_latency_indices, decode_latency_values, 'teal', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Decode Latency')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Decode Latency (s)', fontweight='bold')
            axes[plot_idx].set_title('Decode Latency Over Time (Lower is Better)', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(decode_latency_values) > 10:
                window = min(50, len(decode_latency_values) // 10)
                moving_avg = np.convolve(decode_latency_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(decode_latency_indices[window-1:], moving_avg, 'orange', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No decode latency data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Decode Latency Over Time', fontweight='bold')
    
    # Plot 12: Decode throughput over time (if available)
    if decode_throughput is not None:
        plot_idx += 1
        # Filter out None values for decode_throughput
        valid_decode_throughput = [(i, t) for i, t in enumerate(decode_throughput) if t is not None]
        decode_throughput_indices = [i for i, t in valid_decode_throughput]
        decode_throughput_values = [t for i, t in valid_decode_throughput]
        
        if decode_throughput_values:
            axes[plot_idx].plot(decode_throughput_indices, decode_throughput_values, 'darkgreen', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Decode Throughput')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Decode Throughput (tokens/s)', fontweight='bold')
            axes[plot_idx].set_title('Decode Throughput Over Time (Higher is Better)', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3)
            # Add moving average
            if len(decode_throughput_values) > 10:
                window = min(50, len(decode_throughput_values) // 10)
                moving_avg = np.convolve(decode_throughput_values, np.ones(window)/window, mode='valid')
                axes[plot_idx].plot(decode_throughput_indices[window-1:], moving_avg, 'red', linewidth=2, 
                            label=f'Moving Avg (w={window})')
            axes[plot_idx].legend()
        else:
            axes[plot_idx].text(0.5, 0.5, 'No decode throughput data available', ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=12)
            axes[plot_idx].set_title('Decode Throughput Over Time', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining metrics plot saved to: {save_path}")
    plt.close()
    
    # Save data to CSV for later replotting
    # Determine log file path based on save_path
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Extract filename from save_path and create corresponding CSV filename
    plot_filename = os.path.basename(save_path)
    csv_filename = plot_filename.replace('.png', '.csv')
    csv_path = os.path.join(log_dir, csv_filename)
    
    # Prepare data for CSV - align all arrays to same length (max length)
    max_len = max(
        len(entropies) if entropies else 0,
        len(latencies) if latencies else 0,
        len(num_layers) if num_layers else 0,
        len(temperatures) if temperatures else 0,
        len(rewards) if rewards else 0,
        len(entropy_penalties) if entropy_penalties else 0,
        len(prefill_freq) if prefill_freq else 0,
        len(decode_freq) if decode_freq else 0,
        len(batch_size) if batch_size else 0,
        len(per_token_latency) if per_token_latency else 0,
        len(decode_latency) if decode_latency else 0
    )
    
    # Create DataFrame with all data
    data_dict = {
        'step': list(range(max_len)),
        'episode': [episode_display] * max_len,
        'entropy': [entropies[i] if i < len(entropies) and entropies[i] is not None else None for i in range(max_len)],
        'latency': [latencies[i] if i < len(latencies) else None for i in range(max_len)],
        'num_layers': [num_layers[i] if i < len(num_layers) else None for i in range(max_len)],
    }
    
    # Add optional columns
    if temperatures is not None:
        data_dict['temperature'] = [temperatures[i] if i < len(temperatures) and temperatures[i] is not None else None for i in range(max_len)]
    else:
        data_dict['temperature'] = [None] * max_len
    
    if rewards is not None:
        data_dict['reward'] = [rewards[i] if i < len(rewards) else None for i in range(max_len)]
    else:
        data_dict['reward'] = [None] * max_len
    
    if entropy_penalties is not None:
        data_dict['entropy_penalty'] = [entropy_penalties[i] if i < len(entropy_penalties) else None for i in range(max_len)]
    else:
        data_dict['entropy_penalty'] = [None] * max_len
    
    if prefill_freq is not None:
        data_dict['prefill_freq'] = [prefill_freq[i] if i < len(prefill_freq) and prefill_freq[i] is not None else None for i in range(max_len)]
    else:
        data_dict['prefill_freq'] = [None] * max_len
    
    if decode_freq is not None:
        data_dict['decode_freq'] = [decode_freq[i] if i < len(decode_freq) and decode_freq[i] is not None else None for i in range(max_len)]
    else:
        data_dict['decode_freq'] = [None] * max_len
    
    if batch_size is not None:
        data_dict['batch_size'] = [batch_size[i] if i < len(batch_size) and batch_size[i] is not None else None for i in range(max_len)]
    else:
        data_dict['batch_size'] = [None] * max_len
    
    if per_token_latency is not None:
        data_dict['per_token_latency'] = [per_token_latency[i] if i < len(per_token_latency) and per_token_latency[i] is not None else None for i in range(max_len)]
    else:
        data_dict['per_token_latency'] = [None] * max_len
    
    if decode_latency is not None:
        data_dict['decode_latency'] = [decode_latency[i] if i < len(decode_latency) and decode_latency[i] is not None else None for i in range(max_len)]
    else:
        data_dict['decode_latency'] = [None] * max_len
    
    # Add max_temp if provided
    if max_temp is not None:
        data_dict['max_temp'] = [max_temp] * max_len
    else:
        data_dict['max_temp'] = [None] * max_len
    
    # Save to CSV
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_path, index=False)
    print(f"Training metrics data saved to: {csv_path}")


# ============================================================
# PLOTTING POLICY METRICS (LOSS, KL DIVERGENCE, ENTROPY)
# ============================================================

def _plot_policy_metrics(policy_losses, kl_divergences, policy_entropies, step_entropies,
                        current_episode=None, save_path='plots/policy_metrics.png'):
    """
    Plot policy loss, KL divergence, policy entropy, and step-wise entropy over time.
    
    Args:
        policy_losses: List of policy loss values
        kl_divergences: List of KL divergence values
        policy_entropies: List of entropy values from policy updates
        step_entropies: List of entropy values at each environment step
        current_episode: Current episode number (for title)
        save_path: Path to save the plot
    """
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Determine episode display number
    episode_display = current_episode if current_episode and current_episode > 0 else 1
    fig.suptitle(f'Policy Training Metrics (Episode {episode_display})', fontsize=14, fontweight='bold', y=0.995)
    
    # Plot 1: Policy Loss
    if policy_losses:
        axes[0, 0].plot(policy_losses, 'b-', alpha=0.6, linewidth=1.5, label='Policy Loss')
        axes[0, 0].set_xlabel('Update Step', fontweight='bold')
        axes[0, 0].set_ylabel('Policy Loss', fontweight='bold')
        axes[0, 0].set_title('Policy Loss Over Training (Lower is Better)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving average if enough data
        if len(policy_losses) > 10:
            window = min(20, len(policy_losses) // 5)
            moving_avg = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(policy_losses)), moving_avg, 'orange', 
                          linewidth=2, label=f'Moving Avg (w={window})')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No policy loss data available', ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Policy Loss Over Training', fontweight='bold')
    
    # Plot 2: KL Divergence
    if kl_divergences:
        axes[0, 1].plot(kl_divergences, 'r-', alpha=0.6, linewidth=1.5, label='KL Divergence')
        axes[0, 1].set_xlabel('Update Step', fontweight='bold')
        axes[0, 1].set_ylabel('KL Divergence', fontweight='bold')
        axes[0, 1].set_title('KL Divergence from Reference Policy (Lower is Better)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add moving average if enough data
        if len(kl_divergences) > 10:
            window = min(20, len(kl_divergences) // 5)
            moving_avg = np.convolve(kl_divergences, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(kl_divergences)), moving_avg, 'orange', 
                          linewidth=2, label=f'Moving Avg (w={window})')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No KL divergence data available', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('KL Divergence from Reference Policy', fontweight='bold')
    
    # Plot 3: Policy Entropy (from updates)
    if policy_entropies:
        axes[1, 0].plot(policy_entropies, 'g-', alpha=0.6, linewidth=1.5, label='Policy Update Entropy')
        axes[1, 0].set_xlabel('Update Step', fontweight='bold')
        axes[1, 0].set_ylabel('Entropy', fontweight='bold')
        axes[1, 0].set_title('Policy Entropy During Training (Higher = More Diverse)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add moving average if enough data
        if len(policy_entropies) > 10:
            window = min(20, len(policy_entropies) // 5)
            moving_avg = np.convolve(policy_entropies, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(policy_entropies)), moving_avg, 'orange', 
                          linewidth=2, label=f'Moving Avg (w={window})')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No policy entropy data available', ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Policy Entropy During Training', fontweight='bold')
    
    # Plot 4: Step-wise Entropy (from environment)
    if step_entropies:
        # Filter out None values
        valid_entropies = [(i, e) for i, e in enumerate(step_entropies) if e is not None]
        entropy_indices = [i for i, e in valid_entropies]
        entropy_values = [e for i, e in valid_entropies]
        
        if entropy_values:
            axes[1, 1].plot(entropy_indices, entropy_values, 'purple', alpha=0.6, linewidth=1, 
                          label='Step Entropy')
            axes[1, 1].set_xlabel('Environment Step', fontweight='bold')
            axes[1, 1].set_ylabel('Model Entropy', fontweight='bold')
            axes[1, 1].set_title('Model Entropy at Each Environment Step (Lower = More Confident)', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add moving average if enough data
            if len(entropy_values) > 50:
                window = min(100, len(entropy_values) // 10)
                moving_avg = np.convolve(entropy_values, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(window-1, len(entropy_values)), moving_avg, 'orange', 
                              linewidth=2, label=f'Moving Avg (w={window})')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No step entropy data available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Model Entropy at Each Step', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No step entropy data available', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Entropy at Each Step', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPolicy metrics plot saved to: {save_path}")
    plt.close()
    
    # Save data to CSV
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    plot_filename = os.path.basename(save_path)
    csv_filename = plot_filename.replace('.png', '.csv')
    csv_path = os.path.join(log_dir, csv_filename)
    
    # Prepare data for CSV - separate tables for policy updates and environment steps
    # Policy loss/KL/entropy are from update steps (fewer data points)
    policy_data_len = max(
        len(policy_losses) if policy_losses else 0,
        len(kl_divergences) if kl_divergences else 0,
        len(policy_entropies) if policy_entropies else 0
    )
    
    # Pad policy metrics to same length
    policy_loss_padded = list(policy_losses) if policy_losses else []
    kl_div_padded = list(kl_divergences) if kl_divergences else []
    policy_ent_padded = list(policy_entropies) if policy_entropies else []
    
    # Pad to max length
    while len(policy_loss_padded) < policy_data_len:
        policy_loss_padded.append(None)
    while len(kl_div_padded) < policy_data_len:
        kl_div_padded.append(None)
    while len(policy_ent_padded) < policy_data_len:
        policy_ent_padded.append(None)
    
    # Create policy metrics dataframe
    if policy_data_len > 0:
        policy_dict = {
            'update_step': list(range(policy_data_len)),
            'policy_loss': policy_loss_padded,
            'kl_divergence': kl_div_padded,
            'policy_entropy': policy_ent_padded,
        }
        df_policy = pd.DataFrame(policy_dict)
    else:
        df_policy = pd.DataFrame()
    
    # Create step entropy dataframe
    if step_entropies:
        valid_entropies = [(i, e) for i, e in enumerate(step_entropies) if e is not None]
        step_dict = {
            'env_step': [i for i, e in valid_entropies],
            'step_entropy': [e for i, e in valid_entropies],
        }
        df_steps = pd.DataFrame(step_dict)
    else:
        df_steps = pd.DataFrame()
    
    # Save policy metrics
    if not df_policy.empty:
        df_policy.to_csv(csv_path, index=False)
        print(f"Policy metrics data saved to: {csv_path}")
    elif not df_steps.empty:
        df_steps.to_csv(csv_path, index=False)
        print(f"Step entropy data saved to: {csv_path}")
    else:
        print(f"No data to save to: {csv_path}")


# ============================================================
# CONTINUE TRAINING FROM CHECKPOINT WITH NEW REWARD FUNCTION
# ============================================================

def train_with_grpo_from_checkpoint(checkpoint_path, csv_path, num_episodes=500, max_steps=50,
                                    update_freq=200,
                                    group_size=8, use_simulated_env=False, alpha=0.8,
                                    slo_target=15.0, max_temp=95.0, min_temp=40.0, max_power=15.0,
                                    num_layers=2):
    """Continue training a GRPO agent from a saved checkpoint with a potentially different reward function.
    
    This function allows you to:
    1. Load a previously trained model checkpoint
    2. Continue training with the same or different reward function parameters
    3. Fine-tune on new data or with different constraints
    
    Args:
        checkpoint_path: Path to saved model checkpoint (.pth file)
        csv_path: Path to CSV training data
        num_episodes: Number of episodes to train
        max_steps: Number of steps per episode
        update_freq: Frequency of policy updates
        group_size: GRPO group size
        use_simulated_env: Use simulated env (True) or real Jetson (False)
        alpha: Weight for energy vs latency
        slo_target: SLO target for end-to-end latency
        max_temp: Maximum temperature constraint
        min_temp: Minimum temperature for normalization
        max_power: Maximum power constraint
        num_layers: Number of layers in policy network
    
    Returns:
        Same as train_with_grpo (agent, rewards, energies, etc.)
    """
    
    freq_bins = 11  # For 11×11×6 = 726 actions base
    max_batch = 31  # Batch range with step of 6
    state_dim = 5  # prefill_freq, decode_freq, batch_size, layers, temperature_headroom
    
    # Calculate action dimension (batch sizes: [1, 7, 13, 19, 25, 31] with step of 6)
    num_batch_sizes = len(list(range(1, 32, 6)))  # range(1, 32, 6) gives [1, 7, 13, 19, 25, 31]
    num_early_exit_layers = 8  # Layers 9, 10, 11, 12, 13, 14, 15, 16 = 8 options
    action_dim = freq_bins * freq_bins * num_batch_sizes * num_early_exit_layers
    print(f"Action dimension: {freq_bins} x {freq_bins} x {num_batch_sizes} x {num_early_exit_layers} = {action_dim}")
    
    print("="*60)
    print("LOADING CHECKPOINT FOR CONTINUED TRAINING")
    print("="*60)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the saved policy
    pretrained_policy = GRPOPolicy(state_dim, action_dim, num_layers=num_layers,
                                   freq_bins=freq_bins, max_batch=max_batch)
    try:
        pretrained_policy.load_state_dict(torch.load(checkpoint_path))
        print(f"✓ Successfully loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"  Please ensure the file exists and is a valid .pth file")
        return None
    except Exception as e:
        print(f"✗ ERROR: Failed to load checkpoint: {e}")
        return None
    
    print("\n" + "="*60)
    print("PHASE 2: CONTINUED TRAINING WITH POTENTIALLY NEW REWARD FUNCTION")
    print("="*60)
    
    # Load the offline dataset to initialize normalization ranges
    print(f"Loading offline dataset from: {csv_path}")
    dataset = OfflineDataset(csv_path, freq_bins=freq_bins, max_batch=max_batch, alpha=alpha,
                             slo_target=slo_target, max_temp=max_temp, max_power=max_power)
    
    if use_simulated_env:
        print("Using SIMULATED environment (CSV data with noise)")
        print("Power source: CSV statistics (mean_power with 10% noise)")
        env = SimulatedEnergyLatencyEnv(csv_path, freq_bins=freq_bins, max_batch=max_batch, alpha=alpha,
                                        slo_target=slo_target, max_temp=max_temp, max_power=max_power)
    else:
        print("Using REAL Jetson environment (actual hardware sensors)")
        print("Power source: tegrastats (VDD_GPU_SOC + VDD_CPU_CV + VDDQ_VDD2_1V8AO + VIN_SYS_5V0)")
        env = RealJetsonEnv(freq_bins=freq_bins, max_batch=max_batch,
                           slo_target=slo_target, max_temp=max_temp, min_temp=min_temp, max_power=max_power)
    
    # Initialize normalization ranges from offline dataset
    env.throughput_min = getattr(dataset, 'throughput_min', 0) or 0
    env.throughput_max = getattr(dataset, 'throughput_max', 1) or 1
    env.edp_efficiency_min = getattr(dataset, 'edp_efficiency_min', 0) or 0
    env.edp_efficiency_max = getattr(dataset, 'edp_efficiency_max', 1) or 1
    env.energy_min = getattr(dataset, 'energy_min', None)
    env.energy_max = getattr(dataset, 'energy_max', None)
    env._track_normalization = True  # Enable automatic range updates
    
    print(f"\n[Continued Training] Normalization ranges (from offline dataset):")
    print(f"  Throughput: [{env.throughput_min:.4f}, {env.throughput_max:.4f}] tokens/sec")
    print(f"  EDP efficiency: [{env.edp_efficiency_min:.6f}, {env.edp_efficiency_max:.6f}]")
    print(f"  Energy: [{env.energy_min:.6f}, {env.energy_max:.6f}] J")
    
    # Create agent with loaded policy
    agent = GRPOAgent(
        state_dim, action_dim, 
        policy=pretrained_policy, 
        lr=1e-4,
        group_size=group_size,
        epsilon_clip=0.2,
        epochs=2,
        entropy_coef=0.02,
        kl_coef=0.1,
        num_layers=num_layers,
        freq_bins=freq_bins,
        max_batch=max_batch,
    )
    
    print(f"\n✓ Agent initialized with loaded policy")
    print(f"  Ready to continue training with potentially different reward function")
    print(f"  Constraints:")
    print(f"    - SLO: E2E latency <= {slo_target:.3f}s")
    print(f"    - Temperature: <= {max_temp}°C")
    print(f"    - Power: <= {max_power}W")
    
    # Entropy coefficient decay schedule
    initial_entropy_coef = 0.2  # Start lower since model is already trained
    final_entropy_coef = 0.005
    entropy_coef_schedule = []
    initial_kl = 0.1
    final_kl = 0.02
    kl_coef_schedule = []
    
    # Initialize tracking lists
    episode_rewards, episode_energies, episode_latencies, episode_e2e_latencies = [], [], [], []
    episode_temperatures, episode_powers, losses = [], [], []
    episode_slo_violations, episode_temp_violations, episode_power_violations = [], [], []
    step_entropies, step_latencies, step_num_layers, step_temperatures = [], [], [], []
    step_prefill_freq, step_decode_freq, step_batch_size = [], [], []
    step_per_token_latency = []
    step_decode_latency = []
    step_decode_throughput = []
    step_rewards, step_entropy_penalties = [], []
    step_policy_losses, step_kl_divergences, step_policy_entropies = [], [], []
    total_steps = 0
    configuration_stats = {}
    
    print(f"\nStarting continued training for {num_episodes} episodes...")
    
    # fan_on()
    # for episode in range(num_episodes):
    #     if episode == 300:
    #         fan_off()
    #     if episode >= 400:
    #         if episode % 100 == 0:
    #             fan_on()
    #         if episode % 100 == 30:
    #             fan_off()
    # fan_on()
    # time.sleep(120)
    fan_off()
    for episode in range(num_episodes):
        if episode >= 100:
            if episode % 100 == 0:
                fan_on()
            if episode % 100 == 30:
                fan_off()
        
        
        # Update entropy coefficient with decay schedule
        current_entropy_coef = initial_entropy_coef - (initial_entropy_coef - final_entropy_coef) * (episode / num_episodes)
        agent.entropy_coef = current_entropy_coef
        entropy_coef_schedule.append(current_entropy_coef)
        agent.kl_coef = initial_kl * (1 - (episode / num_episodes)) + final_kl * (episode / num_episodes)
        kl_coef_schedule.append(agent.kl_coef)
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes} (Entropy Coef: {current_entropy_coef:.4f})")
        print(f"{'='*60}")
        state = env.reset()
        episode_reward = 0
        episode_energy, episode_latency, episode_e2e_latency = [], [], []
        episode_temp, episode_power = [], []
        episode_entropy = []
        episode_num_layers = []
        slo_violations = 0
        temp_violations = 0
        power_violations = 0
        
        for step in range(max_steps):
            print(f"\n  Step {step + 1}/{max_steps} (Episode {episode + 1}/{num_episodes})...", end=" ", flush=True)
            
            # GRPO: Sample a group of actions and get rewards for all
            actions, log_probs = agent.sample_group(state, use_smoothing=False)
            
            # Save previous action state
            saved_prev_prefill = env.prev_prefill_bin
            saved_prev_decode = env.prev_decode_bin
            saved_prev_batch = env.prev_batch_size
            saved_prev_layer = env.prev_early_exit_layer
            
            # Get rewards for all sampled actions
            group_rewards = []
            group_infos = []
            for i, action in enumerate(actions):
                env.prev_prefill_bin = saved_prev_prefill
                env.prev_decode_bin = saved_prev_decode
                env.prev_batch_size = saved_prev_batch
                env.prev_early_exit_layer = saved_prev_layer
                
                next_state_temp, reward, _, info = env.step(action)
                group_rewards.append(reward)
                group_infos.append(info)
            
            # Restore prev_* state before taking the actual best action
            env.prev_prefill_bin = saved_prev_prefill
            env.prev_decode_bin = saved_prev_decode
            env.prev_batch_size = saved_prev_batch
            env.prev_early_exit_layer = saved_prev_layer
            
            # Store all group transitions
            agent.store_group_transitions(state, actions, log_probs, group_rewards)
            
            # Select best action to actually take
            best_idx = np.argmax(group_rewards)
            best_action = actions[best_idx]
            
            # Actually take the best action
            next_state, reward, done, info = env.step(best_action)
            
            # Extract early_exit_layer
            early_exit_layer = info.get('early_exit_layer', None)
            if early_exit_layer is None:
                if hasattr(env, 'actions') and best_action < len(env.actions):
                    action_tuple = env.actions[best_action]
                    if len(action_tuple) >= 4:
                        early_exit_layer = action_tuple[3]
            
            if early_exit_layer is None:
                if hasattr(env, 'runner') and env.runner is not None:
                    early_exit_layer = getattr(env.runner, 'current_early_exit', None)
                    if early_exit_layer is not None and early_exit_layer >= 16:
                        early_exit_layer = 16
            
            # Track metrics
            avg_entropy = info.get('avg_entropy', None)
            if avg_entropy is not None:
                if hasattr(avg_entropy, 'item'):
                    avg_entropy = avg_entropy.item()
                episode_entropy.append(float(avg_entropy))
                step_entropies.append(float(avg_entropy))
            else:
                episode_entropy.append(None)
                step_entropies.append(None)
            
            if early_exit_layer is not None:
                episode_num_layers.append(early_exit_layer)
                step_num_layers.append(early_exit_layer)
            else:
                num_layers_default = getattr(env.runner, 'num_layers', 16) if hasattr(env, 'runner') and env.runner is not None else 16
                episode_num_layers.append(num_layers_default)
                step_num_layers.append(num_layers_default)
            
            step_latencies.append(info['end_to_end_latency'])
            if 'temperature' in info:
                step_temperatures.append(info['temperature'])
            else:
                step_temperatures.append(None)
            
            if 'reward' in info:
                step_rewards.append(info['reward'])
            if 'penalty_entropy' in info:
                step_entropy_penalties.append(info['penalty_entropy'])
            
            # Track configuration
            prefill_freq = env.prefill_freq if hasattr(env, 'prefill_freq') else None
            decode_freq = env.decode_freq if hasattr(env, 'decode_freq') else None
            batch_size = env.batch_size if hasattr(env, 'batch_size') else None
            
            step_prefill_freq.append(prefill_freq)
            step_decode_freq.append(decode_freq)
            step_batch_size.append(batch_size)
            
            if 'per_token_latency' in info:
                step_per_token_latency.append(info['per_token_latency'])
            else:
                step_per_token_latency.append(None)
            
            if 'decode_latency' in info:
                step_decode_latency.append(info['decode_latency'])
            else:
                step_decode_latency.append(None)
            
            if 'tokens_decode' in info and 'decode_latency' in info and info['decode_latency'] > 0:
                decode_throughput = info['tokens_decode'] / info['decode_latency']
                step_decode_throughput.append(decode_throughput)
            else:
                step_decode_throughput.append(None)
            
            if early_exit_layer is None:
                early_exit_layer = getattr(env, 'early_exit_layer', 16)
            
            if prefill_freq is not None and decode_freq is not None and batch_size is not None:
                config_key = (prefill_freq, decode_freq, batch_size, early_exit_layer)
                if config_key not in configuration_stats:
                    configuration_stats[config_key] = {
                        'rewards': [], 'e2e_latencies': [], 'energies': [],
                        'temperatures': [], 'powers': [], 'count': 0,
                        'throughput': [], 'early_exit_layers': []
                    }
                
                configuration_stats[config_key]['rewards'].append(reward)
                configuration_stats[config_key]['e2e_latencies'].append(info['end_to_end_latency'])
                configuration_stats[config_key]['energies'].append(info['energy'])
                configuration_stats[config_key]['temperatures'].append(info['temperature'])
                configuration_stats[config_key]['powers'].append(info['power'])
                configuration_stats[config_key]['early_exit_layers'].append(early_exit_layer)
                configuration_stats[config_key]['count'] += 1
                
                if 'tokens_prefill' in info and 'tokens_decode' in info:
                    total_tokens = info.get('tokens_prefill', 0) + info.get('tokens_decode', 0)
                    throughput = total_tokens / info['end_to_end_latency'] if info['end_to_end_latency'] > 0 else 0
                    configuration_stats[config_key]['throughput'].append(throughput)
            
            action_desc = env.get_action_description(best_action) if hasattr(env, 'get_action_description') else f"action_idx={best_action}"
            print(f"✓ E2E={info['end_to_end_latency']:.2f}s, Reward={reward:.4f}, {action_desc}")
            
            episode_reward += reward
            episode_energy.append(info['energy'])
            episode_latency.append(info['latency'])
            episode_e2e_latency.append(info['end_to_end_latency'])
            episode_temp.append(info['temperature'])
            episode_power.append(info['power'])
            
            # Check constraint violations
            if info['slo_violation']:
                slo_violations += 1
            if info['temp_violation']:
                temp_violations += 1
            if info['power_violation']:
                power_violations += 1
            
            state = next_state
            total_steps += 1
            
            if (step + 1) % 5 == 0:
                avg_reward = episode_reward / (step + 1)
                print(f"    [Progress] Avg reward: {avg_reward:.4f}, Total steps: {total_steps}")
            
            if total_steps % update_freq == 0:
                print(f"\n    [Update] Updating agent at step {total_steps}...", end=" ", flush=True)
                loss_dict = agent.update()
                losses.append(loss_dict['total_loss'])
                step_policy_losses.append(loss_dict['policy_loss'])
                step_kl_divergences.append(loss_dict['kl_divergence'])
                step_policy_entropies.append(loss_dict['entropy'])
                print(f"Total Loss: {loss_dict['total_loss']:.4f}, Policy: {loss_dict['policy_loss']:.4f}, KL: {loss_dict['kl_divergence']:.4f}, Entropy: {loss_dict['entropy']:.4f}")
                
                if total_steps % (update_freq * 5) == 0:
                    print(f"    [Update] Updating reference policy at step {total_steps}")
                    agent.update_reference_policy()
                    print(f"    [Hyperparams] Entropy coef: {agent.entropy_coef:.4f}, KL coef: {agent.kl_coef:.4f}")
        
        episode_rewards.append(episode_reward)
        episode_energies.append(np.mean(episode_energy) if episode_energy else 0)
        episode_latencies.append(np.mean(episode_latency) if episode_latency else 0)
        episode_e2e_latencies.append(np.mean(episode_e2e_latency) if episode_e2e_latency else 0)
        episode_temperatures.append(np.mean(episode_temp) if episode_temp else 0)
        episode_powers.append(np.mean(episode_power) if episode_power else 0)
        episode_slo_violations.append(slo_violations / max_steps if max_steps > 0 else 0)
        episode_temp_violations.append(temp_violations / max_steps if max_steps > 0 else 0)
        episode_power_violations.append(power_violations / max_steps if max_steps > 0 else 0)
        
        # Print episode summary
        print(f"\n  Episode {episode + 1} Complete:")
        print(f"    Total Reward: {episode_reward:.4f}")
        print(f"    Avg E2E Latency: {np.mean(episode_e2e_latency) if episode_e2e_latency else 0:.3f}s")
        print(f"    Avg Energy: {np.mean(episode_energy) if episode_energy else 0:.2f}J")
        print(f"    Avg Power: {np.mean(episode_power) if episode_power else 0:.2f}W")
        print(f"    Avg Temp: {np.mean(episode_temp) if episode_temp else 0:.1f}°C")
        print(f"    Violations: SLO={slo_violations}, Temp={temp_violations}, Power={power_violations}")
        if episode_entropy:
            valid_entropies = [e for e in episode_entropy if e is not None]
            if valid_entropies:
                print(f"    Avg Entropy: {np.mean(valid_entropies):.4f}")
        if episode_num_layers:
            print(f"    Avg Layers: {np.mean(episode_num_layers):.1f}")
        print(f"    Progress: {episode + 1}/{num_episodes} episodes ({((episode + 1) / num_episodes * 100):.1f}%)")
        
        # Checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            checkpoint_path = f'models/grpo_early_exit_continued_checkpoint_episode_{episode + 1}.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"    [CHECKPOINT] Model saved to '{checkpoint_path}'")
        
        # Plot every 50 episodes
        should_plot = (episode + 1) % 50 == 0 or episode == num_episodes - 1
        if should_plot:
            avg_e2e = np.mean(episode_e2e_latencies[-50:])
            avg_temp = np.mean(episode_temperatures[-50:])
            avg_power = np.mean(episode_powers[-50:])
            slo_viol = np.mean(episode_slo_violations[-50:]) * 100
            temp_viol = np.mean(episode_temp_violations[-50:]) * 100
            power_viol = np.mean(episode_power_violations[-50:]) * 100
            
            recent_entropies = [e for e in episode_entropy[-50:] if e is not None]
            recent_layers = episode_num_layers[-50:] if len(episode_num_layers) >= 50 else episode_num_layers
            avg_entropy = np.mean(recent_entropies) if recent_entropies else None
            avg_layers = np.mean(recent_layers) if recent_layers else None
            
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {np.mean(episode_rewards[-50:]):.4f}, "
                  f"E2E: {avg_e2e:.3f}s, Temp: {avg_temp:.1f}°C, Power: {avg_power:.1f}W")
            if avg_entropy is not None:
                print(f"  Entropy: {avg_entropy:.4f}, Avg Layers: {avg_layers:.1f}")
            print(f"  Violations - SLO: {slo_viol:.1f}%, Temp: {temp_viol:.1f}%, Power: {power_viol:.1f}%")
            
            _plot_training_metrics(step_entropies, step_latencies, step_num_layers,
                                 episode + 1, save_path=f'plots/early_exit_continued_training_metrics_episode_{episode + 1}.png',
                                 rewards=step_rewards if step_rewards else None,
                                 entropy_penalties=step_entropy_penalties if step_entropy_penalties else None,
                                 temperatures=step_temperatures if step_temperatures else None,
                                 max_temp=max_temp,
                                 prefill_freq=step_prefill_freq if step_prefill_freq else None,
                                 decode_freq=step_decode_freq if step_decode_freq else None,
                                 batch_size=step_batch_size if step_batch_size else None,
                                 per_token_latency=step_per_token_latency if step_per_token_latency else None,
                                 decode_latency=step_decode_latency if step_decode_latency else None,
                                 decode_throughput=step_decode_throughput if step_decode_throughput else None)
            
            _plot_policy_metrics(step_policy_losses, step_kl_divergences, step_policy_entropies,
                               step_entropies, current_episode=episode + 1,
                               save_path=f'plots/policy_metrics_continued_episode_{episode + 1}.png')
    
    # Final plots
    _plot_training_metrics(step_entropies, step_latencies, step_num_layers,
                         num_episodes, save_path='plots/early_exit_continued_training_metrics_final.png',
                         rewards=step_rewards if step_rewards else None,
                         entropy_penalties=step_entropy_penalties if step_entropy_penalties else None,
                         temperatures=step_temperatures if step_temperatures else None,
                         max_temp=max_temp,
                         prefill_freq=step_prefill_freq if step_prefill_freq else None,
                         decode_freq=step_decode_freq if step_decode_freq else None,
                         batch_size=step_batch_size if step_batch_size else None,
                         per_token_latency=step_per_token_latency if step_per_token_latency else None,
                         decode_latency=step_decode_latency if step_decode_latency else None,
                         decode_throughput=step_decode_throughput if step_decode_throughput else None)
    
    _plot_policy_metrics(step_policy_losses, step_kl_divergences, step_policy_entropies,
                        step_entropies, current_episode=num_episodes,
                        save_path='plots/policy_metrics_continued_final.png')
    
    print(f"\n✓ Continued training completed!")
    print(f"  Final model checkpoint: models/grpo_early_exit_continued_checkpoint_episode_{num_episodes}.pth")
    
    return (agent, episode_rewards, episode_energies, episode_latencies, episode_e2e_latencies,
            episode_temperatures, episode_powers, episode_slo_violations,
            episode_temp_violations, episode_power_violations, losses, [], configuration_stats,
            step_policy_losses, step_kl_divergences, step_policy_entropies, step_entropies, step_decode_throughput)


# ============================================================
# TRAINING & TESTING WITH GRPO
# ============================================================

def train_with_grpo(csv_path, num_episodes=500, max_steps=50, 
                    update_freq=200, pretrain_epochs=50,
                    group_size=8, use_simulated_env=False, alpha=0.8, 
                    slo_target=15.0, max_temp=95.0, min_temp=40.0, max_power=15.0,
                    num_layers=2):
    """Train GRPO agent with pre-trained policy and hardware constraints.
    
    Args:
        alpha: Weight for energy vs latency (0 = latency only, 1 = energy only)
        slo_target: SLO target for end-to-end latency in seconds
        max_temp: Maximum temperature constraint (Celsius)
        min_temp: Minimum temperature for normalization (Celsius)
        max_power: Maximum power constraint (Watts)
        use_simulated_env: If True, use simulated env (CSV data). If False, use real Jetson.
    """
    
    freq_bins = 11  # For 11×11×6 = 726 actions base
    max_batch = 31  # Batch range with step of 6
    state_dim = 5  # prefill_freq, decode_freq, batch_size, layers, temperature_headroom
    
    
    # Calculate action dimension: must match OfflineDataset and environment action space!
    # Action space: prefill_freq (11) x decode_freq (11) x batch_size (6 options: [1,7,13,19,25,31] with step=6) x early_exit_layer (8: layers 9-16)
    num_batch_sizes = len(list(range(1, 32, 6)))  # range(1, 32, 6) gives [1, 7, 13, 19, 25, 31] = 6 options
    num_early_exit_layers = 8  # Layers 9, 10, 11, 12, 13, 14, 15, 16 = 8 options
    action_dim = freq_bins * freq_bins * num_batch_sizes * num_early_exit_layers 
    print(f"Action dimension: {freq_bins} x {freq_bins} x {num_batch_sizes} x {num_early_exit_layers} = {action_dim}")

    print("="*60)
    print("PHASE 1: PRE-TRAINING FROM OFFLINE DATA (GRPO)")
    print(f"Unified Reward Function: R = H_eff * T_n + (1 - H_eff) * EDP_n - penalties")
    print(f"  H_eff adaptively weights throughput vs EDP based on temperature and power headroom")
    print(f"Hardware Constraints:")
    print(f"  - Max Power: {max_power}W")
    print(f"  - Max Temperature: {max_temp}°C")
    print(f"  - SLO Target: {slo_target}s")
    print("="*60)
    
    pretrained_policy, pretrain_losses, dataset = pretrain_grpo_from_offline_data(
        csv_path, freq_bins=freq_bins, max_batch=max_batch, epochs=pretrain_epochs, alpha=alpha,
        slo_target=slo_target, max_temp=max_temp, max_power=max_power,
        num_layers=num_layers
    )
    
    print("\n" + "="*60)
    print("PHASE 2: ONLINE FINE-TUNING WITH GRPO")
    print("="*60)
    
    if use_simulated_env:
        print("Using SIMULATED environment (CSV data with noise)")
        print("Power source: CSV statistics (mean_power with 10% noise)")
        env = SimulatedEnergyLatencyEnv(csv_path, freq_bins=freq_bins, max_batch=max_batch, alpha=alpha,
                                        slo_target=slo_target, max_temp=max_temp, max_power=max_power)
    else:
        print("Using REAL Jetson environment (actual hardware sensors)")
        print("Power source: tegrastats (VDD_GPU_SOC + VDD_CPU_CV + VDDQ_VDD2_1V8AO + VIN_SYS_5V0)")
        env = RealJetsonEnv(freq_bins=freq_bins, max_batch=max_batch,
                           slo_target=slo_target, max_temp=max_temp, min_temp=min_temp, max_power=max_power)
    
    # Initialize normalization ranges from offline dataset
    # Enable tracking in environment so ranges are updated during online training
    env.throughput_min = getattr(dataset, 'throughput_min', 0) or 0
    env.throughput_max = getattr(dataset, 'throughput_max', 1) or 1
    env.edp_efficiency_min = getattr(dataset, 'edp_efficiency_min', 0) or 0
    env.edp_efficiency_max = getattr(dataset, 'edp_efficiency_max', 1) or 1
    env.energy_min = getattr(dataset, 'energy_min', None)
    env.energy_max = getattr(dataset, 'energy_max', None)
    env._track_normalization = True  # Enable automatic range updates in RealJetsonEnv.step()
    
    print(f"\n[Online Training] Initial normalization ranges (from offline dataset):")
    print(f"  Throughput: [{env.throughput_min:.4f}, {env.throughput_max:.4f}] tokens/sec")
    print(f"  EDP efficiency: [{env.edp_efficiency_min:.6f}, {env.edp_efficiency_max:.6f}]")
    print(f"  Energy: [{env.energy_min:.6f}, {env.energy_max:.6f}] J")
    print(f"  Note: Ranges will be automatically updated during training to include online data")
    
    agent = GRPOAgent(
        state_dim, action_dim, 
        policy=pretrained_policy, 
        lr=1e-4,
        group_size=group_size,
        epsilon_clip=0.2,
        epochs=1,
        entropy_coef=0.02,
        kl_coef=0.1,
        num_layers=num_layers,
        freq_bins=freq_bins,
        max_batch=max_batch,
    )
    
    # Entropy coefficient decay schedule
    initial_entropy_coef = 0.1
    final_entropy_coef = 0.005
    entropy_coef_schedule = []  # Track entropy coefficient over episodes
    # Add KL coefficient schedule  
    initial_kl = 0.1
    final_kl = 0.05 
    kl_coef_schedule = []  # Track KL coefficient over episodes

    
    episode_rewards, episode_energies, episode_latencies, episode_e2e_latencies = [], [], [], []
    episode_temperatures, episode_powers, losses = [], [], []
    episode_slo_violations, episode_temp_violations, episode_power_violations = [], [], []
    # Track entropy, latency, layers, and temperature over time for plotting
    step_entropies, step_latencies, step_num_layers, step_temperatures = [], [], [], []
    # Track prefill_freq, decode_freq, and batch_size for plotting
    step_prefill_freq, step_decode_freq, step_batch_size = [], [], []
    # Track per-token latency and decode latency for plotting
    step_per_token_latency = []
    step_decode_latency = []
    # Track decode throughput (tokens/second) for analysis
    step_decode_throughput = []
    # Track reward and entropy penalty for debugging
    step_rewards, step_entropy_penalties = [], []
    # Track policy loss and KL divergence for analysis
    step_policy_losses, step_kl_divergences, step_policy_entropies = [], [], []
    total_steps = 0
    
    # Track configurations used during training for visualization
    configuration_stats = {}  # {(prefill, decode, batch): {rewards: [], e2e: [], ...}}
    
    print(f"\nTraining for {num_episodes} episodes with GRPO (group_size={group_size})...")
    print(f"Constraints:")
    print(f"  - SLO: E2E latency <= {slo_target:.3f}s")
    print(f"  - Temperature: <= {max_temp}°C")
    print(f"  - Power: <= {max_power}W")
    print(f"Adaptive Strategy:")
    print(f"  - H_eff > 0.5: Prioritize throughput/latency (more weight on throughput)")
    print(f"  - H_eff <= 0.5: Prioritize EDP (more weight on energy efficiency)")
    print(f"  - H_eff = w_temp * H_T + w_power * H_P (effective headroom)")
    
    # initialize_fan()
    # time.sleep(120)
    fan_off()
    for episode in range(num_episodes):
        if episode == 100:
            fan_on()
        if episode >= 200:
            if episode % 100 == 50:
                fan_on()
            if episode % 100 == 0:
                fan_off()
            

        # Update entropy coefficient with decay schedule
        current_entropy_coef = initial_entropy_coef - (initial_entropy_coef - final_entropy_coef) * (episode / num_episodes)
        agent.entropy_coef = current_entropy_coef
        entropy_coef_schedule.append(current_entropy_coef)
        # Increase KL penalty over time
        agent.kl_coef = initial_kl * (1 - (episode / num_episodes)) + final_kl * (episode / num_episodes)
        kl_coef_schedule.append(agent.kl_coef)

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes} (Entropy Coef: {current_entropy_coef:.4f})")
        print(f"{'='*60}")
        state = env.reset()
        episode_reward = 0
        episode_energy, episode_latency, episode_e2e_latency = [], [], []
        episode_temp, episode_power = [], []
        episode_entropy = []  # Track entropy per episode
        episode_num_layers = []  # Track layers per episode
        slo_violations = 0
        temp_violations = 0
        power_violations = 0
        
        for step in range(max_steps):
            print(f"\n  Step {step + 1}/{max_steps} (Episode {episode + 1}/{num_episodes})...", end=" ", flush=True)
            # GRPO: Sample a group of actions and get rewards for all
            actions, log_probs = agent.sample_group(state, use_smoothing=False)
            
            # Save previous action state before sampling group
            # This ensures prev_* variables are only set once per step, not per sampled action
            saved_prev_prefill = env.prev_prefill_bin
            saved_prev_decode = env.prev_decode_bin
            saved_prev_batch = env.prev_batch_size
            saved_prev_layer = env.prev_early_exit_layer
            
            # Get rewards for all sampled actions
            group_rewards = []
            group_infos = []
            for i, action in enumerate(actions):
                # Restore prev_* state before each group action evaluation
                # This ensures all group actions are evaluated with the same prev state
                env.prev_prefill_bin = saved_prev_prefill
                env.prev_decode_bin = saved_prev_decode
                env.prev_batch_size = saved_prev_batch
                env.prev_early_exit_layer = saved_prev_layer
                
                # Temporarily step environment to get reward
                next_state_temp, reward, _, info = env.step(action)
                group_rewards.append(reward)
                group_infos.append(info)
                
                # Note: Normalization ranges are automatically updated in RealJetsonEnv.step()
                # when _track_normalization is enabled, so no manual tracking needed here
                
            # Debug: Print E2E latency for first few actions in first step
            if step == 0 and i < 2:  # First step, first 2 actions
                action_desc = env.get_action_description(action) if hasattr(env, 'get_action_description') else f"action_idx={action}"
                print(f"\n    [Group Action {i+1}/{len(actions)}] E2E={info['end_to_end_latency']:.4f}s, {action_desc}")
            
            # Restore prev_* state before taking the actual best action
            env.prev_prefill_bin = saved_prev_prefill
            env.prev_decode_bin = saved_prev_decode
            env.prev_batch_size = saved_prev_batch
            env.prev_early_exit_layer = saved_prev_layer
            
            # Store all group transitions
            agent.store_group_transitions(state, actions, log_probs, group_rewards)
            
            # Select best action to actually take (or random from group)
            best_idx = np.argmax(group_rewards)
            best_action = actions[best_idx]
            
            # Actually take the best action
            # Note: Normalization ranges are automatically updated in RealJetsonEnv.step()
            # when _track_normalization is enabled
            next_state, reward, done, info = env.step(best_action)
            
            # Extract early_exit_layer from info dict (actual value used during inference)
            # This is more reliable than extracting from action tuple, as it reflects what was actually used
            early_exit_layer = info.get('early_exit_layer', None)
            
            # Fallback: Extract from action tuple if not in info
            if early_exit_layer is None:
                if hasattr(env, 'actions') and best_action < len(env.actions):
                    action_tuple = env.actions[best_action]
                    if len(action_tuple) >= 4:
                        early_exit_layer = action_tuple[3]
                        print(f"[DEBUG] Extracted early_exit_layer={early_exit_layer} from action tuple: {action_tuple}")
            
            # Final fallback: Get from runner's current_early_exit if available
            if early_exit_layer is None:
                if hasattr(env, 'runner') and env.runner is not None:
                    early_exit_layer = getattr(env.runner, 'current_early_exit', None)
                    if early_exit_layer is not None:
                        if early_exit_layer >= 16:
                            # If current_early_exit is 16 or more, it means full model was used
                            early_exit_layer = 16
                        print(f"[DEBUG] Extracted early_exit_layer={early_exit_layer} from runner.current_early_exit")
            
            # Debug: Print what we're tracking
            if step < 3:  # Only print for first few steps to avoid spam
                print(f"[DEBUG] Tracking: early_exit_layer={early_exit_layer}, from_info={info.get('early_exit_layer', 'NOT_IN_INFO')}")
            
            # Track entropy and number of layers for plotting
            avg_entropy = info.get('avg_entropy', None)
            if avg_entropy is not None:
                if hasattr(avg_entropy, 'item'):
                    avg_entropy = avg_entropy.item()
                episode_entropy.append(float(avg_entropy))
                step_entropies.append(float(avg_entropy))
            else:
                episode_entropy.append(None)
                step_entropies.append(None)
            
            # Track number of layers used (use actual value from inference)
            if early_exit_layer is not None:
                episode_num_layers.append(early_exit_layer)
                step_num_layers.append(early_exit_layer)
            else:
                # Full model - use num_layers
                num_layers = getattr(env.runner, 'num_layers', 16) if hasattr(env, 'runner') and env.runner is not None else 16
                episode_num_layers.append(num_layers)
                step_num_layers.append(num_layers)
                print(f"[WARNING] Could not determine early_exit_layer, defaulting to {num_layers}")
            
            # Track latency for plotting
            step_latencies.append(info['end_to_end_latency'])
            
            # Track temperature for plotting
            if 'temperature' in info:
                step_temperatures.append(info['temperature'])
            else:
                step_temperatures.append(None)
            
            # Track reward and entropy penalty for debugging
            if 'reward' in info:
                step_rewards.append(info['reward'])
            if 'penalty_entropy' in info:
                step_entropy_penalties.append(info['penalty_entropy'])
            
            # Track configuration statistics for visualization
            # Get action configuration from environment
            prefill_freq = env.prefill_freq if hasattr(env, 'prefill_freq') else None
            decode_freq = env.decode_freq if hasattr(env, 'decode_freq') else None
            batch_size = env.batch_size if hasattr(env, 'batch_size') else None
            
            # Track prefill_freq, decode_freq, and batch_size for plotting
            step_prefill_freq.append(prefill_freq)
            step_decode_freq.append(decode_freq)
            step_batch_size.append(batch_size)
            
            # Track per-token latency for plotting
            if 'per_token_latency' in info:
                step_per_token_latency.append(info['per_token_latency'])
            else:
                step_per_token_latency.append(None)
            
            # Track decode latency for plotting
            if 'decode_latency' in info:
                step_decode_latency.append(info['decode_latency'])
            else:
                step_decode_latency.append(None)
            
            # Calculate and track decode throughput (tokens/second)
            if 'tokens_decode' in info and 'decode_latency' in info and info['decode_latency'] > 0:
                decode_throughput = info['tokens_decode'] / info['decode_latency']
                step_decode_throughput.append(decode_throughput)
            else:
                step_decode_throughput.append(None)
            
            # Get early_exit_layer from the action tuple (already extracted above)
            # early_exit_layer is extracted from action_tuple[3] at line 2684
            # If not available, use None or default to 16 (full model)
            if early_exit_layer is None:
                # Try to get from environment if available
                early_exit_layer = getattr(env, 'early_exit_layer', None)
                if early_exit_layer is None:
                    # Default to full model (16 layers)
                    early_exit_layer = 16
            
            if prefill_freq is not None and decode_freq is not None and batch_size is not None:
                # Include early_exit_layer in config_key
                config_key = (prefill_freq, decode_freq, batch_size, early_exit_layer)
                if config_key not in configuration_stats:
                    configuration_stats[config_key] = {
                        'rewards': [], 'e2e_latencies': [], 'energies': [],
                        'temperatures': [], 'powers': [], 'count': 0,
                        'throughput': [], 'early_exit_layers': []  # Track early exit layers
                    }
                
                configuration_stats[config_key]['rewards'].append(reward)
                configuration_stats[config_key]['e2e_latencies'].append(info['end_to_end_latency'])
                configuration_stats[config_key]['energies'].append(info['energy'])
                configuration_stats[config_key]['temperatures'].append(info['temperature'])
                configuration_stats[config_key]['powers'].append(info['power'])
                configuration_stats[config_key]['early_exit_layers'].append(early_exit_layer)
                configuration_stats[config_key]['count'] += 1
                
                # Calculate throughput if tokens info available
                if 'tokens_prefill' in info and 'tokens_decode' in info:
                    total_tokens = info.get('tokens_prefill', 0) + info.get('tokens_decode', 0)
                    throughput = total_tokens / info['end_to_end_latency'] if info['end_to_end_latency'] > 0 else 0
                    configuration_stats[config_key]['throughput'].append(throughput)
            
            # Debug: Print E2E latency for the selected action (first few steps)
            # Print progress after each step
            action_desc = env.get_action_description(best_action) if hasattr(env, 'get_action_description') else f"action_idx={best_action}"
            print(f"✓ E2E={info['end_to_end_latency']:.2f}s, Reward={reward:.4f}, {action_desc}")
            
            if step < 3:
                print(f"    [Selected Action] E2E={info['end_to_end_latency']:.4f}s, Reward={reward:.4f}, {action_desc}")
            
            episode_reward += reward
            episode_energy.append(info['energy'])
            episode_latency.append(info['latency'])
            episode_e2e_latency.append(info['end_to_end_latency'])
            episode_temp.append(info['temperature'])
            episode_power.append(info['power'])
            
            # Check constraint violations
            if info['slo_violation']:
                slo_violations += 1
            if info['temp_violation']:
                temp_violations += 1
            if info['power_violation']:
                power_violations += 1
            
            state = next_state
            total_steps += 1
            
            # Print episode summary every few steps
            if (step + 1) % 5 == 0:
                avg_reward = episode_reward / (step + 1)
                print(f"    [Progress] Avg reward: {avg_reward:.4f}, Total steps: {total_steps}")
            
            if total_steps % update_freq == 0:
                print(f"\n    [Update] Updating agent at step {total_steps}...", end=" ", flush=True)
                loss_dict = agent.update()
                losses.append(loss_dict['total_loss'])
                step_policy_losses.append(loss_dict['policy_loss'])
                step_kl_divergences.append(loss_dict['kl_divergence'])
                step_policy_entropies.append(loss_dict['entropy'])
                print(f"Total Loss: {loss_dict['total_loss']:.4f}, Policy: {loss_dict['policy_loss']:.4f}, KL: {loss_dict['kl_divergence']:.4f}, Entropy: {loss_dict['entropy']:.4f}")
            
                if total_steps % (update_freq * 5) == 0:
                    print(f"    [Update] Updating reference policy at step {total_steps}")
                    agent.update_reference_policy()
                    print(f"    [Hyperparams] Entropy coef: {agent.entropy_coef:.4f}, KL coef: {agent.kl_coef:.4f}")
               
                # if episode % 10 == 0:
                #     print(f"Episode {episode}, Update {len(losses)}")
                #     print(f"  Entropy coef: {current_entropy_coef:.4f}")
                #     print(f"  KL coef: {current_kl:.4f}")

        episode_rewards.append(episode_reward)
        episode_energies.append(np.mean(episode_energy) if episode_energy else 0)
        episode_latencies.append(np.mean(episode_latency) if episode_latency else 0)
        episode_e2e_latencies.append(np.mean(episode_e2e_latency) if episode_e2e_latency else 0)
        episode_temperatures.append(np.mean(episode_temp) if episode_temp else 0)
        episode_powers.append(np.mean(episode_power) if episode_power else 0)
        episode_slo_violations.append(slo_violations / max_steps if max_steps > 0 else 0)
        episode_temp_violations.append(temp_violations / max_steps if max_steps > 0 else 0)
        episode_power_violations.append(power_violations / max_steps if max_steps > 0 else 0)
        
        # Print episode summary
        print(f"\n  Episode {episode + 1} Complete:")
        print(f"    Total Reward: {episode_reward:.4f}")
        print(f"    Avg E2E Latency: {np.mean(episode_e2e_latency) if episode_e2e_latency else 0:.3f}s")
        print(f"    Avg Energy: {np.mean(episode_energy) if episode_energy else 0:.2f}J")
        print(f"    Avg Power: {np.mean(episode_power) if episode_power else 0:.2f}W")
        print(f"    Avg Temp: {np.mean(episode_temp) if episode_temp else 0:.1f}°C")
        print(f"    Violations: SLO={slo_violations}, Temp={temp_violations}, Power={power_violations}")
        if episode_entropy:
            valid_entropies = [e for e in episode_entropy if e is not None]
            if valid_entropies:
                print(f"    Avg Entropy: {np.mean(valid_entropies):.4f}")
        if episode_num_layers:
            print(f"    Avg Layers: {np.mean(episode_num_layers):.1f}")
        print(f"    Progress: {episode + 1}/{num_episodes} episodes ({((episode + 1) / num_episodes * 100):.1f}%)")
        
        # Checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            checkpoint_path = f'models/grpo_early_exit_checkpoint_episode_{episode + 1}.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(agent.policy.state_dict(), checkpoint_path)
            print(f"    [CHECKPOINT] Model saved to '{checkpoint_path}'")
        
        # Plot every 50 episodes, or at the final episode if num_episodes < 50
        should_plot = (episode + 1) % 50 == 0 or episode == num_episodes - 1
        if should_plot:
            avg_e2e = np.mean(episode_e2e_latencies[-50:])
            avg_temp = np.mean(episode_temperatures[-50:])
            avg_power = np.mean(episode_powers[-50:])
            slo_viol = np.mean(episode_slo_violations[-50:]) * 100
            temp_viol = np.mean(episode_temp_violations[-50:]) * 100
            power_viol = np.mean(episode_power_violations[-50:]) * 100
            
            # Calculate average entropy and layers for recent episodes
            recent_entropies = [e for e in episode_entropy[-50:] if e is not None]
            recent_layers = episode_num_layers[-50:] if len(episode_num_layers) >= 50 else episode_num_layers
            avg_entropy = np.mean(recent_entropies) if recent_entropies else None
            avg_layers = np.mean(recent_layers) if recent_layers else None
            
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {np.mean(episode_rewards[-50:]):.4f}, "
                  f"E2E: {avg_e2e:.3f}s, Temp: {avg_temp:.1f}°C, Power: {avg_power:.1f}W")
            if avg_entropy is not None:
                print(f"  Entropy: {avg_entropy:.4f}, Avg Layers: {avg_layers:.1f}")
            print(f"  Violations - SLO: {slo_viol:.1f}%, Temp: {temp_viol:.1f}%, Power: {power_viol:.1f}%")
            print(f"  Entropy Coefficient: {current_entropy_coef:.4f} (schedule: {initial_entropy_coef:.4f} → {final_entropy_coef:.4f})")
            print(f"  Normalization ranges (offline + online): "
                  f"Throughput=[{env.throughput_min:.4f}, {env.throughput_max:.4f}], "
                  f"EDP=[{env.edp_efficiency_min:.6f}, {env.edp_efficiency_max:.6f}]")
            
            # Plot metrics over time
            _plot_training_metrics(step_entropies, step_latencies, step_num_layers, 
                                 episode + 1, save_path=f'plots/early_exit_training_metrics_episode_{episode + 1}.png',
                                 rewards=step_rewards if step_rewards else None,
                                 entropy_penalties=step_entropy_penalties if step_entropy_penalties else None,
                                 temperatures=step_temperatures if step_temperatures else None,
                                 max_temp=max_temp,
                                 prefill_freq=step_prefill_freq if step_prefill_freq else None,
                                 decode_freq=step_decode_freq if step_decode_freq else None,
                                 batch_size=step_batch_size if step_batch_size else None,
                                 per_token_latency=step_per_token_latency if step_per_token_latency else None,
                                 decode_latency=step_decode_latency if step_decode_latency else None,
                                 decode_throughput=step_decode_throughput if step_decode_throughput else None)
            
            # Plot policy metrics (loss, KL divergence, entropy)
            _plot_policy_metrics(step_policy_losses, step_kl_divergences, step_policy_entropies, 
                               step_entropies, current_episode=episode + 1,
                               save_path=f'plots/policy_metrics_episode_{episode + 1}.png')
    
    # Final plot at end of training
    _plot_training_metrics(step_entropies, step_latencies, step_num_layers, 
                         num_episodes, save_path='plots/early_exit_training_metrics_final.png',
                         rewards=step_rewards if step_rewards else None,
                         entropy_penalties=step_entropy_penalties if step_entropy_penalties else None,
                         temperatures=step_temperatures if step_temperatures else None,
                         max_temp=max_temp,
                         prefill_freq=step_prefill_freq if step_prefill_freq else None,
                         decode_freq=step_decode_freq if step_decode_freq else None,
                         batch_size=step_batch_size if step_batch_size else None,
                         per_token_latency=step_per_token_latency if step_per_token_latency else None,
                         decode_latency=step_decode_latency if step_decode_latency else None,
                         decode_throughput=step_decode_throughput if step_decode_throughput else None)
    
    # Final policy metrics plot
    _plot_policy_metrics(step_policy_losses, step_kl_divergences, step_policy_entropies,
                        step_entropies, current_episode=num_episodes,
                        save_path='plots/policy_metrics_final.png')
    
    return (agent, episode_rewards, episode_energies, episode_latencies, episode_e2e_latencies, 
            episode_temperatures, episode_powers, episode_slo_violations, 
            episode_temp_violations, episode_power_violations, losses, pretrain_losses, configuration_stats,
            step_policy_losses, step_kl_divergences, step_policy_entropies, step_entropies, step_decode_throughput)


# def test_grpo_agent(agent, env, num_episodes=5, slo_target=None, max_temp=None, max_power=None):
#     """Test the trained GRPO agent with constraint checking."""
#     print("\n" + "="*60)
#     print("TESTING TRAINED GRPO AGENT")
#     print("Constraints:")
#     if slo_target:
#         print(f"  - SLO: E2E latency <= {slo_target:.3f}s")
#     if max_temp:
#         print(f"  - Temperature: <= {max_temp}°C")
#     if max_power:
#         print(f"  - Power: <= {max_power}W")
#     print("="*60)
    
#     for episode in range(num_episodes):
#         state = env.reset()
#         print(f"\nTest Episode {episode + 1}")
#         print(f"Initial: Prefill freq={env.prefill_freq}, Decode freq={env.decode_freq}, Batch={env.batch_size}")
        
#         total_reward = 0
#         e2e_latencies, temperatures, powers = [], [], []
#         slo_violations, temp_violations, power_violations = 0, 0, 0
        
#         for step in range(10):
#             action, _ = agent.select_action(state, deterministic=True)
#             next_state, reward, done, info = env.step(action)
#             total_reward += reward
#             e2e_latencies.append(info['end_to_end_latency'])
#             temperatures.append(info['temperature'])
#             powers.append(info['power'])
            
#             # Check constraints
#             violations = []
#             if info['slo_violation']:
#                 violations.append("SLO")
#                 slo_violations += 1
#             if info['temp_violation']:
#                 violations.append("TEMP")
#                 temp_violations += 1
#             if info['power_violation']:
#                 violations.append("POWER")
#                 power_violations += 1
            
#             status = "✓" if not violations else f"✗ {', '.join(violations)}"
            
#             state = next_state
            
#             if step < 5:
#                 print(f"  Step {step + 1}: {env.get_action_description(action)}")
#                 print(f"    Energy={info['energy']:.2f}J, E2E Latency={info['end_to_end_latency']:.3f}s {status}")
#                 print(f"    Temp={info['temperature']:.1f}°C, Power={info['power']:.1f}W, Headroom={info['power_headroom']:.2f}")
        
#         print(f"Final: Prefill freq={env.prefill_freq}, Decode freq={env.decode_freq}, Batch={env.batch_size}")
#         print(f"Total Reward: {total_reward:.4f}")
#         print(f"Averages: E2E={np.mean(e2e_latencies):.3f}s, Temp={np.mean(temperatures):.1f}°C, Power={np.mean(powers):.1f}W")
#         print(f"Compliance: SLO {(1-slo_violations/10)*100:.0f}%, Temp {(1-temp_violations/10)*100:.0f}%, Power {(1-power_violations/10)*100:.0f}%")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_results(episode_rewards, episode_energies, episode_latencies, episode_e2e_latencies,
                          episode_temperatures, episode_powers,
                          episode_slo_violations, episode_temp_violations, episode_power_violations,
                          online_losses, pretrain_losses, 
                          slo_target=None, max_temp=None, max_power=None,
                          save_path='plots/early_exit_training_results.png'):
    """Plot training results with constraint monitoring."""
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # Pre-training loss
    axes[0, 0].plot(pretrain_losses, color='purple')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GRPO Pre-training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode rewards
    axes[0, 1].plot(episode_rewards, color='blue')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].set_title('Online Training - Episode Rewards')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Moving average of rewards
    window = min(10, len(episode_rewards))
    if window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(moving_avg, color='blue')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Avg Reward')
        axes[0, 2].set_title(f'Reward (Moving Avg, window={window})')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Energy
    axes[1, 0].plot(episode_energies, color='red')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Energy')
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latency
    axes[1, 1].plot(episode_latencies, color='green')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Latency')
    axes[1, 1].set_title('Latency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Online training loss
    if online_losses:
        axes[1, 2].plot(online_losses, color='orange')
        axes[1, 2].set_xlabel('Update Step')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('GRPO Online Training Loss')
        axes[1, 2].grid(True, alpha=0.3)
    
    # End-to-end latency
    axes[2, 0].plot(episode_e2e_latencies, color='purple')
    if slo_target:
        axes[2, 0].axhline(y=slo_target, color='red', linestyle='--', linewidth=2, label=f'SLO Target ({slo_target}s)')
        axes[2, 0].legend()
    # Zoom into actual data range
    if len(episode_e2e_latencies) > 0:
        data_min = np.min(episode_e2e_latencies)
        data_max = np.max(episode_e2e_latencies)
        data_range = data_max - data_min
        padding = max(0.1 * data_range, 0.5)  # 10% padding or at least 0.5s
        y_min = max(0, data_min - padding)
        y_max = data_max + padding
        # Include SLO target if it's within reasonable range
        if slo_target and slo_target < y_max * 1.5:
            y_max = max(y_max, slo_target * 1.1)
        axes[2, 0].set_ylim([y_min, y_max])
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('End-to-End Latency (seconds)')
    axes[2, 0].set_title('End-to-End Latency (Prefill + Decode)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # SLO violations
    if slo_target and episode_slo_violations:
        violation_rates = np.array(episode_slo_violations) * 100
        axes[2, 1].plot(violation_rates, color='red')
        # Zoom into actual data range
        data_max = np.max(violation_rates)
        if data_max > 0:
            # If there are violations, show up to max + 10%
            y_max = min(100, data_max * 1.1 + 5)
            axes[2, 1].set_ylim([0, y_max])
        else:
            # If no violations, show small range around 0
            axes[2, 1].set_ylim([-1, 5])
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('SLO Violation Rate (%)')
        axes[2, 1].set_title('SLO Violation Rate per Episode')
        axes[2, 1].grid(True, alpha=0.3)
    
    # Combined latency vs energy scatter
    if len(episode_e2e_latencies) > 0:
        scatter_colors = episode_rewards if len(episode_rewards) > 0 else None
        scatter = axes[2, 2].scatter(episode_energies, episode_e2e_latencies, c=scatter_colors, 
                          cmap='viridis', alpha=0.6, s=30)
        if slo_target:
            axes[2, 2].axhline(y=slo_target, color='red', linestyle='--', linewidth=2, 
                              label=f'SLO Target ({slo_target}s)')
            axes[2, 2].legend()
        if scatter_colors is not None:
            cbar = plt.colorbar(scatter, ax=axes[2, 2])
            cbar.set_label('Reward', rotation=270, labelpad=15)
        axes[2, 2].set_xlabel('Average Energy (J)')
        axes[2, 2].set_ylabel('End-to-End Latency (s)')
        axes[2, 2].set_title('Energy vs End-to-End Latency Trade-off')
        axes[2, 2].grid(True, alpha=0.3)
    
    # Temperature monitoring
    axes[3, 0].plot(episode_temperatures, color='orange')
    if max_temp:
        axes[3, 0].axhline(y=max_temp, color='red', linestyle='--', linewidth=2, 
                          label=f'Max Temp ({max_temp}°C)')
        axes[3, 0].legend()
    # Zoom into actual data range
    if len(episode_temperatures) > 0:
        data_min = np.min(episode_temperatures)
        data_max = np.max(episode_temperatures)
        data_range = data_max - data_min
        padding = max(0.1 * data_range, 2.0)  # 10% padding or at least 2°C
        y_min = max(0, data_min - padding)
        y_max = data_max + padding
        # Include max_temp if it's within reasonable range
        if max_temp and max_temp < y_max * 1.5:
            y_max = max(y_max, max_temp * 1.05)
        axes[3, 0].set_ylim([y_min, y_max])
    axes[3, 0].set_xlabel('Episode')
    axes[3, 0].set_ylabel('Temperature (°C)')
    axes[3, 0].set_title('GPU Temperature Over Time')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Power monitoring
    axes[3, 1].plot(episode_powers, color='brown')
    if max_power:
        axes[3, 1].axhline(y=max_power, color='red', linestyle='--', linewidth=2, 
                          label=f'Max Power ({max_power}W)')
        axes[3, 1].legend()
    # Zoom into actual data range
    if len(episode_powers) > 0:
        data_min = np.min(episode_powers)
        data_max = np.max(episode_powers)
        data_range = data_max - data_min
        padding = max(0.1 * data_range, 2.0)  # 10% padding or at least 2W
        y_min = max(0, data_min - padding)
        y_max = data_max + padding
        # Include max_power if it's within reasonable range
        if max_power and max_power < y_max * 1.5:
            y_max = max(y_max, max_power * 1.05)
        axes[3, 1].set_ylim([y_min, y_max])
    axes[3, 1].set_xlabel('Episode')
    axes[3, 1].set_ylabel('Power (W)')
    axes[3, 1].set_title('Power Consumption Over Time')
    axes[3, 1].grid(True, alpha=0.3)
    
    # All violations combined
    axes[3, 2].plot(np.array(episode_slo_violations) * 100, label='SLO', color='red', alpha=0.7)
    axes[3, 2].plot(np.array(episode_temp_violations) * 100, label='Temp', color='orange', alpha=0.7)
    axes[3, 2].plot(np.array(episode_power_violations) * 100, label='Power', color='brown', alpha=0.7)
    axes[3, 2].set_xlabel('Episode')
    axes[3, 2].set_ylabel('Violation Rate (%)')
    axes[3, 2].set_title('All Constraint Violations')
    axes[3, 2].legend()
    axes[3, 2].grid(True, alpha=0.3)
    axes[3, 2].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved as '{save_path}'")
    plt.close()


def analyze_dataset(csv_path, freq_bins=11, max_batch=1, alpha=0.8, save_path=None):
    """Analyze the offline dataset and visualize insights.
    
    Args:
        alpha: Weight for energy vs latency (0 = latency only, 1 = energy only)
        save_path: Path to save the analysis plot (default: auto-generated based on alpha)
    """
    dataset = OfflineDataset(csv_path, freq_bins=freq_bins, max_batch=max_batch, alpha=alpha)
    
    # Check if we have any processed data
    if len(dataset.processed_data) == 0:
        print("\nWARNING: No valid data points found in CSV files!")
        print("Possible reasons:")
        print("  1. Batch sizes in CSV don't match max_batch parameter")
        print("  2. Frequency bins in CSV don't match expected range")
        print("  3. Missing required columns in CSV")
        print("  4. Early exit layer values don't match expected range [4, 16]")
        return dataset
    
    # Analyze reward distribution
    rewards = [d['reward'] for d in dataset.processed_data]
    # Use H_eff (via adaptive_alpha_latency) instead of power_headroom threshold
    # adaptive_alpha_latency = H_eff, so H_eff > 0.5 means more weight on throughput
    edp_mode_count = sum(1 for d in dataset.processed_data if d.get('adaptive_alpha_latency', 0.5) <= 0.5)
    throughput_mode_count = len(dataset.processed_data) - edp_mode_count
    
    print(f"\nUnified Reward Function: R = H_eff * T_n + (1 - H_eff) * EDP_n - λ_L * f(x_L) - λ_T * f(x_T) - λ_P * f(x_P)")
    print(f"  Where H_eff = w_T * H_T + w_P * H_P (effective headroom)")
    print(f"  - H_eff > 0.5: More weight on throughput (adaptive_alpha_latency > 0.5)")
    print(f"  - H_eff <= 0.5: More weight on EDP (adaptive_alpha_latency <= 0.5)")
    print(f"Reward statistics:")
    if len(rewards) > 0:
        print(f"  Min reward: {np.min(rewards):.4f}")
        print(f"  Max reward: {np.max(rewards):.4f}")
        print(f"  Mean reward: {np.mean(rewards):.4f}")
        print(f"  Samples with H_eff > 0.5 (throughput-weighted): {throughput_mode_count} ({throughput_mode_count/len(dataset.processed_data)*100:.1f}%)")
        print(f"  Samples with H_eff <= 0.5 (EDP-weighted): {edp_mode_count} ({edp_mode_count/len(dataset.processed_data)*100:.1f}%)")
    else:
        print("  No rewards available")
    
    # Create visualization - expanded to 7 rows to accommodate all plots
    fig, axes = plt.subplots(7, 3, figsize=(18, 32))
    
    # 1. Reward breakdown by prefill frequency bin
    prefill_freq_data = {}
    prefill_freq_data_edp = {}  # Separate data for EDP-weighted samples (H_eff <= 0.5)
    prefill_freq_data_throughput = {}  # Separate data for throughput-weighted samples (H_eff > 0.5)
    
    for d in dataset.processed_data:
        freq = d['prefill_freq_bin']
        if freq not in prefill_freq_data:
            prefill_freq_data[freq] = {'latency': [], 'energy': [], 'reward': []}
            prefill_freq_data_edp[freq] = {'latency': [], 'energy': [], 'reward': []}
            prefill_freq_data_throughput[freq] = {'latency': [], 'energy': [], 'reward': []}
        
        # Apply adaptive alpha weighting based on optimization mode (EDP vs throughput)
        # Use adaptive_alpha if available, otherwise fall back to fixed alpha
        alpha_latency = d.get('adaptive_alpha_latency', 1 - alpha)
        alpha_energy = d.get('adaptive_alpha_energy', alpha)
        
        # All data (adaptive)
        prefill_freq_data[freq]['latency'].append(alpha_latency * d['latency_component'])
        prefill_freq_data[freq]['energy'].append(alpha_energy * d['energy_component'])
        prefill_freq_data[freq]['reward'].append(d['reward'])
        
        # EDP-weighted breakdown (for visualization: shows what EDP component looks like when H_eff is low)
        # Note: Actual reward uses unified function, this is just for visualization
        prefill_freq_data_edp[freq]['latency'].append(0.2 * d['latency_component'])
        prefill_freq_data_edp[freq]['energy'].append(0.8 * d['energy_component'])
        prefill_freq_data_edp[freq]['reward'].append(d['reward'])
        
        # Throughput-weighted breakdown (for visualization: shows what throughput component looks like when H_eff is high)
        # Note: Actual reward uses unified function, this is just for visualization
        prefill_freq_data_throughput[freq]['latency'].append(0.8 * d['latency_component'])
        prefill_freq_data_throughput[freq]['energy'].append(0.2 * d['energy_component'])
        prefill_freq_data_throughput[freq]['reward'].append(d['reward'])
    
    freqs = sorted(prefill_freq_data.keys())
    latency_means = [np.mean(prefill_freq_data[f]['latency']) for f in freqs]
    energy_means = [np.mean(prefill_freq_data[f]['energy']) for f in freqs]
    latency_stds = [np.std(prefill_freq_data[f]['latency']) for f in freqs]
    energy_stds = [np.std(prefill_freq_data[f]['energy']) for f in freqs]
    
    # Calculate average adaptive weights for display
    avg_alpha_latency = np.mean([d.get('adaptive_alpha_latency', 1 - alpha) for d in dataset.processed_data])
    avg_alpha_energy = np.mean([d.get('adaptive_alpha_energy', alpha) for d in dataset.processed_data])
    
    x = np.arange(len(freqs))
    width = 0.35
    axes[0, 0].bar(x - width/2, latency_means, width, yerr=latency_stds, capsize=3, 
                   label=f'Latency Component (α={avg_alpha_latency:.1f})', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, energy_means, width, yerr=energy_stds, capsize=3, 
                   label=f'Energy Component (α={avg_alpha_energy:.1f})', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Prefill Frequency Bin')
    axes[0, 0].set_ylabel('Weighted Component Value')
    axes[0, 0].set_title(f'Reward Breakdown by Prefill Frequency\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n, Adaptive Weighting)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(freqs)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1b. EDP-Weighted Breakdown (for visualization when H_eff <= 0.5) by prefill frequency
    freqs_edp = sorted(prefill_freq_data_edp.keys())
    latency_means_edp = [np.mean(prefill_freq_data_edp[f]['latency']) for f in freqs_edp]
    energy_means_edp = [np.mean(prefill_freq_data_edp[f]['energy']) for f in freqs_edp]
    latency_stds_edp = [np.std(prefill_freq_data_edp[f]['latency']) for f in freqs_edp]
    energy_stds_edp = [np.std(prefill_freq_data_edp[f]['energy']) for f in freqs_edp]
    
    x_edp = np.arange(len(freqs_edp))
    axes[0, 1].bar(x_edp - width/2, latency_means_edp, width, yerr=latency_stds_edp, capsize=3,
                   label='Latency Component (α=0.2)', color='blue', alpha=0.7)
    axes[0, 1].bar(x_edp + width/2, energy_means_edp, width, yerr=energy_stds_edp, capsize=3,
                   label='Energy Component (α=0.8)', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Prefill Frequency Bin')
    axes[0, 1].set_ylabel('Weighted Component Value')
    axes[0, 1].set_title('EDP-Weighted Mode (H_eff <= 0.5)\nby Prefill Frequency\n(Shows EDP component when H_eff is low)')
    axes[0, 1].set_xticks(x_edp)
    axes[0, 1].set_xticklabels(freqs_edp)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Reward breakdown by decode frequency bin
    decode_freq_data = {}
    decode_freq_data_edp = {}  # Separate data for EDP mode
    for d in dataset.processed_data:
        freq = d['decode_freq_bin']
        if freq not in decode_freq_data:
            decode_freq_data[freq] = {'latency': [], 'energy': [], 'reward': []}
            decode_freq_data_edp[freq] = {'latency': [], 'energy': [], 'reward': []}
        # Apply adaptive alpha weighting based on optimization mode (EDP vs throughput)
        # Use adaptive_alpha if available, otherwise fall back to fixed alpha
        alpha_latency = d.get('adaptive_alpha_latency', 1 - alpha)
        alpha_energy = d.get('adaptive_alpha_energy', alpha)
        decode_freq_data[freq]['latency'].append(alpha_latency * d['latency_component'])
        decode_freq_data[freq]['energy'].append(alpha_energy * d['energy_component'])
        decode_freq_data[freq]['reward'].append(d['reward'])
        
        # EDP-weighted breakdown (for visualization when H_eff <= 0.5)
        # Note: Actual reward uses unified function, this is just for visualization
        decode_freq_data_edp[freq]['latency'].append(0.2 * d['latency_component'])
        decode_freq_data_edp[freq]['energy'].append(0.8 * d['energy_component'])
        decode_freq_data_edp[freq]['reward'].append(d['reward'])
    
    freqs = sorted(decode_freq_data.keys())
    latency_means = [np.mean(decode_freq_data[f]['latency']) for f in freqs]
    energy_means = [np.mean(decode_freq_data[f]['energy']) for f in freqs]
    latency_stds = [np.std(decode_freq_data[f]['latency']) for f in freqs]
    energy_stds = [np.std(decode_freq_data[f]['energy']) for f in freqs]
    
    x = np.arange(len(freqs))
    axes[0, 2].bar(x - width/2, latency_means, width, yerr=latency_stds, capsize=3, 
                   label=f'Latency Component (α={avg_alpha_latency:.1f})', color='blue', alpha=0.7)
    axes[0, 2].bar(x + width/2, energy_means, width, yerr=energy_stds, capsize=3, 
                   label=f'Energy Component (α={avg_alpha_energy:.1f})', color='red', alpha=0.7)
    axes[0, 2].set_xlabel('Decode Frequency Bin')
    axes[0, 2].set_ylabel('Weighted Component Value')
    axes[0, 2].set_title(f'Reward Breakdown by Decode Frequency\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n, Adaptive Weighting)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(freqs)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 2b. EDP-Weighted Breakdown (for visualization when H_eff <= 0.5) by decode frequency
    freqs_decode_edp = sorted(decode_freq_data_edp.keys())
    latency_means_decode_edp = [np.mean(decode_freq_data_edp[f]['latency']) for f in freqs_decode_edp]
    energy_means_decode_edp = [np.mean(decode_freq_data_edp[f]['energy']) for f in freqs_decode_edp]
    latency_stds_decode_edp = [np.std(decode_freq_data_edp[f]['latency']) for f in freqs_decode_edp]
    energy_stds_decode_edp = [np.std(decode_freq_data_edp[f]['energy']) for f in freqs_decode_edp]
    
    x_decode_edp = np.arange(len(freqs_decode_edp))
    axes[1, 0].bar(x_decode_edp - width/2, latency_means_decode_edp, width, yerr=latency_stds_decode_edp, capsize=3,
                   label='Latency Component (α=0.2)', color='blue', alpha=0.7)
    axes[1, 0].bar(x_decode_edp + width/2, energy_means_decode_edp, width, yerr=energy_stds_decode_edp, capsize=3,
                   label='Energy Component (α=0.8)', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Decode Frequency Bin')
    axes[1, 0].set_ylabel('Weighted Component Value')
    axes[1, 0].set_title('EDP-Weighted Mode (H_eff <= 0.5)\nby Decode Frequency\n(Shows EDP component when H_eff is low)')
    axes[1, 0].set_xticks(x_decode_edp)
    axes[1, 0].set_xticklabels(freqs_decode_edp)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Reward distribution by prefill frequency bin (REMOVED - duplicate of breakdown plot)
    # The breakdown plots already show reward by frequency
    
    # 4. Reward distribution by decode frequency bin (REMOVED - duplicate of breakdown plot)  
    # The breakdown plots already show reward by frequency
    
    # 5. Reward distribution by batch size
    batch_rewards = {}
    for d in dataset.processed_data:
        batch = d['batch_size']
        if batch not in batch_rewards:
            batch_rewards[batch] = []
        batch_rewards[batch].append(d['reward'])
    
    batches = sorted(batch_rewards.keys())
    reward_means = [np.mean(batch_rewards[b]) for b in batches]
    reward_stds = [np.std(batch_rewards[b]) for b in batches]
    
    axes[1, 1].bar(batches, reward_means, yerr=reward_stds, capsize=3, color='green')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Mean Reward')
    axes[1, 1].set_title('Reward by Batch Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Energy vs Latency scatter - assuming H_eff = 0 (EDP-only) and H_eff = 1 (Throughput-only)
    energies = [d['energy'] for d in dataset.processed_data]
    latencies = [d['latency'] for d in dataset.processed_data]
    
    # Get normalization ranges for calculating what-if rewards
    throughput_min = dataset.throughput_min if hasattr(dataset, 'throughput_min') else None
    throughput_max = dataset.throughput_max if hasattr(dataset, 'throughput_max') else None
    edp_min = dataset.edp_efficiency_min if hasattr(dataset, 'edp_efficiency_min') else None
    edp_max = dataset.edp_efficiency_max if hasattr(dataset, 'edp_efficiency_max') else None
    
    # Calculate what-if rewards assuming H_eff = 0 (EDP-only) and H_eff = 1 (Throughput-only)
    rewards_heff_0 = []  # EDP-only rewards (H_eff = 0)
    rewards_heff_1 = []  # Throughput-only rewards (H_eff = 1)
    
    for d in dataset.processed_data:
        # Get raw metrics
        throughput_raw = d.get('throughput_raw', 0.0)
        energy_total = d.get('energy', 0.0)
        end_to_end_latency = d.get('end_to_end_latency', 0.0)
        avg_power = d.get('power', 0.0)
        avg_temp = d.get('temperature', 50.0)
        
        # Calculate EDP efficiency
        if energy_total > 0 and end_to_end_latency > 0:
            edp_efficiency_raw = 1.0 / (energy_total * end_to_end_latency)
        else:
            edp_efficiency_raw = 0.0
        
        # Normalize throughput
        if throughput_min is not None and throughput_max is not None and throughput_max > throughput_min:
            throughput_normalized = np.clip((throughput_raw - throughput_min) / (throughput_max - throughput_min), 0.0, 1.0)
        else:
            throughput_normalized = np.clip(throughput_raw / 200.0, 0.0, 1.0)
        
        # Normalize EDP efficiency
        if edp_min is not None and edp_max is not None and edp_max > edp_min:
            edp_efficiency_normalized = np.clip((edp_efficiency_raw - edp_min) / (edp_max - edp_min), 0.0, 1.0)
        else:
            edp_efficiency_normalized = np.clip(edp_efficiency_raw / 0.001, 0.0, 1.0)
        
        # Calculate constraint penalties (same for both modes)
        slo_target = dataset.slo_target
        max_temp = dataset.max_temp
        max_power = dataset.max_power
        
        # Calculate overages
        temp_min = 40.0
        power_min = 10.0
        slo_max_over = 1.5 * slo_target if slo_target > 0 else 15.0
        temp_max_over = 1.2 * max_temp
        power_max_over = 1.3 * max_power
        
        # Latency overage
        if slo_max_over > slo_target:
            x_L = np.clip((end_to_end_latency - slo_target) / (slo_max_over - slo_target), 0.0, 1.0)
        else:
            x_L = 1.0 if end_to_end_latency > slo_target else 0.0
        
        # Temperature overage
        if temp_max_over > max_temp:
            x_T = np.clip((avg_temp - max_temp) / (temp_max_over - max_temp), 0.0, 1.0)
        else:
            x_T = 1.0 if avg_temp > max_temp else 0.0
        
        # Power overage
        if power_max_over > max_power:
            x_P = np.clip((avg_power - max_power) / (power_max_over - max_power), 0.0, 1.0)
        else:
            x_P = 1.0 if avg_power > max_power else 0.0
        
        # Exponential penalty function
        k_penalty = 5.0
        lambda_latency = 1.0
        lambda_temp = 1.0
        lambda_power = 1.0
        
        def exp_penalty(x, k):
            if k <= 0:
                return x
            exp_k = np.exp(k)
            if x <= 0:
                return 0.0
            return (np.exp(k * x) - 1.0) / (exp_k - 1.0)
        
        penalty_latency = lambda_latency * exp_penalty(x_L, k_penalty)
        penalty_temp = lambda_temp * exp_penalty(x_T, k_penalty)
        penalty_power = lambda_power * exp_penalty(x_P, k_penalty)
        constraint_penalty = penalty_latency + penalty_temp + penalty_power
        
        # H_eff = 0: EDP-only reward (no throughput component)
        reward_heff_0 = edp_efficiency_normalized - constraint_penalty
        rewards_heff_0.append(reward_heff_0)
        
        # H_eff = 1: Throughput-only reward (no EDP component)
        reward_heff_1 = throughput_normalized - constraint_penalty
        rewards_heff_1.append(reward_heff_1)
    
    # Energy vs Latency - H_eff = 0 (EDP-only mode) at axes[1, 2]
    if len(energies) > 0:
        scatter_heff_0 = axes[1, 2].scatter(energies, latencies, c=rewards_heff_0, 
                                           cmap='viridis', alpha=0.6, s=12, edgecolors='black', linewidth=0.3)
        axes[1, 2].set_xlabel('Energy (J)', fontweight='bold')
        axes[1, 2].set_ylabel('Latency (s/token)', fontweight='bold')
        axes[1, 2].set_title('Energy vs Latency (H_eff = 0)\n(EDP-only mode, colored by EDP-only reward)', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter_heff_0, ax=axes[1, 2], label='EDP-Only Reward')
    else:
        axes[1, 2].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Energy vs Latency (H_eff = 0, EDP-only)')
    
    # Energy vs Latency - H_eff = 1 (Throughput-only mode) at axes[5, 2]
    if len(energies) > 0:
        scatter_heff_1 = axes[5, 2].scatter(energies, latencies, c=rewards_heff_1, 
                                           cmap='viridis', alpha=0.6, s=12, edgecolors='black', linewidth=0.3)
        axes[5, 2].set_xlabel('Energy (J)', fontweight='bold')
        axes[5, 2].set_ylabel('Latency (s/token)', fontweight='bold')
        axes[5, 2].set_title('Energy vs Latency (H_eff = 1)\n(Throughput-only mode, colored by Throughput-only reward)', fontweight='bold')
        axes[5, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter_heff_1, ax=axes[5, 2], label='Throughput-Only Reward')
    else:
        axes[5, 2].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[5, 2].transAxes)
        axes[5, 2].set_title('Energy vs Latency (H_eff = 1, Throughput-only)')
    
    # 7. Energy vs Power scatter
    powers = [d['power'] for d in dataset.processed_data]
    
    scatter2 = axes[2, 0].scatter(energies, powers, c=rewards, cmap='viridis', alpha=0.5, s=10)
    axes[2, 0].set_xlabel('Energy')
    axes[2, 0].set_ylabel('Power')
    axes[2, 0].set_title('Energy vs Power (colored by reward)')
    plt.colorbar(scatter2, ax=axes[2, 0], label='Reward')
    
    # 8. Heatmap of max reward by (prefill_freq, decode_freq)
    # Only use VALID actions (meeting all constraints) to avoid low rewards from violations
    action_stats = dataset.get_action_statistics()
    
    reward_matrix = np.full((freq_bins, freq_bins), np.nan)  # Initialize with NaN
    count_matrix = np.zeros((freq_bins, freq_bins))
    valid_count_matrix = np.zeros((freq_bins, freq_bins))  # Count of valid samples
    
    # Collect rewards per (prefill, decode) pair, only from valid samples
    for d in dataset.processed_data:
        prefill_freq = int(d['prefill_freq_bin'])
        decode_freq = int(d['decode_freq_bin'])
        if prefill_freq < freq_bins and decode_freq < freq_bins:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                # Track maximum reward for this (prefill, decode) pair
                if np.isnan(reward_matrix[prefill_freq, decode_freq]):
                    reward_matrix[prefill_freq, decode_freq] = d['reward']
                else:
                    reward_matrix[prefill_freq, decode_freq] = max(reward_matrix[prefill_freq, decode_freq], d['reward'])
                valid_count_matrix[prefill_freq, decode_freq] += 1
            # Count all samples for reference
            count_matrix[prefill_freq, decode_freq] += 1
    
    # Create masked array for NaN values
    masked_reward_matrix = np.ma.masked_invalid(reward_matrix)
    
    # Use pcolormesh instead of imshow to get better cell boundaries with visible edges
    # Create edge coordinates for pcolormesh (one more point than data)
    x_edges = np.arange(-0.5, freq_bins + 0.5, 1)
    y_edges = np.arange(-0.5, freq_bins + 0.5, 1)
    X, Y = np.meshgrid(x_edges, y_edges)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_for_plot = np.array(masked_reward_matrix)
    
    # Use pcolormesh with edgecolors for clear cell separation
    im = axes[2, 1].pcolormesh(X, Y, reward_matrix_for_plot, cmap='viridis', 
                               vmin=0, vmax=1.0, edgecolors='white', linewidths=1.2, 
                               shading='flat', alpha=1.0)
    axes[2, 1].set_xlabel('Decode Frequency Bin', fontweight='bold')
    axes[2, 1].set_ylabel('Prefill Frequency Bin', fontweight='bold')
    axes[2, 1].set_title('Max Reward: Prefill vs Decode Frequency\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n - penalties)', fontweight='bold')
    
    # Set proper tick marks at integer positions (center of each cell)
    tick_positions = np.arange(0, freq_bins, max(1, (freq_bins - 1) // 5))
    if tick_positions[-1] != freq_bins - 1:
        tick_positions = np.append(tick_positions, freq_bins - 1)
    tick_labels = [str(int(t)) for t in tick_positions]
    axes[2, 1].set_xticks(tick_positions)
    axes[2, 1].set_xticklabels(tick_labels)
    axes[2, 1].set_yticks(tick_positions)
    axes[2, 1].set_yticklabels(tick_labels)
    axes[2, 1].set_xlim(-0.5, freq_bins - 0.5)
    axes[2, 1].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations showing reward values on each cell
    for i in range(freq_bins):
        for j in range(freq_bins):
            if not np.isnan(reward_matrix[i, j]):
                # Show reward value on each cell
                reward_val = reward_matrix[i, j]
                # Choose text color based on cell brightness
                text_color = 'white' if reward_val > 0.5 else 'black'
                axes[2, 1].text(j, i, f'{reward_val:.2f}', 
                              ha='center', va='center', fontsize=7, 
                              color=text_color, weight='bold')
            elif valid_count_matrix[i, j] > 0 and valid_count_matrix[i, j] < 3:
                # Low sample count - mark with small text
                axes[2, 1].text(j, i, f'n={int(valid_count_matrix[i, j])}', 
                              ha='center', va='center', fontsize=6, 
                              color='gray', weight='bold')
    
    plt.colorbar(im, ax=axes[2, 1], label='Max Reward (valid only)')
    
    # 8b. Heatmap of mean reward by (prefill_freq, decode_freq)
    # Same as max reward but calculating mean instead
    reward_matrix_mean = np.zeros((freq_bins, freq_bins))
    valid_count_matrix_mean = np.zeros((freq_bins, freq_bins))
    
    # Collect rewards per (prefill, decode) pair, only from valid samples
    for d in dataset.processed_data:
        prefill_freq = int(d['prefill_freq_bin'])
        decode_freq = int(d['decode_freq_bin'])
        if prefill_freq < freq_bins and decode_freq < freq_bins:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                reward_matrix_mean[prefill_freq, decode_freq] += d['reward']
                valid_count_matrix_mean[prefill_freq, decode_freq] += 1
    
    # Calculate mean reward from valid samples only
    # If no valid samples, use NaN (will show as white/transparent)
    for i in range(freq_bins):
        for j in range(freq_bins):
            if valid_count_matrix_mean[i, j] > 0:
                reward_matrix_mean[i, j] = reward_matrix_mean[i, j] / valid_count_matrix_mean[i, j]
            else:
                reward_matrix_mean[i, j] = np.nan  # No valid samples
    
    # Create masked array for NaN values
    masked_reward_matrix_mean = np.ma.masked_invalid(reward_matrix_mean)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_mean_for_plot = np.array(masked_reward_matrix_mean)
    
    # Use pcolormesh with edgecolors for clear cell separation
    im_mean = axes[2, 2].pcolormesh(X, Y, reward_matrix_mean_for_plot, cmap='viridis', 
                                    vmin=0, vmax=1.0, edgecolors='white', linewidths=1.2, 
                                    shading='flat', alpha=1.0)
    axes[2, 2].set_xlabel('Decode Frequency Bin', fontweight='bold')
    axes[2, 2].set_ylabel('Prefill Frequency Bin', fontweight='bold')
    axes[2, 2].set_title('Mean Reward: Prefill vs Decode Frequency\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n - penalties)', fontweight='bold')
    
    # Set proper tick marks at integer positions (center of each cell)
    axes[2, 2].set_xticks(tick_positions)
    axes[2, 2].set_xticklabels(tick_labels)
    axes[2, 2].set_yticks(tick_positions)
    axes[2, 2].set_yticklabels(tick_labels)
    axes[2, 2].set_xlim(-0.5, freq_bins - 0.5)
    axes[2, 2].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations showing reward values on each cell
    for i in range(freq_bins):
        for j in range(freq_bins):
            if not np.isnan(reward_matrix_mean[i, j]):
                # Show reward value on each cell
                reward_val = reward_matrix_mean[i, j]
                # Choose text color based on cell brightness
                text_color = 'white' if reward_val > 0.5 else 'black'
                axes[2, 2].text(j, i, f'{reward_val:.2f}', 
                               ha='center', va='center', fontsize=7, 
                               color=text_color, weight='bold')
            elif valid_count_matrix_mean[i, j] > 0 and valid_count_matrix_mean[i, j] < 3:
                # Low sample count - mark with small text
                axes[2, 2].text(j, i, f'n={int(valid_count_matrix_mean[i, j])}', 
                               ha='center', va='center', fontsize=6, 
                               color='gray', weight='bold')
    
    plt.colorbar(im_mean, ax=axes[2, 2], label='Mean Reward (valid only)')
    
    # 8c. Heatmap of mean EDP-only reward by (prefill_freq, decode_freq)
    # Calculate what reward would be if we ONLY used EDP efficiency (no adaptive weighting)
    reward_matrix_edp_only = np.zeros((freq_bins, freq_bins))
    valid_count_matrix_edp_only = np.zeros((freq_bins, freq_bins))
    
    # First, calculate min/max EDP efficiency for normalization from raw data
    edp_efficiency_values = []
    for d in dataset.processed_data:
        # Calculate EDP efficiency from stored energy and latency
        # Need to get energy_total - check if it's stored, otherwise estimate
        # Note: end_to_end_latency is per-request latency (total time), not per-token latency
        energy_total = d.get('energy', 0.0)  # energy field might be total energy
        end_to_end_latency = d.get('end_to_end_latency', 0.0)  # per-request latency (seconds)
        if energy_total > 0 and end_to_end_latency > 0:
            edp_efficiency_raw = 1.0 / (energy_total * end_to_end_latency)
            edp_efficiency_values.append(edp_efficiency_raw)
    
    # Get normalization ranges (use dataset's ranges if available, otherwise calculate)
    edp_min = dataset.edp_efficiency_min if hasattr(dataset, 'edp_efficiency_min') and dataset.edp_efficiency_min is not None else (min(edp_efficiency_values) if edp_efficiency_values else 0.0)
    edp_max = dataset.edp_efficiency_max if hasattr(dataset, 'edp_efficiency_max') and dataset.edp_efficiency_max is not None else (max(edp_efficiency_values) if edp_efficiency_values else 0.001)
    
    # Collect EDP-only rewards per (prefill, decode) pair, only from valid samples
    for d in dataset.processed_data:
        prefill_freq = int(d['prefill_freq_bin'])
        decode_freq = int(d['decode_freq_bin'])
        if prefill_freq < freq_bins and decode_freq < freq_bins:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                # Calculate EDP-only reward (always use EDP efficiency, not adaptive)
                # Calculate EDP efficiency from stored data
                # Note: end_to_end_latency is per-request latency (total time), not per-token latency
                energy_total = d.get('energy', 0.0)
                end_to_end_latency = d.get('end_to_end_latency', 0.0)  # per-request latency (seconds)
                
                if energy_total > 0 and end_to_end_latency > 0:
                    edp_efficiency_raw = 1.0 / (energy_total * end_to_end_latency)
                    
                    # Normalize EDP efficiency to [0, 1]
                    if edp_max > edp_min:
                        edp_efficiency_normalized = (edp_efficiency_raw - edp_min) / (edp_max - edp_min)
                    else:
                        edp_efficiency_normalized = min(1.0, max(0.0, edp_efficiency_raw / 0.001))
                    
                    # Clamp to [0, 1]
                    edp_efficiency_normalized = min(1.0, max(0.0, edp_efficiency_normalized))
                    
                    # EDP-only reward: use EDP efficiency normalized (no penalties for valid samples)
                    # Valid samples already meet all constraints, so penalties = 0
                    edp_only_reward = edp_efficiency_normalized
                    
                    reward_matrix_edp_only[prefill_freq, decode_freq] += edp_only_reward
                    valid_count_matrix_edp_only[prefill_freq, decode_freq] += 1
    
    # Calculate mean EDP-only reward from valid samples only
    # If no valid samples, use NaN (will show as white/transparent)
    for i in range(freq_bins):
        for j in range(freq_bins):
            if valid_count_matrix_edp_only[i, j] > 0:
                reward_matrix_edp_only[i, j] = reward_matrix_edp_only[i, j] / valid_count_matrix_edp_only[i, j]
            else:
                reward_matrix_edp_only[i, j] = np.nan  # No valid samples
    
    # Create masked array for NaN values
    masked_reward_matrix_edp_only = np.ma.masked_invalid(reward_matrix_edp_only)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_edp_only_for_plot = np.array(masked_reward_matrix_edp_only)
    
    # Find min/max for color scale (may be different from adaptive reward)
    valid_values = reward_matrix_edp_only_for_plot[~np.isnan(reward_matrix_edp_only_for_plot)]
    vmin_edp = np.min(valid_values) if len(valid_values) > 0 else 0.0
    vmax_edp = np.max(valid_values) if len(valid_values) > 0 else 1.0
    
    # Use pcolormesh with edgecolors for clear cell separation
    im_edp_only = axes[4, 1].pcolormesh(X, Y, reward_matrix_edp_only_for_plot, cmap='viridis', 
                                        vmin=vmin_edp, vmax=vmax_edp, edgecolors='white', linewidths=1.2, 
                                        shading='flat', alpha=1.0)
    axes[4, 1].set_xlabel('Decode Frequency Bin', fontweight='bold')
    axes[4, 1].set_ylabel('Prefill Frequency Bin', fontweight='bold')
    axes[4, 1].set_title('Mean EDP-Only Reward: Prefill vs Decode Frequency\n(What-if: H_eff=0, EDP_n only, no throughput component)', fontweight='bold')
    
    # Set proper tick marks at integer positions (center of each cell)
    axes[4, 1].set_xticks(tick_positions)
    axes[4, 1].set_xticklabels(tick_labels)
    axes[4, 1].set_yticks(tick_positions)
    axes[4, 1].set_yticklabels(tick_labels)
    axes[4, 1].set_xlim(-0.5, freq_bins - 0.5)
    axes[4, 1].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations showing reward values on each cell
    for i in range(freq_bins):
        for j in range(freq_bins):
            if not np.isnan(reward_matrix_edp_only[i, j]):
                # Show reward value on each cell
                reward_val = reward_matrix_edp_only[i, j]
                # Choose text color based on cell brightness
                text_color = 'white' if reward_val > (vmin_edp + vmax_edp) / 2 else 'black'
                axes[4, 1].text(j, i, f'{reward_val:.2f}', 
                               ha='center', va='center', fontsize=7, 
                               color=text_color, weight='bold')
            elif valid_count_matrix_edp_only[i, j] > 0 and valid_count_matrix_edp_only[i, j] < 3:
                # Low sample count - mark with small text
                axes[4, 1].text(j, i, f'n={int(valid_count_matrix_edp_only[i, j])}', 
                               ha='center', va='center', fontsize=6, 
                               color='gray', weight='bold')
    
    plt.colorbar(im_edp_only, ax=axes[4, 1], label='Mean EDP-Only Reward (valid only)')
    
    # 9. Heatmap of mean reward by (prefill_freq, batch_size)
    # Only use VALID actions (meeting all constraints) to avoid low rewards from violations
    reward_matrix_batch = np.zeros((freq_bins, max_batch))
    valid_count_matrix_batch = np.zeros((freq_bins, max_batch))
    
    # Collect rewards per (prefill, batch) pair, only from valid samples
    for d in dataset.processed_data:
        prefill_freq = int(d['prefill_freq_bin'])
        batch_size = int(d['batch_size'])
        if prefill_freq < freq_bins and 1 <= batch_size <= max_batch:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                reward_matrix_batch[prefill_freq, batch_size - 1] += d['reward']
                valid_count_matrix_batch[prefill_freq, batch_size - 1] += 1
    
    # Calculate mean reward from valid samples only
    # If no valid samples, use NaN (will show as white/transparent)
    for i in range(freq_bins):
        for j in range(max_batch):
            if valid_count_matrix_batch[i, j] > 0:
                reward_matrix_batch[i, j] = reward_matrix_batch[i, j] / valid_count_matrix_batch[i, j]
            else:
                reward_matrix_batch[i, j] = np.nan  # No valid samples
    
    # Create masked array for NaN values
    masked_reward_matrix_batch = np.ma.masked_invalid(reward_matrix_batch)
    
    # Create edge coordinates for pcolormesh
    x_edges_batch = np.arange(-0.5, max_batch + 0.5, 1)
    y_edges_batch = np.arange(-0.5, freq_bins + 0.5, 1)
    X_batch, Y_batch = np.meshgrid(x_edges_batch, y_edges_batch)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_batch_for_plot = np.array(masked_reward_matrix_batch)
    
    # Use pcolormesh with edgecolors for clear cell separation
    im_batch = axes[3, 0].pcolormesh(X_batch, Y_batch, reward_matrix_batch_for_plot, cmap='viridis', 
                                     vmin=0, vmax=1.0, edgecolors='white', linewidths=1.2, 
                                     shading='flat', alpha=1.0)
    axes[3, 0].set_xlabel('Batch Size', fontweight='bold')
    axes[3, 0].set_ylabel('Prefill Frequency Bin', fontweight='bold')
    axes[3, 0].set_title('Mean Reward: Prefill Frequency vs Batch Size\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n - penalties)', fontweight='bold')
    
    # Set proper tick marks for batch size (every few bins)
    batch_tick_positions = np.arange(0, max_batch, max(1, max_batch // 8))
    if batch_tick_positions[-1] != max_batch - 1:
        batch_tick_positions = np.append(batch_tick_positions, max_batch - 1)
    batch_tick_labels = [str(int(t + 1)) for t in batch_tick_positions]  # batch_size is 1-indexed
    
    # Use same tick positions as other heatmaps for y-axis
    freq_tick_positions = np.arange(0, freq_bins, max(1, (freq_bins - 1) // 5))
    if freq_tick_positions[-1] != freq_bins - 1:
        freq_tick_positions = np.append(freq_tick_positions, freq_bins - 1)
    freq_tick_labels = [str(int(t)) for t in freq_tick_positions]
    
    axes[3, 0].set_xticks(batch_tick_positions)
    axes[3, 0].set_xticklabels(batch_tick_labels)
    axes[3, 0].set_yticks(freq_tick_positions)
    axes[3, 0].set_yticklabels(freq_tick_labels)
    axes[3, 0].set_xlim(-0.5, max_batch - 0.5)
    axes[3, 0].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations for cells with low sample counts (might be unreliable)
    for i in range(freq_bins):
        for j in range(max_batch):
            if valid_count_matrix_batch[i, j] > 0 and valid_count_matrix_batch[i, j] < 3:
                # Low sample count - mark with small text
                axes[3, 0].text(j, i, f'n={int(valid_count_matrix_batch[i, j])}', 
                               ha='center', va='center', fontsize=5, 
                               color='white' if reward_matrix_batch[i, j] > 0.5 else 'black',
                               weight='bold')
    
    plt.colorbar(im_batch, ax=axes[3, 0], label='Mean Reward (valid only)')
    
    # 9b. Heatmap of mean reward by (decode_freq, batch_size)
    # Only use VALID actions (meeting all constraints) to avoid low rewards from violations
    reward_matrix_decode_batch = np.zeros((freq_bins, max_batch))
    valid_count_matrix_decode_batch = np.zeros((freq_bins, max_batch))
    
    # Collect rewards per (decode, batch) pair, only from valid samples
    for d in dataset.processed_data:
        decode_freq = int(d['decode_freq_bin'])
        batch_size = int(d['batch_size'])
        if decode_freq < freq_bins and 1 <= batch_size <= max_batch:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                reward_matrix_decode_batch[decode_freq, batch_size - 1] += d['reward']
                valid_count_matrix_decode_batch[decode_freq, batch_size - 1] += 1
    
    # Calculate mean reward from valid samples only
    # If no valid samples, use NaN (will show as white/transparent)
    for i in range(freq_bins):
        for j in range(max_batch):
            if valid_count_matrix_decode_batch[i, j] > 0:
                reward_matrix_decode_batch[i, j] = reward_matrix_decode_batch[i, j] / valid_count_matrix_decode_batch[i, j]
            else:
                reward_matrix_decode_batch[i, j] = np.nan  # No valid samples
    
    # Create masked array for NaN values
    masked_reward_matrix_decode_batch = np.ma.masked_invalid(reward_matrix_decode_batch)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_decode_batch_for_plot = np.array(masked_reward_matrix_decode_batch)
    
    # Use pcolormesh with edgecolors for clear cell separation
    im_decode_batch = axes[3, 1].pcolormesh(X_batch, Y_batch, reward_matrix_decode_batch_for_plot, cmap='viridis', 
                                            vmin=0, vmax=1.0, edgecolors='white', linewidths=1.2, 
                                            shading='flat', alpha=1.0)
    axes[3, 1].set_xlabel('Batch Size', fontweight='bold')
    axes[3, 1].set_ylabel('Decode Frequency Bin', fontweight='bold')
    axes[3, 1].set_title('Mean Reward: Decode Frequency vs Batch Size\n(Unified: H_eff * T_n + (1-H_eff) * EDP_n - penalties)', fontweight='bold')
    
    # Set proper tick marks (same as prefill vs batch)
    axes[3, 1].set_xticks(batch_tick_positions)
    axes[3, 1].set_xticklabels(batch_tick_labels)
    axes[3, 1].set_yticks(freq_tick_positions)
    axes[3, 1].set_yticklabels(freq_tick_labels)
    axes[3, 1].set_xlim(-0.5, max_batch - 0.5)
    axes[3, 1].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations for cells with low sample counts (might be unreliable)
    for i in range(freq_bins):
        for j in range(max_batch):
            if valid_count_matrix_decode_batch[i, j] > 0 and valid_count_matrix_decode_batch[i, j] < 3:
                # Low sample count - mark with small text
                axes[3, 1].text(j, i, f'n={int(valid_count_matrix_decode_batch[i, j])}', 
                               ha='center', va='center', fontsize=5, 
                               color='white' if reward_matrix_decode_batch[i, j] > 0.5 else 'black',
                               weight='bold')
    
    plt.colorbar(im_decode_batch, ax=axes[3, 1], label='Mean Reward (valid only)')
    
    # Additional plots for GRPO analysis
    # Fill vacant subplot at axes[3, 2] with Reward Distribution
    # 10. Distribution of rewards - moved to row 3, column 2
    axes[3, 2].hist(rewards, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[3, 2].set_xlabel('Reward')
    axes[3, 2].set_ylabel('Count')
    axes[3, 2].set_title('Unified Reward Distribution\n(R = H_eff * T_n + (1-H_eff) * EDP_n - penalties)')
    axes[3, 2].grid(True, alpha=0.3)
    
    # 11. Latency component vs Energy component (adaptively weighted) - moved to row 4, column 0
    latency_comps = [d.get('adaptive_alpha_latency', 1 - alpha) * d['latency_component'] for d in dataset.processed_data]
    energy_comps = [d.get('adaptive_alpha_energy', alpha) * d['energy_component'] for d in dataset.processed_data]
    
    scatter3 = axes[4, 0].scatter(latency_comps, energy_comps, c=rewards, cmap='viridis', alpha=0.5, s=10)
    axes[4, 0].set_xlabel(f'Adaptively Weighted Latency Component (avg α={avg_alpha_latency:.1f})')
    axes[4, 0].set_ylabel(f'Adaptively Weighted Energy Component (avg α={avg_alpha_energy:.1f})')
    axes[4, 0].set_title('Unified Reward Components\n(H_eff * T_n vs (1-H_eff) * EDP_n)')
    plt.colorbar(scatter3, ax=axes[4, 0], label='Reward')
    
    # 12. Throughput vs Reward scatter plot - moved to row 4
    throughputs = []
    throughputs_normalized = []
    rewards_throughput = []
    
    for d in dataset.processed_data:
        if 'throughput_tokens_per_sec' in d:
            throughputs.append(d['throughput_tokens_per_sec'])
        else:
            # Fallback: calculate from available data
            total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
            total_time = d['end_to_end_latency']
            throughput_tps = total_tokens / total_time if total_time > 0 else 0
            throughputs.append(throughput_tps)
        
        throughputs_normalized.append(d.get('throughput_normalized', 0.0))
        rewards_throughput.append(d['reward'])
    
    # Plot both raw throughput (tokens/sec) and normalized throughput vs reward
    # Use normalized throughput for better visualization
    scatter4 = axes[4, 2].scatter(throughputs_normalized, rewards_throughput, 
                                  c=throughputs, cmap='plasma', alpha=0.6, s=15, edgecolors='black', linewidth=0.5)
    axes[4, 2].set_xlabel('Normalized Throughput [0, 1]')
    axes[4, 2].set_ylabel('Reward')
    axes[4, 2].set_title('Throughput vs Unified Reward\n(R = H_eff * T_n + (1-H_eff) * EDP_n - penalties)')
    axes[4, 2].grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=axes[4, 2])
    cbar4.set_label('Throughput (tokens/sec)', rotation=270, labelpad=15)
    
    # Add correlation coefficient
    if len(throughputs_normalized) > 1:
        corr_coef = np.corrcoef(throughputs_normalized, rewards_throughput)[0, 1]
        axes[4, 2].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                        transform=axes[4, 2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 13. Throughput vs Energy scatter plot - row 5, column 0
    throughputs_energy = []
    energies_energy = []
    rewards_energy = []
    
    for d in dataset.processed_data:
        if 'throughput_tokens_per_sec' in d:
            throughputs_energy.append(d['throughput_tokens_per_sec'])
        else:
            # Fallback: calculate from available data
            total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
            total_time = d['end_to_end_latency']
            throughput_tps = total_tokens / total_time if total_time > 0 else 0
            throughputs_energy.append(throughput_tps)
        
        energies_energy.append(d.get('energy', 0.0))
        rewards_energy.append(d['reward'])
    
    scatter5 = axes[5, 0].scatter(throughputs_energy, energies_energy, c=rewards_energy, 
                                  cmap='viridis', alpha=0.6, s=15, edgecolors='black', linewidth=0.5)
    axes[5, 0].set_xlabel('Throughput (tokens/sec)', fontweight='bold')
    axes[5, 0].set_ylabel('Energy (J)', fontweight='bold')
    axes[5, 0].set_title('Throughput vs Energy\n(colored by unified reward)')
    axes[5, 0].grid(True, alpha=0.3)
    cbar5 = plt.colorbar(scatter5, ax=axes[5, 0], label='Reward')
    
    # 14. Heatmap of mean throughput-only reward by (prefill_freq, decode_freq)
    # Calculate what reward would be if we ONLY used throughput (no adaptive weighting)
    reward_matrix_throughput_only = np.zeros((freq_bins, freq_bins))
    valid_count_matrix_throughput_only = np.zeros((freq_bins, freq_bins))
    
    # Collect throughput-only rewards per (prefill, decode) pair, only from valid samples
    for d in dataset.processed_data:
        prefill_freq = int(d['prefill_freq_bin'])
        decode_freq = int(d['decode_freq_bin'])
        if prefill_freq < freq_bins and decode_freq < freq_bins:
            # Only count samples that meet ALL constraints
            if not d['slo_violation'] and not d['temp_violation'] and not d['power_violation']:
                # Calculate throughput-only reward (always use throughput, not adaptive)
                # Throughput normalized is already in the processed data
                throughput_normalized = d.get('throughput_normalized', 0.0)
                
                # Throughput-only reward: use throughput normalized (no penalties for valid samples)
                # Valid samples already meet all constraints, so penalties = 0
                throughput_only_reward = throughput_normalized
                
                reward_matrix_throughput_only[prefill_freq, decode_freq] += throughput_only_reward
                valid_count_matrix_throughput_only[prefill_freq, decode_freq] += 1
    
    # Calculate mean throughput-only reward from valid samples only
    # If no valid samples, use NaN (will show as white/transparent)
    for i in range(freq_bins):
        for j in range(freq_bins):
            if valid_count_matrix_throughput_only[i, j] > 0:
                reward_matrix_throughput_only[i, j] = reward_matrix_throughput_only[i, j] / valid_count_matrix_throughput_only[i, j]
            else:
                reward_matrix_throughput_only[i, j] = np.nan  # No valid samples
    
    # Create masked array for NaN values
    masked_reward_matrix_throughput_only = np.ma.masked_invalid(reward_matrix_throughput_only)
    
    # Convert masked array to regular array with NaN handling for pcolormesh
    reward_matrix_throughput_only_for_plot = np.array(masked_reward_matrix_throughput_only)
    
    # Find min/max for color scale (may be different from adaptive reward)
    valid_values_throughput = reward_matrix_throughput_only_for_plot[~np.isnan(reward_matrix_throughput_only_for_plot)]
    vmin_throughput = np.min(valid_values_throughput) if len(valid_values_throughput) > 0 else 0.0
    vmax_throughput = np.max(valid_values_throughput) if len(valid_values_throughput) > 0 else 1.0
    
    # Use pcolormesh with edgecolors for clear cell separation
    im_throughput_only = axes[5, 1].pcolormesh(X, Y, reward_matrix_throughput_only_for_plot, cmap='viridis', 
                                                vmin=vmin_throughput, vmax=vmax_throughput, edgecolors='white', linewidths=1.2, 
                                                shading='flat', alpha=1.0)
    axes[5, 1].set_xlabel('Decode Frequency Bin', fontweight='bold')
    axes[5, 1].set_ylabel('Prefill Frequency Bin', fontweight='bold')
    axes[5, 1].set_title('Mean Throughput-Only Reward: Prefill vs Decode Frequency\n(What-if: H_eff=1.0, T_n only, no EDP component)', fontweight='bold')
    
    # Set proper tick marks at integer positions (center of each cell)
    axes[5, 1].set_xticks(tick_positions)
    axes[5, 1].set_xticklabels(tick_labels)
    axes[5, 1].set_yticks(tick_positions)
    axes[5, 1].set_yticklabels(tick_labels)
    axes[5, 1].set_xlim(-0.5, freq_bins - 0.5)
    axes[5, 1].set_ylim(-0.5, freq_bins - 0.5)
    
    # Add text annotations showing reward values on each cell
    for i in range(freq_bins):
        for j in range(freq_bins):
            if not np.isnan(reward_matrix_throughput_only[i, j]):
                # Show reward value on each cell
                reward_val = reward_matrix_throughput_only[i, j]
                # Choose text color based on cell brightness
                text_color = 'white' if reward_val > (vmin_throughput + vmax_throughput) / 2 else 'black'
                axes[5, 1].text(j, i, f'{reward_val:.2f}', 
                               ha='center', va='center', fontsize=7, 
                               color=text_color, weight='bold')
            elif valid_count_matrix_throughput_only[i, j] > 0 and valid_count_matrix_throughput_only[i, j] < 3:
                # Low sample count - mark with small text
                axes[5, 1].text(j, i, f'n={int(valid_count_matrix_throughput_only[i, j])}', 
                               ha='center', va='center', fontsize=6, 
                               color='gray', weight='bold')
    
    plt.colorbar(im_throughput_only, ax=axes[5, 1], label='Mean Throughput-Only Reward (valid only)')
    
    # 15. Throughput vs Latency scatter plot - row 6, column 0
    throughputs_latency = []
    latencies_latency = []
    rewards_latency = []
    
    for d in dataset.processed_data:
        if 'throughput_tokens_per_sec' in d:
            throughputs_latency.append(d['throughput_tokens_per_sec'])
        else:
            # Fallback: calculate from available data
            total_tokens = d.get('tokens_prefill', 0) + d.get('tokens_decode', 0)
            total_time = d['end_to_end_latency']
            throughput_tps = total_tokens / total_time if total_time > 0 else 0
            throughputs_latency.append(throughput_tps)
        
        latencies_latency.append(d.get('latency', 0.0))  # latency is in s/token
        rewards_latency.append(d['reward'])
    
    scatter6 = axes[6, 0].scatter(throughputs_latency, latencies_latency, c=rewards_latency, 
                                  cmap='viridis', alpha=0.6, s=15, edgecolors='black', linewidth=0.5)
    axes[6, 0].set_xlabel('Throughput (tokens/sec)', fontweight='bold')
    axes[6, 0].set_ylabel('Latency (s/token)', fontweight='bold')
    axes[6, 0].set_title('Throughput vs Latency\n(colored by unified reward)')
    axes[6, 0].grid(True, alpha=0.3)
    cbar6 = plt.colorbar(scatter6, ax=axes[6, 0], label='Reward')
    
    # 14. H_eff (Effective Headroom) Distribution
    h_eff_all = [d.get('adaptive_alpha_latency', 0.5) for d in dataset.processed_data]
    axes[6, 1].hist(h_eff_all, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[6, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='H_eff = 0.5 (threshold)')
    axes[6, 1].set_xlabel('H_eff (Effective Headroom)', fontweight='bold')
    axes[6, 1].set_ylabel('Count', fontweight='bold')
    axes[6, 1].set_title('H_eff Distribution\n(H_eff = w_T * H_T + w_P * H_P)\nDetermines throughput vs EDP weighting')
    axes[6, 1].legend()
    axes[6, 1].grid(True, alpha=0.3)
    
    # 15. H_eff vs Reward scatter plot
    scatter7 = axes[6, 2].scatter(h_eff_all, rewards, c=rewards, cmap='viridis', alpha=0.6, s=10, edgecolors='black', linewidth=0.3)
    axes[6, 2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='H_eff = 0.5')
    axes[6, 2].set_xlabel('H_eff (Effective Headroom)', fontweight='bold')
    axes[6, 2].set_ylabel('Unified Reward', fontweight='bold')
    axes[6, 2].set_title('H_eff vs Unified Reward\n(R = H_eff * T_n + (1-H_eff) * EDP_n - penalties)')
    axes[6, 2].legend()
    axes[6, 2].grid(True, alpha=0.3)
    cbar7 = plt.colorbar(scatter7, ax=axes[6, 2], label='Reward')
    
    # Add correlation coefficient
    if len(h_eff_all) > 1:
        corr_h_eff = np.corrcoef(h_eff_all, rewards)[0, 1]
        axes[6, 2].text(0.05, 0.95, f'Correlation: {corr_h_eff:.3f}', 
                        transform=axes[6, 2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Auto-generate save path if not provided
    if save_path is None:
        # Save to plots directory
        save_path = os.path.join('plots', f'early_exit_dataset_analysis.png')
    
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDataset analysis saved as '{save_path}'")
    plt.close()
    
    return dataset


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Option 1: Single CSV file
    #CSV_PATH = "/nvme/cs242-team7/grid_search_results_20251128_075945.csv"
    
    # Option 2: Multiple CSV files (uncomment to use)
    CSV_PATH = [
        "data/grid_search_results_with_entropy_and_fan_on.csv",
        "data/grid_search_results_with_entropy_and_fan_off.csv"
    ]
    
    # Legacy alpha parameter (kept for compatibility)
    # Note: Reward function now uses unified form: R = H_eff * T_n + (1 - H_eff) * EDP_n - penalties
    # H_eff adaptively weights throughput vs EDP based on temperature and power headroom
    ALPHA = 0.8  # <-- Legacy parameter (not used in unified reward calculation)
    
    # ============================================================
    # HARDWARE AND SLO CONSTRAINTS FOR JETSON
    # ============================================================
    
    SLO_TARGET = 15.0      # Maximum end-to-end latency (seconds) for chatbot responsiveness
    MAX_TEMP = 95.0      # Maximum GPU temperature (Celsius) for Jetson thermal limit
    MIN_TEMP = 40.0       # Minimum GPU temperature (Celsius) for normalization
    # MAX_POWER is defined as global variable at top of file (60.0 W)
    
    # ============================================================
    # ENVIRONMENT SELECTION
    # ============================================================
    # Set to True: Use REAL Jetson hardware sensors
    #   - Temperature: /sys/devices/virtual/thermal/thermal_zone1/temp (GPU)
    #   - Power: /sys/class/hwmon/hwmon3/ (INA3221 sensors)
    #       * VDD_GPU_SOC (GPU/SOC rail)
    #       * VDD_CPU_CV (CPU rail)
    #       * VIN_SYS_5V0 (System 5V rail)
    #   - Total Power = GPU_SOC + CPU + System
    #
    # Set to False: Use simulated environment (CSV statistics)
    # ============================================================
    
    USE_REAL_JETSON = True  # Currently using REAL hardware sensors
    
    # First, analyze the dataset
    print("="*60)
    print("ANALYZING OFFLINE DATASET")
    print(f"Unified Reward Function: R = H_eff * T_n + (1 - H_eff) * EDP_n - penalties")
    print(f"Adaptive Weighting (based on H_eff = w_T * H_T + w_P * H_P):")
    print(f"  - H_eff > 0.5: More weight on throughput (prioritize speed)")
    print(f"  - H_eff <= 0.5: More weight on EDP (prioritize energy efficiency)")
    print(f"\nHardware Constraints:")
    print(f"  - SLO Target: {SLO_TARGET}s")
    print(f"  - Max Temp: {MAX_TEMP}°C")
    print(f"  - Max Power: {MAX_POWER}W")
    print(f"\nOnline Training Mode:")
    print(f"  - {'REAL JETSON (hardware sensors)' if USE_REAL_JETSON else 'SIMULATED (CSV statistics)'}")
    if isinstance(CSV_PATH, (list, tuple)):
        print(f"\nCSV files ({len(CSV_PATH)} files):")
        for i, path in enumerate(CSV_PATH, 1):
            print(f"  {i}. {path}")
    else:
        print(f"\nCSV file: {CSV_PATH}")
    print("="*60)
    dataset = analyze_dataset(CSV_PATH, alpha=ALPHA)
    
    # Train with GRPO
    print("\n" + "="*60)
    print("TRAINING WITH GRPO AGENT FROM CHECKPOINT")
    print("="*60)
    
    # Train with GRPO from checkpoint and collect configuration statistics
    (agent, rewards, energies, latencies, e2e_latencies, 
     temperatures, powers, slo_violations, temp_violations, power_violations, 
     online_losses, pretrain_losses, configuration_stats,
     step_policy_losses, step_kl_divergences, step_policy_entropies, step_entropies, step_decode_throughput) = train_with_grpo_from_checkpoint(
        checkpoint_path='models/grpo_early_exit_model3.pth',
        csv_path=CSV_PATH,
        num_episodes=400,  
        max_steps=1,
        update_freq=6, 
        group_size=6,
        use_simulated_env=(not USE_REAL_JETSON),
        alpha=ALPHA,
        slo_target=SLO_TARGET,
        max_temp=MAX_TEMP,
        min_temp=MIN_TEMP,
        max_power=MAX_POWER
    )
    # fan_on()
    # Plot training results
    plot_training_results(rewards, energies, latencies, e2e_latencies, 
                         temperatures, powers,
                         slo_violations, temp_violations, power_violations,
                         online_losses, pretrain_losses,
                         slo_target=SLO_TARGET, max_temp=MAX_TEMP, max_power=MAX_POWER)
    # Print configuration stats summary
    if configuration_stats:
        print("\n" + "="*60)
        print("CONFIGURATION STATISTICS SUMMARY")
        print("="*60)
        print(f"Total configurations explored: {len(configuration_stats)}")
        
        # Find top configurations by average reward
        config_summaries = []
        for config_key, stats in configuration_stats.items():
            # Handle both old format (3 elements) and new format (4 elements with early_exit_layer)
            if len(config_key) == 4:
                prefill_freq, decode_freq, batch, early_exit_layer = config_key
            else:
                prefill_freq, decode_freq, batch = config_key
                early_exit_layer = 16  # Default to full model for old format
            
            avg_reward = np.mean(stats['rewards'])
            avg_e2e = np.mean(stats['e2e_latencies'])
            avg_power = np.mean(stats['powers'])
            avg_temp = np.mean(stats['temperatures'])
            avg_early_exit = np.mean(stats.get('early_exit_layers', [early_exit_layer])) if 'early_exit_layers' in stats else early_exit_layer
            
            config_summaries.append({
                'config': config_key,
                'avg_reward': avg_reward,
                'avg_e2e_latency': avg_e2e,
                'avg_power': avg_power,
                'avg_temp': avg_temp,
                'avg_early_exit_layer': avg_early_exit,
                'count': stats['count']
            })
        
        # Sort by reward
        config_summaries.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        print(f"\nTop 10 configurations by reward:")
        for i, summary in enumerate(config_summaries[:10]):
            config = summary['config']
            if len(config) == 4:
                prefill, decode, batch, early_exit = config
            else:
                prefill, decode, batch = config
                early_exit = summary['avg_early_exit_layer']
            print(f"  {i+1}. Prefill={prefill}, Decode={decode}, Batch={batch}, EarlyExit={early_exit}")
            print(f"     Reward: {summary['avg_reward']:.4f}, E2E: {summary['avg_e2e_latency']:.3f}s, "
                  f"Power: {summary['avg_power']:.2f}W, Temp: {summary['avg_temp']:.1f}°C, "
                  f"Count: {summary['count']}")
    
    # # Test the agent
    # if USE_REAL_JETSON:
    #     env = RealJetsonEnv(freq_bins=11, max_batch=1,
    #                        slo_target=SLO_TARGET, max_temp=MAX_TEMP, max_power=MAX_POWER)
    # else:
    #     env = SimulatedEnergyLatencyEnv(CSV_PATH, alpha=ALPHA, freq_bins=11, max_batch=1,
    #                                      slo_target=SLO_TARGET, max_temp=MAX_TEMP, max_power=MAX_POWER)
    
    # test_grpo_agent(agent, env, num_episodes=5, 
    #                slo_target=SLO_TARGET, max_temp=MAX_TEMP, max_power=MAX_POWER)
    
    # Save the model
    model_path = 'models/grpo_early_exit_model.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(agent.policy.state_dict(), model_path)
    print(f"\nModel saved as '{model_path}'")
    
    # # ============================================================
    # # DEMONSTRATE OUTPUT VARIABLES
    # # ============================================================
    # print("\n" + "="*60)
    # print("DEMONSTRATION: GETTING OUTPUT VARIABLES")
    # print("="*60)
    
    # # Create inference object
    # inference = GRPOInference(agent.policy, freq_bins=11, max_batch=1)
    
    # print("\n" + "="*60)
    # print("MODEL PREDICTIONS vs DATASET OPTIMAL ACTIONS")
    # print("="*60)
    # print("\nNOTE: The model learns a STATE-DEPENDENT policy.")
    # print("      Different states → Different optimal actions")
    # print("      The top dataset actions correspond to specific states.")
    # print("\n" + "-"*60)
    
    # # Show dataset top actions
    # print("\n>>> DATASET TOP ACTIONS (actual best in dataset):")
    # top_actions = dataset.get_valid_actions(top_k=3)
    
    # # Get optimal configuration for different input states
    # print("\n>>> MODEL PREDICTION (default state [0.5, 0.5, 0.5, 0.3]):")
    # print("     This state represents: Prefill~bin5, Decode~bin5, Batch~middle, TempHeadroom~30%")
    # optimal_default = inference.print_recommendation(slo_target=SLO_TARGET)
    
    # # Also query with maximum state (high frequency + high batch) to match dataset top actions
    # # Top dataset action: Prefill=10, Decode=9, Batch=32
    # # State: [10/10, 9/10, batch_idx/max_batch_idx, temp_headroom] = [1.0, 0.9, ~1.0, 0.3]
    # batch_sizes = list(range(1, 32 + 1)) #[1] + list(range(2, 32 + 1, 2))  # [1, 2, 4, 6, ..., 32]
    # batch_32_idx = batch_sizes.index(32) if 32 in batch_sizes else len(batch_sizes) - 1
    # batch_32_norm = batch_32_idx / (len(batch_sizes) - 1) if len(batch_sizes) > 1 else 1.0
    
    # # State for top dataset action: Prefill=10, Decode=9, Batch=32, with medium temp headroom
    # state_top_action = np.array([1.0, 9.0/10.0, batch_32_norm, 0.3], dtype=np.float32)  # [prefill=10, decode=9, batch=32, temp_headroom=0.3]
    # print(f"\n>>> MODEL PREDICTION (state matching top dataset action [Prefill=10, Decode=9, Batch=32, TempHeadroom=30%]):")
    # print(f"     State: [1.0, 0.9, {batch_32_norm:.2f}, 0.3] ≈ Prefill bin 10, Decode bin 9, Batch 32, TempHeadroom 30%")
    # optimal_top = inference.print_recommendation(state=state_top_action, slo_target=SLO_TARGET)
    
    # # Compare model prediction vs dataset optimal for this state
    # print(f"\n>>> COMPARISON for state [1.0, 0.9, {batch_32_norm:.2f}, 0.3]:")
    # print(f"     Dataset optimal: Prefill=10, Decode=9, Batch=32")
    # print(f"     Model predicts:  Prefill={optimal_top['prefill_freq_bin']}, Decode={optimal_top['decode_freq_bin']}, Batch={optimal_top['batch_size']}")
    # match = (optimal_top['prefill_freq_bin'] == 10 and optimal_top['decode_freq_bin'] == 9 and optimal_top['batch_size'] == 32)
    # print(f"     Match: {'✓ YES' if match else '✗ NO - Model may need more training'}")
    
    # # Also try maximum state [1.0, 1.0, 1.0, 0.3]
    # print(f"\n>>> MODEL PREDICTION (maximum state [1.0, 1.0, 1.0, 0.3]):")
    # print(f"     This state represents: Prefill bin 10, Decode bin 10, Batch 32, TempHeadroom 30%")
    # state_max = np.array([1.0, 1.0, 1.0, 0.3], dtype=np.float32)
    # optimal_max = inference.print_recommendation(state=state_max, slo_target=SLO_TARGET)
    
    # # Use the one that matches dataset best (default is top_action state result)
    # optimal = optimal_top
    
    # print(f"\n>>> OUTPUT VARIABLES:")
    # print(f"    prefill_freq_bin: {optimal['prefill_freq_bin']}")
    # print(f"    prefill_freq_hz:  {optimal['prefill_freq_hz']}")
    # print(f"    prefill_freq_mhz: {optimal['prefill_freq_mhz']:.1f}")
    # print(f"    decode_freq_bin:  {optimal['decode_freq_bin']}")
    # print(f"    decode_freq_hz:   {optimal['decode_freq_hz']}")
    # print(f"    decode_freq_mhz:  {optimal['decode_freq_mhz']:.1f}")
    # print(f"    batch_size:       {optimal['batch_size']}")
    
    # print(f"\n>>> HARDWARE CONSTRAINTS (Jetson):")
    # print(f"    SLO Target:       <= {SLO_TARGET:.3f}s (chatbot response time)")
    # print(f"    Max Temperature:  <= {MAX_TEMP}°C")
    # print(f"    Max Power:        <= {MAX_POWER}W")
    
    # print(f"\n>>> OPTIMIZATION STRATEGY:")
    # print(f"    Unified Reward: R = H_eff * T_n + (1 - H_eff) * EDP_n - λ_L * f(x_L) - λ_T * f(x_T) - λ_P * f(x_P)")
    # print(f"    H_eff = w_T * H_T + w_P * H_P (effective headroom combining temperature and power)")
    # print(f"    - H_eff > 0.5: More weight on throughput (prioritize speed)")
    # print(f"    - H_eff <= 0.5: More weight on EDP (prioritize energy efficiency)")
    # print(f"    Note: Uses unified reward function with adaptive H_eff weighting (not separate power headroom threshold)")
    
    # # Get top-k actions using the top action state
    # print("\n>>> Top 5 recommended configurations (for top dataset action state):")
    # top_k = inference.get_top_k_actions(state=state_top_action, top_k=5)
    # for i, action in enumerate(top_k):
    #     print(f"  {i+1}. Prefill={action['prefill_freq_mhz']:.0f}MHz, "
    #           f"Decode={action['decode_freq_mhz']:.0f}MHz, "
    #           f"Batch={action['batch_size']}, "
    #           f"Prob={action['action_probability']*100:.2f}%")
    
    # # Show how to use the standalone function
    # print("\n>>> Using standalone function to load and get recommendations:")
    # print("    (This can be used separately without training)")
    # print("    Example: get_grpo_recommendation('grpo_dual_freq_model_option4.pth')")
