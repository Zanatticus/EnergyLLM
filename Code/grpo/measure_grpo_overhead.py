"""
Measure computational overhead of GRPO agent.

This script runs experiments to quantify:
1. Time overhead (action selection, group sampling, policy updates)
2. Sample efficiency (evaluations needed to find good solutions)
3. Overall wall-clock time
4. Power consumption and energy efficiency
"""

import time
import csv
import os
from datetime import datetime
import numpy as np
import torch
import sys
import pandas as pd

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Import the RL agents and environment
from rl_agent0_init_grpo_weight import (
    GRPOAgent, RealJetsonEnv, calculate_reward,
    pretrain_grpo_from_offline_data, OfflineDataset  # For pre-training
)
from rl_agent0 import Llama318BRunner, measure_power_tegrastats
import multiprocessing


class TimedEnvironment:
    """Wrapper environment that tracks timing for each operation."""
    
    def __init__(self, slo_target=15.0, max_temp=100.0, max_power=60.0, freq_bins=11, max_batch=32):
        self.runner = Llama318BRunner()
        self.freq_bins = freq_bins
        self.max_batch = max_batch
        self.state_dim = 4 # prefill_freq, decode_freq, batch_size, temperature_headroom
        
        # Constraint parameters for reward calculation
        self.slo_target = slo_target
        self.max_temp = max_temp
        self.max_power = max_power
        
        # Reward function parameters (defaults)
        self.throughput_min = None
        self.throughput_max = None
        self.edp_efficiency_min = None
        self.edp_efficiency_max = None
        self.energy_min = None
        self.energy_max = None
        self.temp_min = 40.0#30.0
        self.power_min = 10.0
        self.slo_max_over = 1.5 * slo_target if slo_target > 0 else 15.0
        self.temp_max_over = 1.2 * max_temp
        self.power_max_over = 1.3 * max_power
        self.w_temp = 0.5
        self.w_power = 0.5
        self.k_penalty = 5.0
        self.lambda_latency = 1.0
        self.lambda_temp = 1.0
        self.lambda_power = 1.0
        
        # Action space: (prefill_freq, decode_freq, batch)
        self.actions = []
        for prefill_freq in range(0, self.freq_bins):
            for decode_freq in range(0, self.freq_bins):
                for batch in range(1, self.max_batch + 1):
                    self.actions.append((prefill_freq, decode_freq, batch))
        
        self.action_dim = len(self.actions)
        
        # Timing tracking
        self.inference_times = []
        self.batching_times = []  # Tokenization + prompt selection
        self.compute_times = []  # GPU compute time
        self.power_measurements = []
        self.reset()
    
    def reset(self):
        self.prefill_freq = np.random.randint(0, self.freq_bins)
        self.decode_freq = np.random.randint(0, self.freq_bins)
        self.batch_size = np.random.randint(1, self.max_batch + 1)
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            self.prefill_freq / self.freq_bins,
            self.decode_freq / self.freq_bins,
            self.batch_size / self.max_batch
        ], dtype=np.float32)
    
    def step(self, action_idx):
        """Step with timing tracking."""
        print("action_idx: ", action_idx)
        self.prefill_freq, self.decode_freq, self.batch_size = self.actions[action_idx]
        print("prefill_freq: ", self.prefill_freq)
        print("decode_freq: ", self.decode_freq)
        print("batch_size: ", self.batch_size)
        
        # Time the inference
        start_inference = time.time()
        results = self.runner.step_env(
            prefill_freq_bin=self.prefill_freq,
            decode_freq_bin=self.decode_freq,
            batch_size=self.batch_size,
            logging=False,
            max_new_tokens=100,
            set_freq=False
        )
        inference_time = time.time() - start_inference
        self.inference_times.append(inference_time)
        
        # Track batching and compute times separately
        self.batching_times.append(results['tokenize_time'])
        self.compute_times.append(results['prefill_compute_time'])
        
        # Extract metrics from results
        latency_prefill = results['prefill_time']
        latency_decode = results['decode_time']
        tokens_prefill = results['num_prompt_tokens']
        tokens_decode = results['num_decode_tokens']
        power_prefill = results['prefill_TOTAL_POWER']
        power_decode = results['decode_TOTAL_POWER']
        
        # Calculate total metrics from individual values
        total_tokens = tokens_prefill + tokens_decode
        total_time = latency_prefill + latency_decode
        end_to_end_latency = total_time
        avg_power = (power_prefill + power_decode) / 2.0
        total_energy = latency_prefill * power_prefill + latency_decode * power_decode
        
        # Track power measurements
        self.power_measurements.append({
            'avg_power_w': avg_power,
            'total_energy_j': total_energy,
            'prefill_power_w': power_prefill,
            'decode_power_w': power_decode
        })
        
        # Calculate throughput: tokens per second
        throughput_raw = total_tokens / (total_time + 1e-6) if total_time > 0 else 0.0
        
        # Calculate EDP efficiency: 1 / (energy * latency)
        edp = total_energy * end_to_end_latency
        edp_efficiency_raw = 1.0 / (edp + 1e-6) if edp > 0 else 0.0
        
        # Calculate power headroom (0 to 1, where 1 = lots of headroom)
        power_headroom = max(0, (self.max_power - avg_power) / self.max_power)
        
        # Get average temperature from results
        # Try to get from prefill and decode TJ_TEMP (junction temperature, hottest spot)
        temp_prefill = results.get('prefill_TJ_TEMP', None)
        temp_decode = results.get('decode_TJ_TEMP', None)

        if temp_prefill is not None and temp_decode is not None:
            # Average of prefill and decode temperatures
            avg_temp = (temp_prefill + temp_decode) / 2.0

        # Extract individual temperatures for CSV storage only
        # temp_prefill = results.get('prefill_TJ_TEMP', 0)
        # temp_decode = results.get('decode_TJ_TEMP', 0)

        # Calculate reward only if normalization ranges are available
        # During grid search, we collect raw data first, then normalize after all data is collected
        reward = 0.0
        performance_metric = 0.0
        constraint_penalty = 0.0
        throughput_normalized = 0.0
        edp_efficiency_normalized = 0.0

        
        if self.throughput_min is not None and self.throughput_max is not None and \
           self.edp_efficiency_min is not None and self.edp_efficiency_max is not None and \
           self.energy_min is not None and self.energy_max is not None:
            # Normalization ranges available - calculate reward
            reward, performance_metric, constraint_penalty, throughput_normalized, edp_efficiency_normalized = calculate_reward(
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
                energy_min=self.energy_min,
                energy_max=self.energy_max,
                temp_min=self.temp_min,
                power_min=self.power_min,
                slo_max_over=self.slo_max_over,
                temp_max_over=self.temp_max_over,
                power_max_over=self.power_max_over,
                w_temp=self.w_temp,
                w_power=self.w_power,
                k_penalty=self.k_penalty,
                lambda_latency=self.lambda_latency,
                lambda_temp=self.lambda_temp,
                lambda_power=self.lambda_power
            )
        # else:
        #     raise ValueError("Normalization ranges are not available")
        
        next_state = self._get_state()
        done = False
        info = {
            'energy': total_energy,
            'end_to_end_latency': end_to_end_latency,  # Per-request latency
            'inference_time': inference_time,
            'batching_time': results.get('tokenize_time', 0.0),
            'compute_time': results.get('prefill_compute_time', 0.0),
            'avg_power': avg_power,
            'temperature': avg_temp,
            'power_headroom': power_headroom,
            'slo_violation': end_to_end_latency > self.slo_target,
            'temp_violation': avg_temp > self.max_temp,
            'power_violation': avg_power > self.max_power,
            'throughput_raw': throughput_raw,
            'edp_efficiency_raw': edp_efficiency_raw,
            'performance_metric': performance_metric,
            'constraint_penalty': constraint_penalty,
            'prefill_power_w': power_prefill,
            'decode_power_w': power_decode,
            # Individual prefill/decode metrics
            'prefill_time': latency_prefill,
            'decode_time': latency_decode,
            'num_prompt_tokens': tokens_prefill,
            'num_decode_tokens': tokens_decode,
            'prefill_TOTAL_POWER': power_prefill,
            'decode_TOTAL_POWER': power_decode
        }
        
        return next_state, reward, done, info


def measure_grid_search_baseline(num_samples=100, seed=42, warmup_rounds=40, 
                                 slo_target=15.0, max_temp=100.0, max_power=60.0,
                                 freq_bins=11, max_batch=32):
    """Measure baseline grid search performance."""
    print("\n" + "="*60)
    print("MEASURING GRID SEARCH BASELINE")
    print("="*60)
    print(f"Samples: {num_samples}")
    print(f"Constraints: SLO={slo_target}s, MaxTemp={max_temp}°C, MaxPower={max_power}W")
    print(f"Action space: {freq_bins} freq bins x {freq_bins} freq bins x {max_batch} batch sizes = {freq_bins * freq_bins * max_batch} actions")
    
    np.random.seed(seed)
    env = TimedEnvironment(slo_target=slo_target, max_temp=max_temp, max_power=max_power, 
                          freq_bins=freq_bins, max_batch=max_batch)
    
    #Warmup rounds
    if warmup_rounds > 0:
        print(f"\nRunning {warmup_rounds} warmup rounds...")
        
        # Configure warmup action: (prefill_freq, decode_freq, batch)
        # Set these to specific values, or None to use random
        warmup_prefill_freq = 10  # Set to desired prefill frequency bin (0 to freq_bins-1), or None for random
        warmup_decode_freq = 10   # Set to desired decode frequency bin (0 to freq_bins-1), or None for random
        warmup_batch = 32       # Set to desired batch size (1 to max_batch), or None for random
        
        for i in range(warmup_rounds):
            _ = env.reset()
            
            # Determine action for this warmup round
            if warmup_prefill_freq is not None and warmup_decode_freq is not None and warmup_batch is not None:
                # Use specified action - convert (prefill_freq, decode_freq, batch) to action index
                warmup_action = (warmup_prefill_freq, warmup_decode_freq, warmup_batch)
                if warmup_action in env.actions:
                    action_idx = env.actions.index(warmup_action)
                    print(f"  Warmup {i+1}/{warmup_rounds}: prefill_freq={warmup_prefill_freq}, decode_freq={warmup_decode_freq}, batch={warmup_batch} (action_idx={action_idx})")
            else:
                # Use random action
                action_idx = np.random.randint(0, 100)
                action_tuple = env.actions[action_idx]
                print(f"  Warmup {i+1}/{warmup_rounds}: random action {action_idx} (prefill={action_tuple[0]}, decode={action_tuple[1]}, batch={action_tuple[2]})")
            
            _ = env.step(action_idx)
        print("Warmup complete. Starting measurement...\n")
        
        # Reset tracking after warmup
        env.inference_times = []
        env.batching_times = []
        env.compute_times = []
        env.power_measurements = []
    
    # Random grid search
    # start_time = time.time()
    # for i in range(num_samples):
    #     _ = env.reset()
    #     action = np.random.randint(0, env.action_dim)
    #     _, reward, _, _ = env.step(action)
    #     if (i + 1) % 20 == 0:
    #         print(f"  Sample {i + 1}/{num_samples} completed...")
    
    # total_time = time.time() - start_time

    # Create CSV file for grid search results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fan_off_grid_search_results_{timestamp}.csv"
    
    # Define CSV fieldnames based on info structure from TimedEnvironment.step
    # Include normalized values and reward calculated after all data is collected
    info_fieldnames = [
        'energy', 'end_to_end_latency', 'inference_time', 'batching_time', 'compute_time', 'avg_power', 'temperature',
        'power_headroom', 'slo_violation', 'temp_violation', 'power_violation',
        'throughput_raw', 'edp_efficiency_raw', 'performance_metric', 'constraint_penalty',
        'prefill_power_w', 'decode_power_w',
        'prefill_time', 'decode_time', 'num_prompt_tokens', 'num_decode_tokens',
        'prefill_TOTAL_POWER', 'decode_TOTAL_POWER',
        'inverse_energy_normalized', 'headroom_threshold_met', 'energy_min', 'energy_max'
    ]
    fieldnames = ['prefill_freq_bin', 'decode_freq_bin', 'effective_batch_size'] + info_fieldnames
    
    # Open CSV file for writing
    csv_file = open(csv_filename, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    print(f"Writing grid search results to: {csv_filename}")
    total_combinations = env.freq_bins * env.freq_bins * env.max_batch
    print(f"Total combinations to evaluate: {total_combinations}")
    
    # Reset environment and clear tracking before starting grid search
    _ = env.reset()
    env.inference_times = []
    env.batching_times = []
    env.compute_times = []
    env.power_measurements = []
    
    start_time = time.time()
    count = 0
    
    # First pass: Collect all raw data and write to CSV row by row
    all_data = []
    throughput_raw_values = []
    edp_efficiency_raw_values = []
    energy_raw_values = []
    print("First pass: Collecting raw data...")
    for prefill_freq in range(env.freq_bins-1, -1, -2):
        for decode_freq in range(env.freq_bins-1, -1, -1):
            for batch in [32,26,16,8,5,4,3,2,1]:#range(env.max_batch, 0, -1): 
                action = (prefill_freq, decode_freq, batch)
                # Find the action index
                action_idx = env.actions.index(action)
                _, _, _, info = env.step(action_idx)
                row = {
                    'prefill_freq_bin': prefill_freq,
                    'decode_freq_bin': decode_freq,
                    'effective_batch_size': batch,
                    **info
                }
                all_data.append(row)
                writer.writerow(row)
                csv_file.flush()
                
                # Collect raw values for min/max calculation
                if 'throughput_raw' in info:
                    throughput_raw_values.append(info['throughput_raw'])
                if 'edp_efficiency_raw' in info:
                    edp_efficiency_raw_values.append(info['edp_efficiency_raw'])
                if 'energy' in info:
                    energy_raw_values.append(info['energy'])

                count += 1
                if count % 100 == 0:
                    print(f"  Completed {count}/{total_combinations} combinations...")
    
    # Close CSV file after first pass
    csv_file.flush()
    csv_file.close()
    
    # Calculate normalization ranges from all collected data
    if throughput_raw_values:
        throughput_min = min(throughput_raw_values)
        throughput_max = max(throughput_raw_values)
        print(f"\nThroughput range: [{throughput_min:.2f}, {throughput_max:.2f}] tokens/sec")
    else:
        throughput_min = None
        throughput_max = None
    
    if edp_efficiency_raw_values:
        edp_efficiency_min = min(edp_efficiency_raw_values)
        edp_efficiency_max = max(edp_efficiency_raw_values)
        print(f"EDP efficiency range: [{edp_efficiency_min:.6f}, {edp_efficiency_max:.6f}]")
    else:
        edp_efficiency_min = None
        edp_efficiency_max = None
    
    if energy_raw_values:
        energy_min = min(energy_raw_values)
        energy_max = max(energy_raw_values)
        print(f"\nEnergy range: [{energy_min:.6f}, {energy_max:.6f}] J")
    else:
        energy_min = None
        energy_max = None
    # Second pass: Calculate normalized values and rewards, add as new columns
    print("\nSecond pass: Calculating normalized values and rewards...")
    
    # Update fieldnames to include normalized columns
    fieldnames_with_normalized = fieldnames + ['throughput_normalized', 'edp_efficiency_normalized', 'performance_metric', 'reward']
    
    # Reopen CSV file in write mode to rewrite with normalized columns
    csv_file = open(csv_filename, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames_with_normalized)
    writer.writeheader()
    
    for row_data in all_data:
        # Calculate normalized values and reward for this row
        if 'throughput_raw' in row_data and 'edp_efficiency_raw' in row_data and throughput_min is not None and edp_efficiency_min is not None and energy_min is not None and energy_max is not None:
            reward, performance_metric, constraint_penalty, throughput_normalized, edp_efficiency_normalized, inverse_energy_normalized, headroom_threshold_met = calculate_reward(
                throughput_raw=row_data['throughput_raw'],
                edp_efficiency_raw=row_data['edp_efficiency_raw'],
                energy_total=row_data['energy'],
                energy_min=energy_min,
                energy_max=energy_max,
                power_headroom=row_data.get('power_headroom', 0.5),
                end_to_end_latency=row_data.get('end_to_end_latency', 0.0),
                avg_power=row_data.get('avg_power', 0.0),
                avg_temp=row_data.get('temperature', 50.0),
                slo_target=slo_target,
                max_temp=max_temp,
                max_power=max_power,
                throughput_min=throughput_min,
                throughput_max=throughput_max,
                edp_efficiency_min=edp_efficiency_min,
                edp_efficiency_max=edp_efficiency_max
            )
            
            # Add normalized values to row (update constraint_penalty if it exists)
            row_data['throughput_normalized'] = throughput_normalized
            row_data['edp_efficiency_normalized'] = edp_efficiency_normalized
            row_data['performance_metric'] = performance_metric
            row_data['constraint_penalty'] = constraint_penalty
            row_data['reward'] = reward
            row_data['inverse_energy_normalized'] = inverse_energy_normalized
            row_data['headroom_threshold_met'] = headroom_threshold_met
            row_data['energy_min'] = energy_min
            row_data['energy_max'] = energy_max
        else:
            # If no raw data available, set normalized values to None
            row_data['throughput_normalized'] = None
            row_data['edp_efficiency_normalized'] = None
            row_data['performance_metric'] = None
            row_data['reward'] = None
            row_data['inverse_energy_normalized'] = None
            row_data['headroom_threshold_met'] = None
            row_data['energy_min'] = None
            row_data['energy_max'] = None
        # Write complete row with normalized columns
        writer.writerow(row_data)
    
    csv_file.flush()
    csv_file.close()
    total_time = time.time() - start_time
    print(f"\nGrid search complete. Results saved to: {csv_filename}")
    print(f"Total time: {total_time:.2f} sec")
    
    avg_inference_time = np.mean(env.inference_times) if env.inference_times else 0.0
    avg_batching_time = np.mean(env.batching_times) if env.batching_times else 0.0
    avg_compute_time = np.mean(env.compute_times) if env.compute_times else 0.0
    avg_power = np.mean([p['avg_power_w'] for p in env.power_measurements]) if env.power_measurements else 0.0
    total_energy = np.sum([p['total_energy_j'] for p in env.power_measurements]) if env.power_measurements else 0.0
    
    results = {
        'method': 'grid_search',
        'pretrain_time_sec': 0.0,
        'online_time_sec': total_time,
        'total_time_sec': total_time,
        'num_evaluations': count,
        'avg_inference_time_sec': avg_inference_time,
        'avg_batching_time_sec': avg_batching_time,
        'avg_compute_time_sec': avg_compute_time,
        'avg_reward': 0.0,  # Grid search doesn't track rewards
        'max_reward': 0.0,
        'action_selection_time_ms': 0.0,  # Random selection is instant
        'policy_update_time_sec': 0.0,  # No training
        'num_updates': 0,
        'overhead_percentage': 0.0,
        'avg_power_w': avg_power,
        'total_energy_j': total_energy,
        'energy_per_eval_j': total_energy / count if count > 0 else 0.0,
        'slo_violation_rate': 0.0,
        'temp_violation_rate': 0.0,
        'power_violation_rate': 0.0
    }
    
    print(f"\nGrid Search Results:")
    print(f"  Total time: {total_time} sec")
    print(f"  Evaluations: {count}")
    print(f"  Avg inference time: {avg_inference_time} sec")
    print(f"  Avg power: {avg_power} W")
    print(f"  Total energy: {total_energy} J")
    
    return results


def measure_grpo_overhead(csv_path=None, num_episodes=None, max_steps=None, num_evaluations=None,
                          baseline_power=None, warmup_rounds=5, seed=42,
                          convergence_threshold=0.0001, convergence_window=30, action_prob_threshold=0.8,
                          action_prob_window=30, group_size=8, slo_target=15.0, max_temp=100.0, min_temp=40.0, max_power=60.0,
                          pretrain_epochs=20, reward_target_ratio=0.96, reward_stable_window=15):
    """Measure GRPO agent overhead with pre-training + online training.
    
    Args:
        csv_path: Path to CSV file for pre-training, or list/tuple of CSV paths to combine (if None, skips pre-training)
        num_episodes: Number of episodes (if None and num_evaluations is set, will be calculated)
        max_steps: Maximum steps per episode (if None and num_evaluations is set, will be calculated)
        num_evaluations: Total number of online evaluations (if set, overrides num_episodes * max_steps)
        pretrain_epochs: Number of epochs for offline pre-training
    """
    # Calculate num_episodes and max_steps from num_evaluations if provided
    if num_evaluations is not None:
        if num_episodes is None and max_steps is None:
            # Default: use reasonable split (e.g., 10 steps per episode, but ensure we don't exceed num_evaluations)
            max_steps = min(10, num_evaluations)  # Don't exceed num_evaluations
            num_episodes = (num_evaluations + max_steps - 1) // max_steps  # Ceiling division
            # Adjust to ensure we don't exceed num_evaluations
            if num_episodes * max_steps > num_evaluations:
                # Reduce max_steps to match exactly
                max_steps = num_evaluations // num_episodes
                if max_steps == 0:
                    max_steps = 1
                    num_episodes = num_evaluations
        elif num_episodes is None:
            # max_steps provided, calculate num_episodes
            num_episodes = (num_evaluations + max_steps - 1) // max_steps
            # Adjust max_steps to ensure we don't exceed num_evaluations
            if num_episodes * max_steps > num_evaluations:
                max_steps = num_evaluations // num_episodes
                if max_steps == 0:
                    max_steps = 1
                    num_episodes = num_evaluations
        elif max_steps is None:
            # num_episodes provided, calculate max_steps
            max_steps = (num_evaluations + num_episodes - 1) // num_episodes
            # Adjust num_episodes to ensure we don't exceed num_evaluations
            if num_episodes * max_steps > num_evaluations:
                num_episodes = (num_evaluations + max_steps - 1) // max_steps
        else:
            # Both provided, use them but warn if they don't match
            if num_episodes * max_steps != num_evaluations:
                print(f"Warning: num_episodes * max_steps ({num_episodes * max_steps}) != num_evaluations ({num_evaluations})")
                print(f"  Adjusting to match num_evaluations={num_evaluations}")
                # Recalculate to match num_evaluations exactly
                if num_episodes > num_evaluations:
                    num_episodes = num_evaluations
                    max_steps = 1
                else:
                    max_steps = num_evaluations // num_episodes
                    if num_episodes * max_steps < num_evaluations:
                        max_steps += 1
    else:
        # num_evaluations not provided, use defaults if num_episodes/max_steps are None
        if num_episodes is None:
            num_episodes = 100
        if max_steps is None:
            max_steps = 10
    
    # Calculate actual total evaluations (may be less than num_episodes * max_steps due to early exits)
    # But we'll track actual evaluations during the loop
    planned_evaluations = num_episodes * max_steps
    
    print("\n" + "="*60)
    print("MEASURING GRPO OVERHEAD (PRE-TRAINING + ONLINE TRAINING)")
    print("="*60)
    if csv_path:
        if isinstance(csv_path, (list, tuple)):
            print(f"Phase 1: Pre-training from {len(csv_path)} CSV files: {csv_path} ({pretrain_epochs} epochs)")
        else:
            print(f"Phase 1: Pre-training from CSV: {csv_path} ({pretrain_epochs} epochs)")
        print(f"Phase 2: Online training with RealJetsonEnv")
    else:
        print(f"Skipping pre-training, only online training")
    print(f"Online evaluations: {planned_evaluations} planned ({num_episodes} episodes × {max_steps} steps)")
    if num_evaluations is not None:
        print(f"  (Requested: {num_evaluations} evaluations)")
    print(f"Group size: {group_size} (samples per state)")
    print(f"Convergence: loss change < {convergence_threshold} for {convergence_window} consecutive updates")
    print(f"Confidence exit: action prob > {action_prob_threshold} for {action_prob_window} consecutive steps")
    print(f"Reward exit: avg reward >= {reward_target_ratio*100:.0f}% of dataset max for {reward_stable_window} consecutive episodes")
    print(f"Constraints: SLO={slo_target}s, MaxTemp={max_temp}°C, MaxPower={max_power}W")
    
    if baseline_power is not None:
        print(f"Baseline power (Grid Search): {baseline_power:.2f} W")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Phase 1: Pre-training from CSV (if provided)
    pretrain_time = 0.0
    pretrained_policy = None
    dataset = None  # Initialize dataset variable
    if csv_path:
        print(f"\n{'='*60}")
        print(f"PHASE 1: PRE-TRAINING GRPO FROM OFFLINE DATA")
        print(f"{'='*60}")
        pretrain_start = time.time()
        
        # Use the built-in GRPO pre-training function
        pretrained_policy, pretrain_losses, dataset = pretrain_grpo_from_offline_data(
            csv_path, freq_bins=11, max_batch=32, epochs=pretrain_epochs, 
            batch_size=32, lr=1e-4, alpha=0.8,
            slo_target=slo_target, max_temp=max_temp, max_power=max_power
        )
        
        pretrain_time = time.time() - pretrain_start
        print(f"Pre-training completed in {pretrain_time:.2f} sec")
        print(f"\n{'='*60}")
        print(f"PHASE 2: ONLINE DEVELOPMENT OF GRPO AGENT")
        print(f"{'='*60}")
    
    # Use RealJetsonEnv for actual hardware measurements
    env = RealJetsonEnv(freq_bins=11, max_batch=32, slo_target=slo_target,  
                        max_temp=max_temp, min_temp=min_temp, max_power=max_power)
    
    # Load normalization ranges from dataset if available (for proper reward calculation)
    if csv_path and dataset is not None:
        env.throughput_min = dataset.throughput_min
        env.throughput_max = dataset.throughput_max
        env.edp_efficiency_min = dataset.edp_efficiency_min
        env.edp_efficiency_max = dataset.edp_efficiency_max
        env.energy_min = dataset.energy_min
        env.energy_max = dataset.energy_max
        env._track_normalization = True
        print(f"\nLoaded normalization ranges from dataset:")
        print(f"  Throughput: [{env.throughput_min:.2f}, {env.throughput_max:.2f}] tokens/sec")
        print(f"  EDP efficiency: [{env.edp_efficiency_min:.6f}, {env.edp_efficiency_max:.6f}]")
        
        # Show maximum reward from dataset for reference
        if hasattr(dataset, 'processed_data') and dataset.processed_data:
            max_reward = max(d['reward'] for d in dataset.processed_data)
            avg_reward = np.mean([d['reward'] for d in dataset.processed_data])
            print(f"\nDataset reward statistics:")
            print(f"  Maximum reward in dataset: {max_reward:.4f}")
            print(f"  Average reward in dataset: {avg_reward:.4f}")
            print(f"  (Your online rewards should be compared to these values)")
    else:
        print(f"\nWarning: No normalization ranges available. Using fallback normalization.")
        print(f"  This may cause incorrect reward values. Consider providing a CSV file for pre-training.")
    
    # Create GRPO agent with pre-trained policy if available
    if pretrained_policy is not None:
        agent = GRPOAgent(env.state_dim, env.action_dim, policy=pretrained_policy,
                         lr=1e-4, group_size=group_size)  # Lower LR for fine-tuning
        print("Using pre-trained policy for GRPO agent")
    else:
        agent = GRPOAgent(env.state_dim, env.action_dim, lr=1e-4, group_size=group_size)
    
    # if warmup_rounds > 0:
    #     print(f"\nRunning {warmup_rounds} warmup rounds...")
    #     for i in range(warmup_rounds):
    #         _ = env.reset()
    #         action = np.random.randint(0, 3)
    #         _ = env.step(action)
    #     print("Warmup complete. Starting measurement...\n")


    # Warmup rounds
    # if warmup_rounds > 0:
    #     print(f"\nRunning {warmup_rounds} warmup rounds...")
    #     for i in range(warmup_rounds):
    #         state = env.reset()
    #         # GRPO returns (action, log_prob)
    #         result = agent.select_action(state)
    #         if len(result) == 3:
    #             action, _, _ = result
    #         else:
    #             action, _ = result
    #         _ = env.step(action)
    #         if (i + 1) % 10 == 0:
    #             print(f"  Warmup {i + 1}/{warmup_rounds} completed...")
    #     print("Warmup complete. Starting measurement...\n")
        
    #     # Reset tracking after warmup
    #     env.inference_times = []
    #     env.batching_times = []
    #     env.compute_times = []
    #     env.power_measurements = []
    
    action_selection_times = []
    group_sampling_times = []  # Track group sampling time for GRPO
    policy_update_times = []
    action_selection_energies = []  # Track energy during action selection (Joules)
    policy_update_energies = []  # Track energy during policy updates (Joules)
    inference_energies = []  # Track energy during inference (Joules)
    total_evaluations = 0
    rewards = []
    losses = []
    loss_changes = []  # Track loss changes for CSV
    action_probs = []  # Track action probabilities
    converged = False
    high_confidence_exit = False
    reward_target_reached = False
    slo_violations = 0
    temp_violations = 0
    power_violations = 0
    loss_convergence_confidence = 0.0  # Initialize loss convergence confidence
    
    # Get dataset maximum reward for reward-based exit condition
    dataset_max_reward = None
    if csv_path and dataset is not None and hasattr(dataset, 'processed_data') and dataset.processed_data:
        dataset_max_reward = max(d['reward'] for d in dataset.processed_data)
        reward_target = dataset_max_reward * reward_target_ratio
        print(f"\nReward-based exit condition:")
        print(f"  Dataset maximum reward: {dataset_max_reward:.4f}")
        print(f"  Target reward ({reward_target_ratio*100:.0f}% of max): {reward_target:.4f}")
        print(f"  Will exit when average reward >= {reward_target:.4f} for {reward_stable_window} consecutive episodes")
    
    # Track last action for convergence reporting
    last_action_idx = None
    last_prefill_freq = None
    last_decode_freq = None
    last_batch = None

    # Create CSV file for GRPO training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"grpo_training_results_{timestamp}.csv"
    # CSV is saved in the current working directory (where the script is run from)
    # To save in a specific directory, use: csv_filename = f"/path/to/dir/grpo_training_results_{timestamp}.csv"
    
    # Create directory for saving models
    model_dir = f"grpo_models_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved to: {model_dir}/")
    
    # Define CSV fieldnames: episode/step info + action info + GRPO metrics + info fields
    # Note: RealJetsonEnv returns these fields in info: energy, latency, end_to_end_latency,
    # temperature, power, power_headroom, slo_violation, temp_violation, power_violation
    info_fieldnames = [
        'energy', 'per_token_latency', 'end_to_end_latency', 'prefill_latency', 'decode_latency','temperature', 'power',
        'power_headroom', 'slo_violation', 'temp_violation', 'power_violation',
        'throughput_raw', 'edp_efficiency_raw', 'performance_metric',
        'throughput_normalized', 'edp_efficiency_normalized', 
        'inverse_energy_normalized', 'headroom_threshold_met', 'constraint_penalty',
        'throughput_min', 'throughput_max', 'edp_efficiency_min', 'edp_efficiency_max', 'energy_min', 'energy_max'
    ]
    fieldnames = [
        'episode', 'step', 'action_idx', 'prefill_freq', 'decode_freq', 'batch',
        'log_prob', 'action_prob', 'reward', 'group_sampling_time_ms',
        'action_prob_confidence', 'loss_convergence_confidence', 'overall_confidence',
        'loss', 'loss_change'
    ] + info_fieldnames
    
    # Open CSV file for writing
    csv_file = open(csv_filename, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    print(f"Writing GRPO training results to: {csv_filename}")
 
    
    total_start = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        
        for step in range(max_steps):
            # For GRPO, we can sample a group of actions or use select_action
            # Time group sampling (if using sample_group) or action selection
            # start_group = time.time()

            # # Measure power during action selection
            # power_queue = multiprocessing.Queue()
            # power_result_queue = multiprocessing.Queue()
            # power_process = multiprocessing.Process(
            #     target=measure_power_tegrastats, args=(power_queue, power_result_queue)
            # )
            # power_process.start()
            # power_queue.put(True)  # Start signal
            
            # # GRPO: sample a group of actions for the same state
            # actions_group, log_probs_group = agent.sample_group(state)
            
            # # Stop power measurement
            # power_queue.put(False)
            # power_process.join()
            # power_before_action = 0.0
            # if not power_result_queue.empty():
            #     power_result = power_result_queue.get()
            #     power_before_action = power_result.get('TOTAL_POWER', 0.0)
            
            # group_sampling_time = (time.time() - start_group) * 1000  # Convert to ms
            # group_sampling_times.append(group_sampling_time)
            # # Calculate energy: power × time (convert ms to seconds)
            # action_time_sec = group_sampling_time / 1000.0
            # action_energy = power_before_action * action_time_sec  # Watts × seconds = Joules
            # action_selection_energies.append(action_energy)
            # Time group sampling (if using sample_group) or action selection

            # Measure power during action selection
            # Start power measurement process early to allow initialization
            power_queue = multiprocessing.Queue()
            power_result_queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=measure_power_tegrastats, args=(power_queue, power_result_queue)
            )
            power_process.start()
            
            # Give the process time to initialize and start reading tegrastats output
            # tegrastats needs to skip initial lines until it finds VDD_ lines
            time.sleep(0.2)
            power_queue.put(True)  # Start signal
            
            start_group = time.time()
            # GRPO: sample a group of actions for the same state
            actions_group, log_probs_group = agent.sample_group(state)
            group_sampling_time = (time.time() - start_group) * 1000  # Convert to ms
            
            # tegrastats outputs at 1-second intervals, so we need to wait at least 1 second
            # to get at least one sample. Wait 1.2 seconds to ensure we get a complete sample.
            time.sleep(1.2)
            
            # Stop power measurement
            power_queue.put(False)
            # Wait for process to finish collecting and processing samples
            time.sleep(0.3)
            power_process.join(timeout=5.0)  # Add timeout to prevent hanging
            if power_process.is_alive():
                power_process.terminate()
            power_process.join()
            
            power_during_action = 0.0
            if not power_result_queue.empty():
                power_result = power_result_queue.get()
                power_during_action = power_result.get('TOTAL_POWER', 0.0)
                if power_during_action == 0.0:
                    print(f"Warning: Action selection power measurement returned 0.0W")
            else:
                print(f"Warning: Action selection power measurement queue is empty")

            group_sampling_times.append(group_sampling_time)
            # Calculate energy: power × time (convert ms to seconds)
            action_time_sec = group_sampling_time / 1000.0
            action_energy = power_during_action * action_time_sec  # Watts × seconds = Joules
            action_selection_energies.append(action_energy)
            
            # Select the first action from the group for execution
            action = actions_group[0]
            log_prob = log_probs_group[0]
            value = None  # GRPO doesn't have value function
            
            # Track action selection time (per action in group)
            action_time = group_sampling_time / len(actions_group)  # Average per action
            action_selection_times.append(action_time)
            
            # Track action probability (convert from log_prob)
            action_prob = torch.exp(log_prob).item()
            action_probs.append(action_prob)
            
            # Calculate action probability confidence (how close to threshold)
            # Confidence measures how stable/confident the policy is
            action_prob_confidence = 0.0
            if len(action_probs) >= action_prob_window:
                recent_probs = action_probs[-action_prob_window:]
                avg_prob = np.mean(recent_probs)
                std_prob = np.std(recent_probs) if len(recent_probs) > 1 else 0.0
                
                # Confidence based on:
                # 1. How close average is to threshold (normalized)
                # 2. How stable the probabilities are (low std = high confidence)
                # 3. Fraction of probs above threshold
                above_threshold_frac = sum(1 for p in recent_probs if p > action_prob_threshold) / len(recent_probs)
                
                # Normalize average probability to [0, 1] relative to threshold
                avg_normalized = min(1.0, avg_prob / action_prob_threshold) if action_prob_threshold > 0 else 0.0
                
                # Stability: lower std relative to mean = higher confidence
                # Use coefficient of variation (std/mean), invert it for confidence
                stability = 1.0 / (1.0 + std_prob / (avg_prob + 1e-8)) if avg_prob > 0 else 0.0
                
                # Combine: weighted average of threshold proximity, stability, and fraction above
                action_prob_confidence = 0.4 * avg_normalized + 0.3 * stability + 0.3 * above_threshold_frac
            
            # Check for high confidence early exit - BUT ONLY IF REWARDS ARE GOOD
            if len(action_probs) >= action_prob_window:
                recent_probs = action_probs[-action_prob_window:]
                if all(prob > action_prob_threshold for prob in recent_probs):
                    # ADD THIS CHECK: Only exit if rewards are also good
                    if dataset_max_reward is not None and len(rewards) >= reward_stable_window:
                        recent_rewards = rewards[-reward_stable_window:]
                        avg_recent_reward = np.mean(recent_rewards)
                        reward_threshold = dataset_max_reward * 0.8  # At least 80% of max
                        
                        if avg_recent_reward >= reward_threshold:
                            high_confidence_exit = True
                            print(f"\nHigh confidence exit at episode {episode + 1}, step {step + 1}!")
                            print(f"  Avg action prob: {np.mean(recent_probs):.4f}")
                            print(f"  Avg reward: {avg_recent_reward:.4f} (threshold: {reward_threshold:.4f})")
                        else:
                            # High confidence but low rewards - don't exit
                            if step == 0:  # Only print once per episode
                                print(f"  High action prob but low rewards ({avg_recent_reward:.4f} < {reward_threshold:.4f}), continuing...")
                    else:
                        # No reward data - use original behavior
                        high_confidence_exit = True
                        print(f"\nHigh confidence exit (no reward check available) at episode {episode + 1}, step {step + 1}!")
                        print(f"  Avg action prob: {np.mean(recent_probs):.4f}, Confidence: {action_prob_confidence:.4f}")
            
            # Environment step (inference)
            # Track inference energy from environment measurements
            inference_start_time = time.time()
            next_state, reward, done, info = env.step(action)
            inference_time_actual = time.time() - inference_start_time
            rewards.append(reward)
            
            # Get inference energy from info (if available) or calculate from power × time
            inference_energy = info.get('energy', 0.0)  # Energy from environment if available
            if inference_energy == 0.0:
                # Fallback: calculate from average power × time
                inference_power = info.get('avg_power', info.get('power', 0.0))
                inference_energy = inference_power * inference_time_actual
            inference_energies.append(inference_energy)

            # Get action tuple from action index
            prefill_freq, decode_freq, batch = env.actions[action]
            
            # Track last action for convergence reporting
            last_action_idx = action
            last_prefill_freq = prefill_freq
            last_decode_freq = decode_freq
            last_batch = batch
            
            # Calculate overall confidence (weighted average of action and loss confidence)
            # Loss confidence is updated after policy update, so use the last known value
            # If loss confidence is 0 (no updates yet), weight action confidence more
            if loss_convergence_confidence > 0:
                # Both available: equal weight
                overall_confidence = (action_prob_confidence + loss_convergence_confidence) / 2.0
            else:
                # Only action confidence available: use it directly (scaled)
                overall_confidence = action_prob_confidence
            
            # Get current loss and loss change for this step
            # Loss is updated at the end of each episode, so use the last known value
            # For steps within an episode, use the loss from the previous episode's update
            current_loss = losses[-1] if losses else None
            # loss_changes has one entry per episode (after policy update)
            # For step 0 of current episode, use loss_change from previous episode
            # For step > 0 of current episode, use the same loss_change
            if len(loss_changes) > 0:
                # Use the most recent loss change (from last episode's update)
                current_loss_change = loss_changes[-1] if loss_changes[-1] is not None else None
            else:
                current_loss_change = None
            
            # Write row to CSV
            row = {
                'episode': episode,
                'step': step,
                'action_idx': action,
                'prefill_freq': prefill_freq,
                'decode_freq': decode_freq,
                'batch': batch,
                'log_prob': log_prob.item() if torch.is_tensor(log_prob) else log_prob,
                'action_prob': action_prob,
                'reward': reward,
                'group_sampling_time_ms': group_sampling_time,
                'action_prob_confidence': action_prob_confidence,
                'loss_convergence_confidence': loss_convergence_confidence,
                'overall_confidence': overall_confidence,
                'loss': current_loss if current_loss is not None else '',
                'loss_change': current_loss_change if current_loss_change is not None else '',
                **info
            }
            writer.writerow(row)
            csv_file.flush()  # Force write to disk immediately to prevent data loss

            # Store all group transitions (for GRPO group-based learning)
            # But for overhead measurement, we only execute the first action
            # Store the executed action's transition
            agent.store_transition(state, action, log_prob, reward)
            
            # Track constraint violations
            if info.get('latency', 0) > env.slo_target:
                slo_violations += 1
            if info.get('temperature', 0) > env.max_temp:
                temp_violations += 1
            if info.get('avg_power', 0) > env.max_power:
                power_violations += 1
            
            # GRPO's store_transition doesn't take value parameter
            if hasattr(agent, 'store_transition'):
                # Check if it's GRPO (no value parameter) or PPO (has value parameter)
                import inspect
                sig = inspect.signature(agent.store_transition)
                if 'value' in sig.parameters:
                    agent.store_transition(state, action, log_prob, reward, value, done)
                else:
                    agent.store_transition(state, action, log_prob, reward)
            state = next_state
            total_evaluations += 1
            
            if high_confidence_exit:
                break
        
        # Time the policy update
        # Calculate loss convergence confidence before update (for current step)
        # This will be used in the next episode's steps
        if len(agent.memory['states']) > 0:
            start_update = time.time()
            # Measure power during policy update
            power_queue = multiprocessing.Queue()
            power_result_queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=measure_power_tegrastats, args=(power_queue, power_result_queue)
            )
            power_process.start()
            power_queue.put(True)  # Start signal
            
            loss = agent.update()
            
            # Stop power measurement
            power_queue.put(False)
            power_process.join(timeout=5.0)  # Add timeout to prevent hanging
            if power_process.is_alive():
                power_process.terminate()
            power_process.join()
            
            power_during_update = 0.0
            if not power_result_queue.empty():
                power_result = power_result_queue.get()
                power_during_update = power_result.get('TOTAL_POWER', 0.0)
            
            update_time = time.time() - start_update
            policy_update_times.append(update_time)
            losses.append(loss)
            # Calculate energy: power × time
            policy_energy = power_during_update * update_time  # Watts × seconds = Joules
            policy_update_energies.append(policy_energy)
            
            # Save the model after each policy update
            model_path = os.path.join(model_dir, f"grpo_model_episode_{episode + 1}.pth")
            torch.save(agent.policy.state_dict(), model_path)
            
            # Calculate loss change from previous episode (for CSV tracking)
            if len(losses) > 1:
                loss_change = abs(losses[-1] - losses[-2])
                loss_changes.append(loss_change)
            else:
                loss_changes.append(None)  # No change for first episode
            
            # Calculate loss convergence confidence
            if len(losses) >= convergence_window:
                recent_losses = losses[-convergence_window:]
                # Calculate changes for convergence check (local variable)
                recent_loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
                
                # Confidence based on:
                # 1. Fraction of changes below threshold
                # 2. How small the changes are relative to threshold
                below_threshold_frac = sum(1 for change in recent_loss_changes if change < convergence_threshold) / len(recent_loss_changes) if recent_loss_changes else 0.0
                
                # Average change normalized by threshold (smaller = better)
                avg_change = np.mean(recent_loss_changes) if recent_loss_changes else convergence_threshold
                change_normalized = max(0.0, 1.0 - (avg_change / convergence_threshold)) if convergence_threshold > 0 else 0.0
                
                # Combine: weighted average
                loss_convergence_confidence = 0.6 * below_threshold_frac + 0.4 * change_normalized
                
            else:
                # Not enough losses yet, confidence based on how many we have
                loss_convergence_confidence = len(losses) / convergence_window if convergence_window > 0 else 0.0
        
        # PRIORITY 1: Check reward-based exit condition FIRST (reward maximization)
        # Only exit on convergence if rewards are already high
        reward_target_reached_this_episode = False
        if dataset_max_reward is not None and len(rewards) >= reward_stable_window:
            recent_rewards = rewards[-reward_stable_window:]
            avg_recent_reward = np.mean(recent_rewards)
            reward_target = dataset_max_reward * reward_target_ratio
            
            if avg_recent_reward >= reward_target:
                reward_target_reached = True
                reward_target_reached_this_episode = True
                print(f"\n{'='*60}")
                print(f"REWARD TARGET REACHED - Training stopped early!")
                print(f"{'='*60}")
                print(f"  Episode: {episode + 1}")
                print(f"  Average reward over last {reward_stable_window} episodes: {avg_recent_reward:.4f}")
                print(f"  Target reward ({reward_target_ratio*100:.0f}% of dataset max): {reward_target:.4f}")
                print(f"  Dataset maximum reward: {dataset_max_reward:.4f}")
                print(f"  Reward ratio achieved: {avg_recent_reward / dataset_max_reward * 100:.1f}%")
                if last_action_idx is not None:
                    print(f"  Final action: idx={last_action_idx}, prefill_freq={last_prefill_freq}, decode_freq={last_decode_freq}, batch={last_batch}")
                break
        
        # PRIORITY 2: Check convergence ONLY if rewards are high enough
        # Don't exit on convergence if rewards are still low - keep training to improve rewards
        if len(losses) >= convergence_window and not reward_target_reached_this_episode:
            recent_losses = losses[-convergence_window:]
            recent_loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            
            # Only exit on convergence if average reward is reasonable (at least 65% of target)
            avg_recent_reward = np.mean(rewards[-reward_stable_window:]) if len(rewards) >= reward_stable_window else 0.0
            reward_threshold_for_convergence = dataset_max_reward * 0.8 if dataset_max_reward is not None else 0.0
            
            if all(change < convergence_threshold for change in recent_loss_changes):
                # Check if rewards are high enough to allow convergence exit
                if dataset_max_reward is None or avg_recent_reward >= reward_threshold_for_convergence:
                    converged = True
                    loss_convergence_confidence = 1.0
                    print(f"\nConverged at episode {episode + 1}! Loss stable at {losses[-1]:.4f}, Confidence: {loss_convergence_confidence:.4f}")
                    if dataset_max_reward is not None:
                        print(f"  Average reward: {avg_recent_reward:.4f} (threshold for convergence exit: {reward_threshold_for_convergence:.4f})")
                    if last_action_idx is not None:
                        print(f"  Final action: idx={last_action_idx}, prefill_freq={last_prefill_freq}, decode_freq={last_decode_freq}, batch={last_batch}")
                    break
                else:
                    # Loss is stable but rewards are too low - don't exit, keep training
                    print(f"\nLoss stable but rewards too low (avg: {avg_recent_reward:.4f} < {reward_threshold_for_convergence:.4f}). Continuing training...")
        
        if high_confidence_exit:
            break
        
        if (episode + 1) % 5 == 0:
            loss_str = f", Loss: {losses[-1]:.4f}" if losses else ""
            # Calculate confidence metrics for reporting
            # Get the most recent action_prob_confidence from the last step
            recent_action_conf = action_probs[-action_prob_window:] if len(action_probs) >= action_prob_window else action_probs
            if recent_action_conf:
                avg_recent_prob = np.mean(recent_action_conf)
                action_conf_for_report = min(1.0, avg_recent_prob / action_prob_threshold) if action_prob_threshold > 0 else 0.0
            else:
                action_conf_for_report = 0.0
            overall_conf_for_report = (action_conf_for_report + loss_convergence_confidence) / 2.0 if (action_conf_for_report > 0 or loss_convergence_confidence > 0) else 0.0
            confidence_str = f", ActionConf: {action_conf_for_report:.3f}, LossConf: {loss_convergence_confidence:.3f}, OverallConf: {overall_conf_for_report:.3f}"
            print(f"Episode {episode + 1}/{num_episodes} completed{loss_str}{confidence_str}")
    
    total_time = time.time() - total_start

    # Close CSV file
    csv_file.close()
    print(f"GRPO training results saved to: {csv_filename}")
    
    if reward_target_reached:
        print(f"\nTraining stopped early after {episode + 1} episodes due to REWARD TARGET REACHED")
        print(f"  This is the PRIMARY exit condition - reward maximization achieved!")
    elif converged:
        print(f"Training stopped early after {episode + 1} episodes due to convergence (loss stable)")
    elif high_confidence_exit:
        print(f"Training stopped early after {episode + 1} episodes due to high confidence (action prob high)")
    else:
        print(f"Completed all {num_episodes} episodes without reaching reward target, convergence, or high confidence")
    
    # Calculate all averages with safety checks for empty lists
    avg_inference_time = np.mean(env.inference_times) if env.inference_times else 0.0
    avg_batching_time = np.mean(env.batching_times) if env.batching_times else 0.0
    avg_compute_time = np.mean(env.compute_times) if env.compute_times else 0.0
    avg_action_time = np.mean(action_selection_times) if action_selection_times else 0.0
    avg_update_time = np.mean(policy_update_times) if policy_update_times else 0.0
    avg_action_prob = np.mean(action_probs) if action_probs else 0.0
    
    # Calculate average group sampling time (must be before results dict)
    avg_group_sampling_time = np.mean(group_sampling_times) if group_sampling_times else 0.0
    
    # Calculate power metrics
    avg_power = np.mean([p['avg_power_w'] for p in env.power_measurements]) if env.power_measurements else 0.0
    total_energy = np.sum([p['total_energy_j'] for p in env.power_measurements]) if env.power_measurements else 0.0
    
    results = {
        'method': 'grpo',
        'pretrain_time_sec': pretrain_time,
        'online_time_sec': total_time,
        'total_time_sec': pretrain_time + total_time,
        'num_evaluations': total_evaluations,
        'avg_inference_time_sec': avg_inference_time,
        'avg_batching_time_sec': avg_batching_time,
        'avg_compute_time_sec': avg_compute_time,
        'avg_reward': np.mean(rewards) if rewards else 0.0,
        'max_reward': np.max(rewards) if rewards else 0.0,
        'action_selection_time_ms': avg_action_time,
        'group_sampling_time_ms': avg_group_sampling_time,
        'group_size': group_size,
        'policy_update_time_sec': avg_update_time,
        'num_updates': len(policy_update_times),
        'overhead_percentage': 0.0,  # Will calculate after
        'avg_power_w': avg_power,
        'total_energy_j': total_energy,
        'energy_per_eval_j': total_energy / total_evaluations if total_evaluations > 0 else 0.0,
        'slo_violation_rate': slo_violations / total_evaluations if total_evaluations > 0 else 0,
        'temp_violation_rate': temp_violations / total_evaluations if total_evaluations > 0 else 0,
        'power_violation_rate': power_violations / total_evaluations if total_evaluations > 0 else 0
    }
    
    # Calculate RL overhead breakdown
    total_inference_time = avg_inference_time * total_evaluations
    total_action_time = avg_action_time / 1000.0 * total_evaluations  # Convert ms to sec
    total_update_time = avg_update_time * len(policy_update_times)
    total_rl_overhead = total_action_time + total_update_time
    overhead_pct = (total_rl_overhead / total_time) * 100 if total_time > 0 else 0
    overhead_per_eval_ms = (total_rl_overhead / total_evaluations) * 1000 if total_evaluations > 0 else 0.0  # Convert to ms
    
    # Calculate energy breakdown using actual measurements
    # Total energy for each phase (sum of all individual measurements)
    total_inference_energy = np.sum(inference_energies) if inference_energies else 0.0
    total_action_energy = np.sum(action_selection_energies) if action_selection_energies else 0.0
    total_policy_energy = np.sum(policy_update_energies) if policy_update_energies else 0.0
    total_rl_overhead_energy = total_action_energy + total_policy_energy
    total_energy_calculated = total_inference_energy + total_rl_overhead_energy
    
    # Calculate average power for each phase (energy / time)
    avg_inference_power = (total_inference_energy / total_inference_time) if total_inference_time > 0 else 0.0
    avg_action_power = (total_action_energy / total_action_time) if total_action_time > 0 else 0.0
    avg_policy_power = (total_policy_energy / total_update_time) if total_update_time > 0 else 0.0
    
    # Calculate energy percentages
    inference_energy_pct = (total_inference_energy / total_energy_calculated) * 100 if total_energy_calculated > 0 else 0
    action_energy_pct = (total_action_energy / total_energy_calculated) * 100 if total_energy_calculated > 0 else 0
    policy_energy_pct = (total_policy_energy / total_energy_calculated) * 100 if total_energy_calculated > 0 else 0
    rl_overhead_energy_pct = (total_rl_overhead_energy / total_energy_calculated) * 100 if total_energy_calculated > 0 else 0
    
    print(f"\nGRPO Results:")
    print(f"  Total time: {total_time} sec")
    print(f"  Evaluations: {total_evaluations}")
    print(f"  Avg inference time: {avg_inference_time} sec")
    print(f"    - Batching time: {avg_batching_time} sec ({(avg_batching_time/avg_inference_time)*100}%)" if avg_inference_time > 0 else "    - Batching time: N/A")
    print(f"    - GPU compute time: {avg_compute_time} sec ({(avg_compute_time/avg_inference_time)*100}%)" if avg_inference_time > 0 else "    - GPU compute time: N/A")
    print(f"  Avg action selection: {avg_action_time} ms (per action in group)")
    print(f"  Avg group sampling: {avg_group_sampling_time} ms (for {group_size} actions)")
    print(f"  Avg policy update: {avg_update_time} sec")
    print(f"  Avg action probability: {avg_action_prob}")
    print(f"  Avg power: {avg_power} W")
    print(f"  Total energy: {total_energy} J")
    print(f"  Avg reward: {results['avg_reward']}")
    print(f"  Max reward: {results['max_reward']}")
    print(f"  Constraint violations:")
    print(f"    - SLO: {slo_violations}/{total_evaluations} ({slo_violations/total_evaluations*100}%)")
    print(f"    - Temperature: {temp_violations}/{total_evaluations} ({temp_violations/total_evaluations*100}%)")
    print(f"    - Power: {power_violations}/{total_evaluations} ({power_violations/total_evaluations*100}%)")
    print(f"\n  GRPO Overhead Breakdown (Time):")
    print(f"    Pure inference: {total_inference_time} sec ({(total_inference_time/total_time)*100}%)" if total_time > 0 else "    Pure inference: N/A")
    print(f"    Action selection: {total_action_time} sec ({(total_action_time/total_time)*100}%)" if total_time > 0 else "    Action selection: N/A")
    print(f"    Policy updates: {total_update_time} sec ({(total_update_time/total_time)*100}%)" if total_time > 0 else "    Policy updates: N/A")
    print(f"    Total RL overhead: {total_rl_overhead} sec ({overhead_pct}%)")
    print(f"    Overhead per eval: {overhead_per_eval_ms} ms")
    print(f"\n  GRPO Energy Breakdown:")
    print(f"    Pure inference energy: {total_inference_energy} J ({inference_energy_pct}%)" if total_energy_calculated > 0 else "    Pure inference energy: N/A")
    print(f"    Action selection energy: {total_action_energy} J ({action_energy_pct}%)" if total_energy_calculated > 0 else "    Action selection energy: N/A")
    print(f"    Policy updates energy: {total_policy_energy} J ({policy_energy_pct}%)" if total_energy_calculated > 0 else "    Policy updates energy: N/A")
    print(f"    Total RL overhead energy: {total_rl_overhead_energy} J ({rl_overhead_energy_pct}%)" if total_energy_calculated > 0 else "    Total RL overhead energy: N/A")
    # Note: measured total_energy comes from env.power_measurements which may only include inference
    # The calculated energy includes all components (inference + action selection + policy updates)
    print(f"    Total calculated energy: {total_energy_calculated} J")
    print(f"    Measured energy (from env): {total_energy} J")
    if total_energy_calculated > 0 and total_energy > 0:
        diff_pct = abs(total_energy_calculated - total_energy) / total_energy_calculated * 100
        print(f"    Difference: {abs(total_energy_calculated - total_energy):.2f} J ({diff_pct:.2f}%)")
        if diff_pct > 5.0:
            print(f"    Note: Large discrepancy may indicate measured energy only includes inference phase")
    print(f"\n  Average Power (Energy / Time):")
    print(f"    Avg inference power: {avg_inference_power} W")
    print(f"    Avg action selection power: {avg_action_power} W")
    print(f"    Avg policy update power: {avg_policy_power} W")
    if baseline_power is not None:
        power_increase = avg_power - baseline_power
        power_increase_pct = (power_increase / baseline_power) * 100 if baseline_power > 0 else 0
        print(f"\n  Power Comparison vs Baseline:")
        print(f"    Baseline power: {baseline_power} W")
        print(f"    GRPO power: {avg_power} W")
        print(f"    Power increase: {power_increase:+} W ({power_increase_pct:+}%)")
    
    return results


def compare_and_save_results(grid_results, grpo_results, csv_file=None):
    """Compare grid search and GRPO results and save to CSV."""
    if csv_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"grpo_overhead_comparison_{timestamp}.csv"
    
    # Calculate overhead percentages relative to grid search
    baseline_time_per_eval = grid_results['total_time_sec'] / grid_results['num_evaluations'] if grid_results['num_evaluations'] > 0 else 0.0
    baseline_energy_per_eval = grid_results['energy_per_eval_j']
    
    grpo_time_per_eval = grpo_results['total_time_sec'] / grpo_results['num_evaluations'] if grpo_results['num_evaluations'] > 0 else 0.0
    grpo_results['overhead_percentage'] = ((grpo_time_per_eval - baseline_time_per_eval) / baseline_time_per_eval) * 100 if baseline_time_per_eval > 0 else 0.0
    grpo_results['energy_overhead_percentage'] = ((grpo_results['energy_per_eval_j'] - baseline_energy_per_eval) / baseline_energy_per_eval) * 100 if baseline_energy_per_eval > 0 else 0.0
    print("\n" + "=" * 60)

    print("TIME COMPARISON: GRID SEARCH vs GRPO")

    print("=" * 60)

    print(f"\nTotal Time:")
    print("\n" + "=" * 60)
    print("TIME COMPARISON: GRID SEARCH vs GRPO")
    print("=" * 60)
    print(f"\nTotal Time:")
    print(f"  Grid Search: {grid_results['total_time_sec']} sec")
    print(f"  GRPO:        {grpo_results['total_time_sec']} sec")
    time_diff = grpo_results['total_time_sec'] - grid_results['total_time_sec']
    time_diff_pct = (time_diff / grid_results['total_time_sec']) * 100 if grid_results['total_time_sec'] > 0 else 0.0
    print(f"  Difference:  {time_diff:+} sec ({time_diff_pct:+}%)")

    print(f"\nTime per evaluation:")
    print(f"  Grid Search: {baseline_time_per_eval} sec")
    print(f"  GRPO:        {grpo_time_per_eval} sec ({grpo_results['overhead_percentage']:+}%)")

    print(f"\nEnergy per evaluation:")
    print(f"  Grid Search: {baseline_energy_per_eval} J")
    print(f"  GRPO:        {grpo_results['energy_per_eval_j']} J ({grpo_results['energy_overhead_percentage']:+}%)")

    print(f"\nAverage power consumption:")
    print(f"  Grid Search: {grid_results['avg_power_w']} W")
    print(f"  GRPO:        {grpo_results['avg_power_w']} W")

    

    print(f"\nTime per evaluation:")

    print(f"  Grid Search: {baseline_time_per_eval} sec")

    print(f"  GRPO:        {grpo_time_per_eval} sec ({grpo_results['overhead_percentage']}%)")

    

    print(f"\nEnergy per evaluation:")

    print(f"  Grid Search: {baseline_energy_per_eval} J")

    print(f"  GRPO:        {grpo_results['energy_per_eval_j']} J ({grpo_results['energy_overhead_percentage']:}%)")

    

    print(f"\nAverage power consumption:")

    print(f"  Grid Search: {grid_results['avg_power_w']} W")

    print(f"  GRPO:        {grpo_results['avg_power_w']} W")
    print(f"  GRPO:        {grpo_time_per_eval} sec ({grpo_results['overhead_percentage']}%)")

    

    print(f"\nEnergy per evaluation:")

    print(f"  Grid Search: {baseline_energy_per_eval} J")

    print(f"  GRPO:        {grpo_results['energy_per_eval_j']} J ({grpo_results['energy_overhead_percentage']}%)")

    

    print(f"\nAverage power consumption:")

    print(f"  Grid Search: {grid_results['avg_power_w']} W")

    print(f"  GRPO:        {grpo_results['avg_power_w']} W")
    grid_results['energy_overhead_percentage'] = 0.0  # Baseline
    grid_results['overhead_percentage'] = 0.0  # Baseline
    
    # Print comparison
    print("\n" + "=" * 60)
    print("TIME COMPARISON: GRID SEARCH vs GRPO")
    print("=" * 60)
    print(f"\nTotal Time:")
    print(f"  Grid Search: {grid_results['total_time_sec']} sec")
    print(f"  GRPO:        {grpo_results['total_time_sec']} sec")
    time_diff = grpo_results['total_time_sec'] - grid_results['total_time_sec']
    time_diff_pct = (time_diff / grid_results['total_time_sec']) * 100 if grid_results['total_time_sec'] > 0 else 0.0
    print(f"  Difference:  {time_diff} sec ({time_diff_pct}%)")
    
    print(f"\nTime per evaluation:")
    print(f"  Grid Search: {baseline_time_per_eval} sec")
    print(f"  GRPO:        {grpo_time_per_eval} sec ({grpo_results['overhead_percentage']}%)")
    
    print(f"\nEnergy per evaluation:")
    print(f"  Grid Search: {baseline_energy_per_eval} J")
    print(f"  GRPO:        {grpo_results['energy_per_eval_j']} J ({grpo_results['energy_overhead_percentage']}%)")
    
    print(f"\nAverage power consumption:")
    print(f"  Grid Search: {grid_results['avg_power_w']} W")
    print(f"  GRPO:        {grpo_results['avg_power_w']} W")
    
    # Save to CSV
    fieldnames = [
        'method', 'pretrain_time_sec', 'online_time_sec', 'total_time_sec', 'num_evaluations', 'time_per_eval_sec',
        'avg_inference_time_sec', 'avg_batching_time_sec', 'avg_compute_time_sec',
        'action_selection_time_ms', 'group_sampling_time_ms', 'group_size',
        'policy_update_time_sec', 'num_updates', 
        'overhead_percentage', 'avg_power_w', 'total_energy_j', 'energy_per_eval_j', 
        'energy_overhead_percentage', 'avg_reward', 'max_reward',
        'slo_violation_rate', 'temp_violation_rate', 'power_violation_rate'
    ]
    print("\n" + "=" * 60)

    print("TIME COMPARISON: GRID SEARCH vs GRPO")
    print("=" * 60)
    print(f"\nTotal Time:")
    print(f"  Grid Search: {grid_results['total_time_sec']} sec")
    print(f"  GRPO:        {grpo_results['total_time_sec']} sec")
    time_diff = grpo_results['total_time_sec'] - grid_results['total_time_sec']
    time_diff_pct = (time_diff / grid_results['total_time_sec']) * 100 if grid_results['total_time_sec'] > 0 else 0.0
    print(f"  Difference:  {time_diff:+} sec ({time_diff_pct:+}%)")

    print(f"\nTime per evaluation:")
    print(f"  Grid Search: {baseline_time_per_eval} sec")
    print(f"  GRPO:        {grpo_time_per_eval} sec ({grpo_results['overhead_percentage']:+}%)")

    print(f"\nEnergy per evaluation:")
    print(f"  Grid Search: {baseline_energy_per_eval} J")
    print(f"  GRPO:        {grpo_results['energy_per_eval_j']} J ({grpo_results['energy_overhead_percentage']:+}%)")

    print(f"\nAverage power consumption:")
    print(f"  Grid Search: {grid_results['avg_power_w']} W")
    print(f"  GRPO:        {grpo_results['avg_power_w']} W")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for results in [grid_results, grpo_results]:
            row = results.copy()
            row['time_per_eval_sec'] = results['total_time_sec'] / results['num_evaluations'] if results['num_evaluations'] > 0 else 0.0
            # Add default values for fields that may not exist in all methods
            row.setdefault('pretrain_time_sec', 0.0)
            row.setdefault('online_time_sec', results.get('total_time_sec', 0.0))
            row.setdefault('group_sampling_time_ms', 0.0)
            row.setdefault('group_size', 1)
            row.setdefault('slo_violation_rate', 0.0)
            row.setdefault('temp_violation_rate', 0.0)
            row.setdefault('power_violation_rate', 0.0)
            row.setdefault('energy_overhead_percentage', 0.0)
            row.setdefault('overhead_percentage', 0.0)
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_file}")
    return csv_file


if __name__ == "__main__":
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"grpo_overhead_log_{timestamp}.txt"
    
    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    logger = Logger(log_file)
    sys.stdout = logger
    
    print("Starting GRPO Overhead Measurement")
    print("=" * 60)
    print(f"Log file: {log_file}\n")
    
    # Configuration
    # Option 1: Single CSV file
    #CSV_PATH = "grid_search_results_20251128_075945.csv"  # CSV file for pre-training
    
    # Option 2: Multiple CSV files
    # print(f"  CSV for pre-training: {CSV_PATH}")
    # - grid_search_results: High headroom samples (fan on, lower temp)
    # - fan_off_grid_search_results: Low headroom samples (fan off, higher temp)
    CSV_PATH = [
        "grid_search_results_20251128_075945.csv",
        "fan_off_grid_search_results_20251129_092010.csv"  # Low headroom samples
    ]   
    
    PRETRAIN_EPOCHS = 0  #Number of pre-training epochs
    NUM_SAMPLES_GRID = 3500  # Number of samples for grid search baseline
    GROUP_SIZE = 4  # GRPO group size (samples per state)
    SLO_TARGET = 15.0  # SLO target in seconds
    MAX_TEMP = 100.0  # Max temperature in Celsius
    MIN_TEMP = 40.0  # Min temperature in Celsius
    MAX_POWER = 60.0  # Max power in Watts
    
    # Set number of online evaluations
    # Option 1: Set total evaluations directly (recommended)
    NUM_EVALUATIONS = 5  # Total online evaluations
    # Option 2: Set episodes and steps separately (will be used if NUM_EVALUATIONS is None)
    NUM_EPISODES_RL = None  # Will be calculated from NUM_EVALUATIONS if not set
    MAX_STEPS_PER_EPISODE = None  # Will be calculated from NUM_EVALUATIONS if not set
    
    print(f"Configuration:")
    if isinstance(CSV_PATH, (list, tuple)):
        print(f"  CSV files for pre-training ({len(CSV_PATH)} files):")
        for i, path in enumerate(CSV_PATH, 1):
            print(f"    {i}. {path}")
    else:
        print(f"  CSV for pre-training: {CSV_PATH}")
    print(f"  Pre-training epochs: {PRETRAIN_EPOCHS}")
    print(f"  Grid search samples: {NUM_SAMPLES_GRID}")
    if NUM_EVALUATIONS is not None:
        print(f"  Online evaluations: {NUM_EVALUATIONS}")
        if NUM_EPISODES_RL is not None and MAX_STEPS_PER_EPISODE is not None:
            print(f"  RL episodes: {NUM_EPISODES_RL}, Steps per episode: {MAX_STEPS_PER_EPISODE}")
        else:
            print(f"  (Episodes and steps will be calculated automatically)")
    else:
        print(f"  RL episodes: {NUM_EPISODES_RL if NUM_EPISODES_RL is not None else 'auto'}")
        print(f"  Steps per episode: {MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE is not None else 'auto'}")
    print(f"  GRPO group size: {GROUP_SIZE}")
    if NUM_EVALUATIONS is not None:
        print(f"  Total GRPO action samples: {NUM_EVALUATIONS * GROUP_SIZE}")
    print(f"  Constraints: SLO={SLO_TARGET}s, MaxTemp={MAX_TEMP}°C, MaxPower={MAX_POWER}W")
    
    # Run experiments with same random seed for reproducibility
    SEED = 42
    print(f"  Random seed: {SEED}\n")

    # grid_results = measure_grid_search_baseline(num_samples=NUM_SAMPLES_GRID, seed=SEED,
    #                                            slo_target=SLO_TARGET, max_temp=MAX_TEMP, max_power=MAX_POWER)
    
    # Load baseline power from grid search CSV
    # print(f"\n{'='*60}")
    # print("LOADING GRID SEARCH BASELINE FROM CSV")
    # print(f"{'='*60}")
    # grid_search_csv = "grid_search_results_20251128_075945.csv"
    # print(f"Reading baseline data from: {grid_search_csv}")
    
    # df_grid = pd.read_csv(grid_search_csv)
    # baseline_power = df_grid['avg_power'].mean()
    # num_evaluations = len(df_grid)
    
    # print(f"Loaded {num_evaluations} grid search evaluations")
    # print(f"Baseline power (mean): {baseline_power:.2f} W\n")
    
    # # Create minimal grid_results dict for comparison function
    # grid_results = {
    #     'method': 'grid_search',
    #     'pretrain_time_sec': 0.0,
    #     'online_time_sec': 0.0,  # Not measured from CSV
    #     'total_time_sec': 0.0,   # Not measured from CSV
    #     'num_evaluations': num_evaluations,
    #     'avg_inference_time_sec': df_grid['inference_time'].mean() if 'inference_time' in df_grid.columns else 0.0,
    #     'avg_batching_time_sec': 0.0,  # Not in CSV
    #     'avg_compute_time_sec': 0.0,   # Not in CSV
    #     'avg_reward': 0.0,
    #     'max_reward': 0.0,
    #     'action_selection_time_ms': 0.0,
    #     'policy_update_time_sec': 0.0,
    #     'num_updates': 0,
    #     'overhead_percentage': 0.0,
    #     'avg_power_w': baseline_power,
    #     'total_energy_j': df_grid['energy'].sum() if 'energy' in df_grid.columns else 0.0,
    #     'energy_per_eval_j': df_grid['energy'].mean() if 'energy' in df_grid.columns else 0.0,
    #     'slo_violation_rate': df_grid['slo_violation'].mean() if 'slo_violation' in df_grid.columns else 0.0,
    #     'temp_violation_rate': df_grid['temp_violation'].mean() if 'temp_violation' in df_grid.columns else 0.0,
    #     'power_violation_rate': df_grid['power_violation'].mean() if 'power_violation' in df_grid.columns else 0.0
    # }
    #Measure GRPO overhead (matching total steps with grid search)
    baseline_power = 28.0
    #baseline_power = grid_results['avg_power_w']
    grpo_results = measure_grpo_overhead(csv_path=CSV_PATH, 
                                         num_evaluations=NUM_EVALUATIONS,
                                         num_episodes=NUM_EPISODES_RL, 
                                         max_steps=MAX_STEPS_PER_EPISODE, 
                                         baseline_power=baseline_power, seed=SEED, group_size=GROUP_SIZE,
                                         slo_target=SLO_TARGET, max_temp=MAX_TEMP, min_temp=MIN_TEMP, max_power=MAX_POWER,
                                         pretrain_epochs=PRETRAIN_EPOCHS)
    
    # # Compare and save results
    # csv_file = compare_and_save_results(grid_results, grpo_results)
    
    # print("\n" + "=" * 60)
    # print("MEASUREMENT COMPLETE")
    # print("=" * 60)
    # print(f"Log saved to: {log_file}")
    # print(f"Results saved to: {csv_file}")
    
    # Close the logger
    logger.close()
    sys.stdout = logger.terminal

