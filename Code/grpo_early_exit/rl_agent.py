import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
from datasets import load_dataset, Dataset
from datasets import load_from_disk
import subprocess
import time
import re
import multiprocessing
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import sys
from typing import List, Optional
import json
import csv
from datetime import datetime
from datasets import load_dataset
from datasets import load_from_disk
import random
from copy import deepcopy


def set_gpu_frequency(freq_hz: int) -> bool:
    """Set GPU frequency in Hz. Returns True if successful."""
    try:
        # freq_khz = freq_hz // 1000
        base_path = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
        min_freq_path = f"{base_path}/min_freq"
        max_freq_path = f"{base_path}/max_freq"
        
        # Set min and max to the same value to lock frequency
        cmd1 = f"echo {freq_hz} | sudo /usr/bin/tee {min_freq_path}"
        cmd2 = f"echo {freq_hz} | sudo /usr/bin/tee {max_freq_path}"
        
        result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
        result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)

        if result1.returncode == 0 and result2.returncode == 0:
            # Wait a moment for frequency to stabilize, then verify
            time.sleep(0.1)
            # Retry verification up to 3 times with small delays
            cur_freq = None
            for attempt in range(3):
                cur_freq = get_gpu_frequency()
                if cur_freq is not None:
                    print(f"Current frequency: {cur_freq/1e6:.0f}MHz")
                if cur_freq is None:
                    if attempt < 2:
                        time.sleep(0.1)
                        continue
                    print(f"Warning: Could not read GPU frequency after setting to {freq_hz/1e6:.0f}MHz")
                    return False, None
                if cur_freq == freq_hz:
                    return True, cur_freq
                if attempt < 2:
                    time.sleep(0.1)
            
            # If we get here, frequency didn't match after retries
            if cur_freq is None or cur_freq != freq_hz:
                if cur_freq is None:
                    print(f"Warning: Could not verify GPU frequency after setting to {freq_hz/1e6:.0f}MHz")
                else:
                    print(f"Warning: GPU frequency set to {cur_freq/1e6:.0f}MHz instead of {freq_hz/1e6:.0f}MHz")
                return False, None
        else:
            print(f"Warning: Could not set GPU frequency to {freq_hz/1e6:.0f}MHz")
            if result1.stderr:
                print(f"  min_freq error: {result1.stderr}")
            if result2.stderr:
                print(f"  max_freq error: {result2.stderr}")
            return False, None
    except Exception as e:
        print(f"Warning: Could not set GPU frequency: {e}")
        return False, None

def get_gpu_frequency() -> Optional[int]:
    """Get current GPU frequency in Hz."""
    # Correct path for Jetson AGX Orin
    cur_freq_path = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"
    
    try:
        with open(cur_freq_path, "r") as f:
            freq_str = f.read().strip()
            # The file contains frequency in Hz (e.g., "306000000")
            freq_hz = int(freq_str)
            return freq_hz
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Warning: Could not read GPU frequency from {cur_freq_path}: {e}")
        return None

def set_gpu_freq_bin(bin: int):
    GPU_FREQ_BINS = [
        306000000, 
        408000000, 
        510000000, 
        612000000, 714000000, 816000000, 918000000, 1020000000, 1122000000, 1224000000, 1300500000
    ]
    assert 0 <= bin <= 10, f"expected 0..10 inclusive, got {bin}"
    target_freq = GPU_FREQ_BINS[bin]
    success, actual_freq = set_gpu_frequency(target_freq)
    if not success:
        print(f"Warning: Failed to set GPU frequency to {target_freq/1e6:.0f} MHz, skipping...")

def measure_power_tegrastats(queue, result_queue):
    cmd = ["tegrastats", "--interval", "1"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_samples, cpu_samples, mem_samples, io_power_samples = [], [], [], []
    temp_cpu_samples, temp_gpu_samples, temp_tj_samples = [], [], []
    sample_interval = 0.001
    first_line = process.stdout.readline()
    while not re.search(r"VDD_", first_line):
        first_line = process.stdout.readline()
    queue.get()
    try:
        while True:
            if not queue.empty():
                stop_signal = queue.get()
                if stop_signal is False:
                    break
            line = process.stdout.readline()
            if not line:
                continue
            match_gpu = re.search(r"VDD_GPU_SOC (\d+)mW", line)
            if match_gpu:
                gpu_samples.append(int(match_gpu.group(1)))
            match_cpu = re.search(r"VDD_CPU_CV (\d+)mW", line)
            if match_cpu:
                cpu_samples.append(int(match_cpu.group(1)))
            match_mem = re.search(r"VDDQ_VDD2_1V8AO (\d+)mW", line)
            if match_mem:
                mem_samples.append(int(match_mem.group(1)))
            match_io_power = re.search(r"VIN_SYS_5V0 (\d+)mW", line)
            if match_io_power:
                io_power_samples.append(int(match_io_power.group(1)))

            # -----------------------------
            # Temperature Sensors
            # -----------------------------
            # CPU temperature (cpu@40.812C)
            match_temp_cpu = re.search(r"cpu@([0-9.]+)C", line)
            if match_temp_cpu:
                temp_cpu_samples.append(float(match_temp_cpu.group(1)))

            # GPU temperature (gpu@36.718C)
            match_temp_gpu = re.search(r"gpu@([0-9.]+)C", line)
            if match_temp_gpu:
                temp_gpu_samples.append(float(match_temp_gpu.group(1)))

            # TJ (junction temp, hottest spot)
            match_temp_tj = re.search(r"tj@([0-9.]+)C", line)
            if match_temp_tj:
                temp_tj_samples.append(float(match_temp_tj.group(1)))

            time.sleep(sample_interval)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
    avg_gpu = sum(gpu_samples) / len(gpu_samples) / 1000 if gpu_samples else 0
    avg_cpu = sum(cpu_samples) / len(cpu_samples) / 1000 if cpu_samples else 0
    avg_mem = sum(mem_samples) / len(mem_samples) / 1000 if mem_samples else 0
    avg_io_power = sum(io_power_samples) / len(io_power_samples) / 1000 if io_power_samples else 0
    total_power = avg_gpu + avg_cpu + avg_mem + avg_io_power

    avg_temp_cpu = sum(temp_cpu_samples) / len(temp_cpu_samples) if temp_cpu_samples else 0
    avg_temp_gpu = sum(temp_gpu_samples) / len(temp_gpu_samples) if temp_gpu_samples else 0
    avg_temp_tj = sum(temp_tj_samples) / len(temp_tj_samples) if temp_tj_samples else 0

    result_queue.put({"GPU": avg_gpu, "CPU": avg_cpu, "MEM": avg_mem, "I/O_POWER": avg_io_power, "TOTAL_POWER": total_power,
    "CPU_TEMP": avg_temp_cpu, "GPU_TEMP": avg_temp_gpu, "TJ_TEMP": avg_temp_tj})       
    

class Llama318BRunner:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        #load dataset from disk
        # Try multiple possible paths for nq_subset dataset
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        # Try: parent_dir/nq_subset, script_dir/nq_subset, or just "nq_subset" (current working directory)
        possible_paths = [
            os.path.join(parent_dir, "nq_subset"),  # /nvme/cs242-team7/nq_subset
            os.path.join(script_dir, "nq_subset"),  # /nvme/cs242-team7/grpo_early_exit/nq_subset
            "nq_subset"  # Current working directory
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            raise FileNotFoundError(
                f"Could not find 'nq_subset' dataset directory. Tried:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                f"\n\nPlease ensure the dataset is available at one of these locations."
            )
        
        train_data_sample = ds = load_from_disk(dataset_path) #1000 prompts
        self.prompts = train_data_sample["question"]["text"]
        self.prompt_data_cur_index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Set pad_token to eos_token if it doesn't exist (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.full_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True, 
            device_map={"": "cuda"}
        )
        # Extra attributes to enable early exit
        self.model = self.full_model
        self.num_layers = len(self.full_model.model.layers)
        self.current_early_exit = self.num_layers  # Track current exit point
    
    def set_early_exit(self, early_exit_layer: int):
        if early_exit_layer is None or early_exit_layer >= self.num_layers:
            # Reset to full model
            self.model = self.full_model
            self.current_early_exit = self.num_layers
            print(f"Using full model ({self.num_layers} layers)")
        else:
            # Create truncated model with shared weights
            if early_exit_layer < 1:
                raise ValueError(f"early_exit_layer must be >= 1, got {early_exit_layer}")
            
            # Delete old truncated model if it exists
            if hasattr(self, '_truncated_model') and self._truncated_model is not None:
                del self._truncated_model
                torch.cuda.empty_cache()
            
            # Create new truncated model
            memo = {id(w): w for w in self.full_model.parameters()}

            # print("full model should be 16!:", len(self.full_model.model.layers))
            self._truncated_model = deepcopy(self.full_model, memo=memo)
            # print("this should be 16!:", len(self._truncated_model.model.layers))
            self._truncated_model.model.layers = self._truncated_model.model.layers[:early_exit_layer]
            
            self.model = self._truncated_model
            self.current_early_exit = early_exit_layer
            print(f"Using early exit at layer {early_exit_layer}/{self.num_layers}")


    def prefill_batch(self, prompts: List[str]):
        """Prefill phase for batch of prompts."""
        # Time the tokenization/batching (CPU operations)
        start_tokenize = time.time()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.device)
        tokenize_time = time.time() - start_tokenize
    
        # Time the GPU prefill computation
        torch.cuda.synchronize()
        start_compute = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        compute_time = time.time() - start_compute
    
        return inputs, outputs.past_key_values, tokenize_time, compute_time

    def decode_batch(self, inputs, past_key_values, max_new_tokens=100, 
                    temperature=0.3, top_p=0.5, repetition_penalty=1.2, verbose=False):
        """Decode phase for batch using greedy search (no early stopping)."""
        generated_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = generated_ids.shape[0]
        original_length = generated_ids.shape[1]
        
        # No early stopping - generate exactly max_new_tokens
        # finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # eos_token_id = self.tokenizer.eos_token_id
        # tokens_generated = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Track finish times for each sequence (not used in greedy mode)
        # finish_times = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        start_time = time.time()  # Track decode time
        
        if verbose:
            print(f"\n=== Starting Decode (batch_size={batch_size}) ===")
            print(f"Original prompt length: {original_length} tokens")
            print(f"Greedy search: generating exactly {max_new_tokens} tokens")
            print("-" * 60)

        all_entropies = []
        with torch.no_grad():
            for i in range(max_new_tokens):
                current_seq_len = generated_ids.shape[1]
                
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)
                
                outputs = self.model(
                    input_ids=generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Calculate entropy for Early Exit Layer
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)  # [batch_size, vocab_size]
                log_probs = torch.log(probs + 1e-10)   # [batch_size, vocab_size]
                entropies = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]
                all_entropies.append(entropies)
                
                # Greedy search: take argmax (highest probability token)
                next_tokens = torch.argmax(logits, dim=-1)
                
                # # Original sampling code (commented out for greedy)
                # # Repetition penalty
                # vocab_size = logits.shape[-1]
                # token_indices = generated_ids.clamp(0, vocab_size - 1)
                # gathered_logits = torch.gather(logits, dim=1, index=token_indices)
                # gathered_logits = torch.where(
                #     gathered_logits > 0,
                #     gathered_logits / repetition_penalty,
                #     gathered_logits * repetition_penalty
                # )
                # logits.scatter_(dim=1, index=token_indices, src=gathered_logits)
                # 
                # # Temperature
                # logits = logits / temperature
                # 
                # # Top-p sampling
                # sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # 
                # sorted_indices_to_remove = cumulative_probs > top_p
                # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                # sorted_indices_to_remove[..., 0] = False
                # 
                # sorted_logits[sorted_indices_to_remove] = float('-inf')
                # logits = torch.zeros_like(logits).scatter_(dim=1, index=sorted_indices, src=sorted_logits)
                # 
                # # Sample
                # probs = torch.softmax(logits, dim=-1)
                # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # # Early stopping logic (commented out for greedy)
                # # Track which sequences just finished
                # just_finished = ~finished & (next_tokens == eos_token_id)
                # tokens_generated[just_finished] = i + 1
                # 
                # # Record finish time for sequences that just finished
                # finish_times[just_finished] = time.time() - start_time
                # 
                # next_tokens = torch.where(
                #     finished,
                #     torch.full_like(next_tokens, eos_token_id),
                #     next_tokens
                # )
                
                generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)
                past_key_values = outputs.past_key_values
                
                # # Early stopping check (commented out for greedy)
                # finished = finished | (next_tokens == eos_token_id)
                # 
                # if finished.all():
                #     if verbose:
                #         print(f"\n=== All sequences finished at step {i+1} ===")
                #     break
        
        # Greedy search: all sequences generate exactly max_new_tokens
        # No early stopping, so total tokens = batch_size * max_new_tokens
        decode_time = time.time() - start_time
        total_tokens = batch_size * max_new_tokens
        all_entropies = torch.stack(all_entropies)  # [num_steps, batch_size]
        avg_entropy_per_sequence = all_entropies.mean(dim=0)  # [batch_size]
        avg_entropy_overall = all_entropies.mean()  # scalar        
        if verbose:
            print(f"\n=== Decode Complete (Greedy) ===")
            print(f"Generated exactly {max_new_tokens} tokens per sequence")
            print(f"Total tokens: {total_tokens}")
            print(f"Decode time: {decode_time:.4f}s")

        return generated_ids, total_tokens, decode_time, avg_entropy_overall

    def step_env(self, prefill_freq_bin, decode_freq_bin, batch_size, logging=False, set_freq=True, max_new_tokens=100, early_exit_layer=None):
        #This is for when we run into the end of the prompts, this wraps around so there are no out of bounds issues
        def slice_wrap(lst, i, n):
            """Return n items starting at index i (wrap), and the new i."""
            if not lst or n <= 0:
                return [], i
            L = len(lst)
            start = i % L
            chunk = [lst[(start + k) % L] for k in range(n)]
            new_i = (start + n) % L
            return chunk, new_i
        
        try:
            # Time the prompt selection/batching
            prompt_selection_start = time.time()
            if logging:
                print(f"Logging enabled. Batch size: {batch_size}")
                batch_prompts = random.sample(self.prompts, batch_size)
            else:
                batch_prompts, self.prompt_data_cur_index = slice_wrap(self.prompts, self.prompt_data_cur_index, batch_size)
            prompt_selection_time = time.time() - prompt_selection_start

            if early_exit_layer is not None:
                self.set_early_exit(early_exit_layer)

            # ---------------- Prefill Phase ----------------
            queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=measure_power_tegrastats, args=(queue, result_queue)
            )

            #set prefill freq bin on GPU
            if set_freq:
                # Clamp to valid range [0, 10] since GPU_FREQ_BINS has 11 elements (indices 0-10)
                prefill_freq_bin_clamped = max(0, min(10, int(prefill_freq_bin)))
                if prefill_freq_bin != prefill_freq_bin_clamped:
                    print(f"Warning: Clamping prefill_freq_bin from {prefill_freq_bin} to {prefill_freq_bin_clamped}")
                set_gpu_freq_bin(prefill_freq_bin_clamped)
            power_process.start()
            # Give power process a moment to initialize tegrastats
            time.sleep(0.1)

            queue.put(True)
            torch.cuda.synchronize()
            start_prefill = time.time()
            inputs, past_key_values, tokenize_time, compute_time = self.prefill_batch(batch_prompts)
            
            # Add prompt selection time to tokenize time (both are CPU preprocessing)
            tokenize_time += prompt_selection_time

            torch.cuda.synchronize()
            end_prefill = time.time()

            queue.put(False)
            power_process.join()
            prefill_power = result_queue.get()
            
            prefill_time = end_prefill - start_prefill

            # inputs['input_ids'].shape = (effective_batch_size, seq_len)
            effective_batch = inputs["input_ids"].shape[0]
            seq_len = inputs["input_ids"].shape[1]
            num_prompt_tokens = effective_batch * seq_len
            prefill_throughput = num_prompt_tokens / prefill_time if prefill_time > 0 else 0

            print(f"\n=== Prefill Phase ===")
            print(f"Effective batch size: {effective_batch}")
            print(f"Tokenize time: {tokenize_time:.4f} sec")
            print(f"Prefill compute time: {compute_time:.4f} sec")
            print(f"Total prefill time: {prefill_time:.4f} sec")
            print(f"Input Tokens: {num_prompt_tokens}")
            print(f"**Prefill Throughput: {prefill_throughput:.2f} tokens/sec**")
            print(f"Power: {prefill_power}")

            # ---------------- Decode Phase ----------------
            queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            power_process = multiprocessing.Process(
                target=measure_power_tegrastats, args=(queue, result_queue)
            )

            #set decode freq bin on GPU
            if set_freq:
                # Clamp to valid range [0, 10] since GPU_FREQ_BINS has 11 elements (indices 0-10)
                decode_freq_bin_clamped = max(0, min(10, int(decode_freq_bin)))
                if decode_freq_bin != decode_freq_bin_clamped:
                    print(f"Warning: Clamping decode_freq_bin from {decode_freq_bin} to {decode_freq_bin_clamped}")
                set_gpu_freq_bin(decode_freq_bin_clamped)
            power_process.start()
            # Give power process a moment to initialize tegrastats
            time.sleep(0.1)
            queue.put(True)
            torch.cuda.synchronize()
            start_decode = time.time() 

            generated_ids, total_generated_tokens, decode_time_from_batch, avg_entropy_overall = self.decode_batch(inputs, past_key_values, max_new_tokens=max_new_tokens, verbose=False)

            torch.cuda.synchronize()
            end_decode = time.time()

            queue.put(False)
            power_process.join()
            decode_power = result_queue.get()

            # Use wall-clock time for decode phase
            decode_time = end_decode - start_decode
            decode_throughput = ( total_generated_tokens / decode_time ) if decode_time > 0 else 0  #tokens/second (batch-level)
            # Decode latency per sequence (not affected by batch size)
            tokens_per_sequence = total_generated_tokens / effective_batch if effective_batch > 0 else 0
            decode_latency_ms = (
                (decode_time / tokens_per_sequence) * 1000 if tokens_per_sequence > 0 else 0  #ms/token per sequence
            )

            print(f"\n=== Decode Phase ===")
            print(f"Time: {decode_time:.4f} sec")
            print(f"Generated Tokens: {total_generated_tokens} (effective_batch_size={effective_batch})")
            print(f"**Decode Throughput: {decode_throughput:.2f} tokens/sec**")
            print(f"**Decode Latency: {decode_latency_ms:.2f} ms/token**")
            print(f"Power: {decode_power}")
            
            # Build result dictionary
            row = {
                "batch_size_config": batch_size,
                "effective_batch_size": effective_batch,
                "tokenize_time": round(tokenize_time, 4),
                "prefill_compute_time": round(compute_time, 4),
                "prefill_time": round(prefill_time, 4),
                "prefill_throughput": round(prefill_throughput, 4),
                "num_prompt_tokens": num_prompt_tokens,
                "decode_time": round(decode_time, 4),
                "decode_throughput": round(decode_throughput, 4),
                "decode_latency_ms": round(decode_latency_ms, 4),
                "num_decode_tokens": total_generated_tokens,
                "request_latency_sec": round(prefill_time + decode_time, 4),
                "avg_entropy": round(avg_entropy_overall.item(), 4)
            }

            # Add power metrics
            if isinstance(prefill_power, dict):
                for k, v in prefill_power.items():
                    row[f"prefill_{k}"] = round(v, 4) if isinstance(v, (int, float)) else v

            if isinstance(decode_power, dict):
                for k, v in decode_power.items():
                    row[f"decode_{k}"] = round(v, 4) if isinstance(v, (int, float)) else v
            return row

        except Exception as e:
            print(
                f"Error during step_env with batch_size={batch_size}, "
                f"prefill_freq_bin={prefill_freq_bin}, decode_freq_bin={decode_freq_bin}: {e}"
            )
            import traceback
            traceback.print_exc()
            # Return a default result or re-raise
            raise

# Add this visualization function before the main block

# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_results(episode_rewards, episode_energies, episode_latencies, 
                          losses, save_path='ppo_training_results.png'):
    """Plot training results for online PPO."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, color='blue')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of rewards
    window = min(10, len(episode_rewards))
    if window > 0 and len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                        color='red', linewidth=2, label=f'Moving Avg (w={window})')
        axes[0, 0].legend()
    
    # Energy
    axes[0, 1].plot(episode_energies, color='red')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Energy')
    axes[0, 1].set_title('Energy Consumption')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Latency
    axes[1, 0].plot(episode_latencies, color='green')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Latency')
    axes[1, 0].set_title('Latency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training loss
    if losses:
        axes[1, 1].plot(losses, color='orange')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No updates yet', ha='center', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved as '{save_path}'")
    plt.close()


def plot_action_distribution(agent, env, save_path='action_distribution.png'):
    """Visualize what actions the trained policy prefers."""
    freq_bins = env.freq_bins
    max_batch = env.max_batch
    
    # Get action probabilities for different states
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Test states
    test_configs = [
        ([0.0, 0.0, 0.0], "Low prefill, Low decode, Low batch"),
        ([1.0, 1.0, 1.0], "High prefill, High decode, High batch"),
        ([0.5, 0.5, 0.5], "Mid prefill, Mid decode, Mid batch"),
        ([0.0, 1.0, 0.5], "Low prefill, High decode, Mid batch"),
    ]
    
    agent.policy.eval()
    with torch.no_grad():
        for ax, (state, title) in zip(axes.flat, test_configs):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = agent.policy(state_tensor)
            probs = action_probs.squeeze().numpy()
            
            # Reshape to (prefill_freq, decode_freq, batch) and sum over batch
            probs_reshaped = probs.reshape(freq_bins, freq_bins, max_batch)
            probs_freq = probs_reshaped.sum(axis=2)  # Sum over batch sizes
            
            im = ax.imshow(probs_freq, cmap='hot', origin='lower', aspect='auto')
            ax.set_xlabel('Decode Frequency Bin')
            ax.set_ylabel('Prefill Frequency Bin')
            ax.set_title(f'{title}\nValue: {value.item():.4f}')
            plt.colorbar(im, ax=ax, label='Probability')
    
    agent.policy.train()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAction distribution plot saved as '{save_path}'")
    plt.close()
