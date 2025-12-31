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

# PPO Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor (larger for more actions)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value

# Environment for Energy-Latency Optimization
# Environment for Energy-Latency Optimization
class EnergyLatencyEnv:
    def __init__(self):
        self.freq_bins = 11  # Frequency bins 0-10
        self.max_batch = 16  # Batch size 1-16
        self.state_dim = 3   # Now 3: prefill_freq, decode_freq, batch_size
        
        self.runner = Llama318BRunner()

        # Action space: ALL possible (prefill_freq, decode_freq, batch) combinations
        self.actions = []
        for prefill_freq in range(0, self.freq_bins):
            for decode_freq in range(0, self.freq_bins):
                for batch in range(1, self.max_batch + 1):
                    self.actions.append((prefill_freq, decode_freq, batch))
        
        self.action_dim = len(self.actions)  # 11 * 11 * 16 = 1936 actions
        print(f"Total actions: {self.action_dim}")
        
        self.reset()
    
    def reset(self):
        # Random initial state
        self.prefill_freq = np.random.randint(0, self.freq_bins)
        self.decode_freq = np.random.randint(0, self.freq_bins)
        self.batch_size = np.random.randint(1, self.max_batch + 1)
        return self._get_state()
    
    def _get_state(self):
        # Normalize state to [0, 1]
        return np.array([
            self.prefill_freq / self.freq_bins,
            self.decode_freq / self.freq_bins,
            self.batch_size / self.max_batch
        ], dtype=np.float32)
    
    def step(self, action_idx):
        # DIRECTLY SET prefill_freq, decode_freq, and batch_size
        self.prefill_freq, self.decode_freq, self.batch_size = self.actions[action_idx]
        
        # Run with separate frequencies
        results = self.runner.step_env(
            prefill_freq_bin=self.prefill_freq,
            decode_freq_bin=self.decode_freq,
            batch_size=self.batch_size
        )
        
        # ... rest of reward calculation stays the same ...
        
        
        latency_prefill = results['prefill_time']
        latency_decode = results['decode_time']
        tokens_prefill = results['num_prompt_tokens']
        tokens_decode = results['num_decode_tokens']
        power_prefill = results['prefill_TOTAL_POWER']
        power_decode = results['decode_TOTAL_POWER']
        
        
        # Reward: minimize both energy and latency (negative for minimization)
        # reward = -(((latency_prefill**2) * power_prefill)/tokens_prefill + ((latency_decode**2) * power_decode)/tokens_decode)
        reward = (tokens_prefill**2/((latency_prefill**2) * power_prefill) + (tokens_decode**2/((latency_decode**2) * power_decode)))
        
        # Normalize reward to make learning easier
        reward = reward / 100.0
        
        next_state = self._get_state()
        done = False  # Continuous task
        
        info = {'energy': latency_prefill * power_prefill + latency_decode * power_decode, 'latency': (latency_prefill + latency_decode) / (tokens_prefill + tokens_decode)}
        
        return next_state, reward, done, info
    
    def get_action_description(self, action_idx):
        """Helper to describe what an action does"""
        freq, batch = self.actions[action_idx]
        return f"Set freq={freq}, batch={batch}"


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon_clip=0.2, epochs=10, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
    
    def compute_returns(self, next_value):
        returns = []
        advantages = []
        
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']
        
        # Compute returns and advantages (GAE)
        R = next_value
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0
                gae = 0
            
            delta = rewards[i] + self.gamma * R - values[i].item()
            gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95
            
            R = rewards[i] + self.gamma * R
            
            returns.insert(0, R)
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def update(self):
        # Get final value for bootstrapping
        final_state = torch.FloatTensor(self.memory['states'][-1]).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.policy(final_state)
            next_value = next_value.item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.stack(self.memory['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.epochs):
            # Forward pass
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        return loss.item()

# Training function
def train_ppo(num_episodes=1000, max_steps=50, update_freq=2048):
    env = EnergyLatencyEnv()
    agent = PPOAgent(env.state_dim, env.action_dim, lr=3e-4)
    
    episode_rewards = []
    episode_energies = []
    episode_latencies = []
    losses = []
    
    total_steps = 0
    
    print("Training PPO Agent with DIRECT action setting...")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print("Each action directly sets (frequency, batch_size)")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_energy = []
        episode_latency = []
        
        for step in range(max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            episode_reward += reward
            episode_energy.append(info['energy'])
            episode_latency.append(info['latency'])
            
            state = next_state
            total_steps += 1
            
            # Update policy
            if total_steps % update_freq == 0:
                loss = agent.update()
                losses.append(loss)
        
        episode_rewards.append(episode_reward)
        episode_energies.append(np.mean(episode_energy))
        episode_latencies.append(np.mean(episode_latency))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_energy = np.mean(episode_energies[-100:])
            avg_latency = np.mean(episode_latencies[-100:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Energy: {avg_energy:.2f}")
            print(f"  Avg Latency: {avg_latency:.2f}")
            print("-" * 50)
    
    return agent, episode_rewards, episode_energies, episode_latencies, losses

# Test the trained agent
def test_agent(agent, env, num_episodes=10):
    print("\nTesting trained agent...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        print(f"\nTest Episode {episode + 1}")
        print(f"Initial State - Freq: {env.freq}, Batch: {env.batch_size}")
        
        total_reward = 0
        trajectory = []
        
        for step in range(10):
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            trajectory.append({
                'action': env.get_action_description(action),
                'freq': env.freq,
                'batch': env.batch_size,
                'energy': info['energy'],
                'latency': info['latency'],
                'reward': reward
            })
            
            total_reward += reward
            state = next_state
            
            if step < 3:  # Show first 3 actions
                print(f"  Step {step + 1}: {trajectory[-1]['action']}")
                print(f"    Energy={trajectory[-1]['energy']:.2f}, Latency={trajectory[-1]['latency']:.2f}")
        
        print(f"Final State - Freq: {env.freq}, Batch: {env.batch_size}")
        print(f"Final Energy: {trajectory[-1]['energy']:.2f}")
        print(f"Final Latency: {trajectory[-1]['latency']:.2f}")
        print(f"Total Reward: {total_reward:.3f}")


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

    # Debug: Print if no temperature samples were collected
    if not temp_tj_samples and not temp_cpu_samples and not temp_gpu_samples:
        print(f"Warning: No temperature samples collected. CPU: {len(temp_cpu_samples)}, GPU: {len(temp_gpu_samples)}, TJ: {len(temp_tj_samples)}")

    result_queue.put({"GPU": avg_gpu, "CPU": avg_cpu, "MEM": avg_mem, "I/O_POWER": avg_io_power, "TOTAL_POWER": total_power,
    "CPU_TEMP": avg_temp_cpu, "GPU_TEMP": avg_temp_gpu, "TJ_TEMP": avg_temp_tj})       
    

class Llama318BRunner:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        #load dataset from disk
        train_data_sample = ds = load_from_disk("nq_subset") #1000 prompts
        self.prompts = train_data_sample["question"]["text"]
        self.prompt_data_cur_index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Set pad_token to eos_token if it doesn't exist (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}
        )

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
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                current_seq_len = generated_ids.shape[1]
                
                current_attention_mask = torch.ones(
                    (batch_size, current_seq_len),
                    device=self.device,
                    dtype=attention_mask.dtype
                )
                
                outputs = self.model(
                    input_ids=generated_ids[:, -1:],
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]
                
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
        
        if verbose:
            print(f"\n=== Decode Complete (Greedy) ===")
            print(f"Generated exactly {max_new_tokens} tokens per sequence")
            print(f"Total tokens: {total_tokens}")
            print(f"Decode time: {decode_time:.4f}s")
        
        return generated_ids, total_tokens, decode_time

    def step_env(self, prefill_freq_bin, decode_freq_bin, batch_size, logging=False, set_freq=True, max_new_tokens=100):
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

            generated_ids, total_generated_tokens, decode_time_from_batch = self.decode_batch(inputs, past_key_values, max_new_tokens=max_new_tokens, verbose=False)

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


# ============================================================
# UPDATED MAIN BLOCK
# ============================================================

# Main execution
if __name__ == "__main__":
    # Train the agent
    agent, rewards, energies, latencies, losses = train_ppo(
        num_episodes=75,
        max_steps=15,
        update_freq=100
    )
    
    # Plot training results
    plot_training_results(rewards, energies, latencies, losses)
    
    # Create environment for testing and visualization
    env = EnergyLatencyEnv()
    
    # Plot action distribution
    plot_action_distribution(agent, env)
    
    # Test the trained agent
    test_agent(agent, env, num_episodes=5)
    
    # Save the model
    torch.save(agent.policy.state_dict(), 'ppo_energy_latency_model.pth')
    print("\nModel saved as 'ppo_energy_latency_model.pth'")