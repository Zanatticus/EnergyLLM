import csv
from datetime import datetime
import time
import re
import subprocess
import threading
from queue import Queue
import select
import sys
import os

# Add grpo_early_exit to path to import from it
sys.path.insert(0, '/nvme/cs242-team7/grpo_early_exit')
from rl_agent import Llama318BRunner

def initialize_fan():
    try:
        command = "sudo jetson_clocks --fan"

        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\Fan turn on")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)
        
def fan_on():
    try:
        command = "sudo sh -c 'echo 128 > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1'"

        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\Fan turn on")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)


def fan_off():
    try:
        command = "sudo sh -c 'echo 0 > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1'"
        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )       
        print(f"\Fan turn off")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)

def read_gpu_freq_hz():
    """Read actual GPU frequency from sysfs."""
    GPU_FREQ_PATH = '/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq'  # Orin
    try:
        with open(GPU_FREQ_PATH, 'r') as f:
            return int(f.read().strip())
    except:
        return None


# ============================================================
# BASELINE TEST (350 REQUESTS WITH FIXED PARAMETERS)
# ============================================================

def test_baseline(prefill_freq_bin=7, decode_freq_bin=6, batch_size=32, fan_off_at_step=50):
    """
    Run baseline inference on 350 requests with fixed parameters.
    
    Args:
        prefill_freq_bin: Prefill frequency bin (default: 0 = lowest)
        decode_freq_bin: Decode frequency bin (default: 0 = lowest)
        batch_size: Batch size (default: 32 = max)
        fan_off_at_step: Step at which to turn off fan (default: 50)
    """
    # Base fields — everything except the nested power dicts
    base_fields = [
        "timestamp", "layers", "prefill_freq_bin", "decode_freq_bin",
        "batch_size_config", "effective_batch_size", "prefill_time",
        "prefill_throughput", "num_prompt_tokens", "decode_time",
        "decode_throughput", "decode_latency_ms", "num_decode_tokens",
        "tokenize_time", "request_latency_sec", "prefill_compute_time",
        "avg_entropy"
    ]
    # Define the power keys based on measure_power_tegrastats() output structure
    # The function returns: {"GPU", "CPU", "MEM", "I/O_POWER", "TOTAL_POWER", "CPU_TEMP", "GPU_TEMP", "TJ_TEMP"}
    power_keys = ["GPU", "CPU", "MEM", "I/O_POWER", "TOTAL_POWER", "CPU_TEMP", "GPU_TEMP", "TJ_TEMP"]
    prefill_power_keys = [f"prefill_{k}" for k in power_keys]
    decode_power_keys = [f"decode_{k}" for k in power_keys]

    fieldnames = base_fields + prefill_power_keys + decode_power_keys
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"temperature_baseline/kevins_experiments/FAN_OFF_RESULTS_{timestamp}.csv"
    temperature_filename = f"temperature_baseline/kevins_experiments/FAN_OFF_TEMPERATURES_{timestamp}.csv"

    stop_monitor = Queue()

    def temperature_monitor_thread(temperature_filename, stop_queue):
        """Runs in background thread, monitors until stop_queue receives False."""
        
        cmd = ["sudo", "tegrastats", "--interval", "100"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        with open(temperature_filename, "w", newline="") as temp_f:
            writer1 = csv.writer(temp_f)
            writer1.writerow(['timestamp', 'layers', 'gpu_freq_mhz', 'cpu_freq_mhz', 'gpu_temp_c', 'cpu_temp_c', 'tj_temp_c', 'vdd_gpu_soc_mw', 'vdd_cpu_cv_mw', 'vin_sys_5v0_mw', 'vddq_vdd2_1v8ao_mw'])
            
            current_layers = None
            
            try:
                while True:
                    # Check for stop signal (non-blocking)
                    if not stop_queue.empty():
                        signal = stop_queue.get()
                        if signal is False:
                            break
                        elif isinstance(signal, int):
                            # Update current layer count
                            current_layers = signal
                            continue
                    
                    # Use select to check if data is available (with timeout)
                    # This prevents blocking indefinitely on readline()
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if not ready:
                        continue
                    
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    timestamp = time.time()
                    gpu_freq_mhz = None
                    match = re.search(r"GR3D_FREQ \d+%@\[(\d+),(\d+)\]", line)
                    if match:
                        gpu_freq_mhz = (int(match.group(1)) + int(match.group(2))) / 2

                    cpu_freq_mhz = None
                    match = re.search(r"CPU \[.*?(\d+)%@(\d+)", line)
                    if match:
                        cpu_freq_mhz = int(match.group(2))

                    gpu_temp = None
                    match = re.search(r"gpu@([\d.]+)C", line)
                    if match:
                        gpu_temp = float(match.group(1))
                    
                    cpu_temp = None
                    match = re.search(r"cpu@([\d.]+)C", line)
                    if match:
                        cpu_temp = float(match.group(1))
                    
                    tj_temp = None
                    match = re.search(r"tj@([\d.]+)C", line)
                    if match:
                        tj_temp = float(match.group(1))

                    # Power readings
                    vdd_gpu_soc = None
                    match = re.search(r"VDD_GPU_SOC (\d+)mW", line)
                    if match:
                        vdd_gpu_soc = int(match.group(1))
                    
                    vdd_cpu_cv = None
                    match = re.search(r"VDD_CPU_CV (\d+)mW", line)
                    if match:
                        vdd_cpu_cv = int(match.group(1))
                    
                    vin_sys_5v0 = None
                    match = re.search(r"VIN_SYS_5V0 (\d+)mW", line)
                    if match:
                        vin_sys_5v0 = int(match.group(1))
                    
                    vddq_vdd2_1v8ao = None
                    match = re.search(r"VDDQ_VDD2_1V8AO (\d+)mW", line)
                    if match:
                        vddq_vdd2_1v8ao = int(match.group(1))
        
                    writer1.writerow([timestamp, current_layers, gpu_freq_mhz, cpu_freq_mhz, gpu_temp, cpu_temp, tj_temp, vdd_gpu_soc, vdd_cpu_cv, vin_sys_5v0, vddq_vdd2_1v8ao])
                    temp_f.flush()
            
            finally:
                # Properly terminate the subprocess
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    # Write CSV header for inference results
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Start temperature monitor in background thread
    stop_queue = Queue()
    monitor_thread = threading.Thread(
        target=temperature_monitor_thread, 
        args=(temperature_filename, stop_queue),
        daemon=True  # Makes thread daemon so it won't prevent program exit
    )
    monitor_thread.start()

    # Run inference loop (runs in main thread)
    try:
        # runner = Llama318BRunner(model_id="facebook/layerskip-llama3.2-1B"
        runner = Llama318BRunner(model_id="meta-llama/Llama-3.2-1B")
        # max_batch = 32
        # prefill_freq = 10
        # decode_freq = 10
        
        initialize_fan()
        time.sleep(60)
        for i in range(350):
            print(f"Starting inference step {i}")
            if i == fan_off_at_step:
                fan_off()
            
            row = runner.step_env(
                prefill_freq_bin=prefill_freq_bin,
                decode_freq_bin=decode_freq_bin,
                batch_size=batch_size,
                logging=False,
                set_freq=False,
                max_new_tokens=100
            )
            row = runner.step_env(
                prefill_freq_bin=prefill_freq_bin,
                decode_freq_bin=decode_freq_bin,
                batch_size=batch_size,
                logging=False,
                set_freq=False,
                max_new_tokens=100
            )

            row['timestamp'] = time.time_ns()
            row['layers'] = i
            row['prefill_freq_bin'] = prefill_freq_bin
            row['decode_freq_bin'] = decode_freq_bin
            
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
                
    except Exception as e:
        print(f"Error during inference loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop the temperature monitor
        print("Stopping temperature monitoring...")
        stop_queue.put(False)
        monitor_thread.join(timeout=10)  # Wait max 10 seconds
        if monitor_thread.is_alive():
            print("Warning: Temperature monitor thread did not terminate cleanly")
        else:
            print("Temperature monitoring stopped successfully")
        
        print(f"\nResults saved to: {csv_file}")
        print(f"Temperature data saved to: {temperature_filename}")



# ============================================================
# TEST GRPO CHECKPOINT ON 350 REQUESTS
# ============================================================

def test_grpo_checkpoint(checkpoint_path, num_requests=350, fan_off_at_step=50):
    """
    Test a GRPO model checkpoint on 350 inference requests.
    Loads policy from grpo_early_exit.py and runs it on Jetson hardware.
    
    Args:
        checkpoint_path: Path to the saved GRPO policy checkpoint (.pth file)
        num_requests: Number of requests to run (default: 350)
        fan_off_at_step: Step at which to turn off fan (default: 50)
    """
    import torch
    import numpy as np
    from grpo_early_exit import GRPOPolicy
    
    # Base fields — everything except the nested power dicts
    base_fields = [
        "timestamp", "layers", "prefill_freq_bin", "decode_freq_bin", 
        "batch_size_config", "effective_batch_size", "prefill_time", 
        "prefill_throughput", "num_prompt_tokens", "decode_time", 
        "decode_throughput", "decode_latency_ms", "num_decode_tokens",
        "tokenize_time", "request_latency_sec", "prefill_compute_time",
        "avg_entropy"
    ]
    
    power_keys = ["GPU", "CPU", "MEM", "I/O_POWER", "TOTAL_POWER", "CPU_TEMP", "GPU_TEMP", "TJ_TEMP"]
    prefill_power_keys = [f"prefill_{k}" for k in power_keys]
    decode_power_keys = [f"decode_{k}" for k in power_keys]
    fieldnames = base_fields + prefill_power_keys + decode_power_keys
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"grpo_checkpoint_test/GRPO_TEST_{timestamp}.csv"
    temperature_filename = f"grpo_checkpoint_test/GRPO_TEST_TEMPERATURES_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Load policy checkpoint
    print(f"\nLoading GRPO policy checkpoint from: {checkpoint_path}")
    try:
        # First, load the checkpoint to see what architecture it has
        checkpoint_dict = torch.load(checkpoint_path)
        
        # Infer architecture from checkpoint
        # features.0.weight shape tells us: [hidden_dim, state_dim]
        # actor.weight shape tells us: [action_dim, hidden_dim]
        if 'features.0.weight' in checkpoint_dict and 'actor.weight' in checkpoint_dict:
            hidden_dim = checkpoint_dict['features.0.weight'].shape[0]
            state_dim = checkpoint_dict['features.0.weight'].shape[1]
            action_dim = checkpoint_dict['actor.weight'].shape[0]
            
            # Count number of layers from checkpoint structure
            # Each layer pair has weight and bias: features.0, features.2, features.4, etc.
            num_layers = 1  # Start with 1 for the first input layer
            layer_idx = 0
            while f'features.{layer_idx * 2}.weight' in checkpoint_dict:
                layer_idx += 1
                if f'features.{layer_idx * 2}.weight' in checkpoint_dict:
                    num_layers += 1
            
            print(f"  Detected architecture: state_dim={state_dim}, hidden_dim={hidden_dim}, action_dim={action_dim}, num_layers={num_layers}")
        else:
            # Fall back to defaults if we can't infer
            state_dim = 5
            hidden_dim = 32
            action_dim = 5808
            num_layers = 2
            print(f"  Using default architecture: state_dim={state_dim}, hidden_dim={hidden_dim}, action_dim={action_dim}, num_layers={num_layers}")
        
        # Create policy with correct architecture
        policy = GRPOPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Load checkpoint
        policy.load_state_dict(checkpoint_dict)
        policy.eval()
        print(f"✓ Successfully loaded GRPO policy checkpoint")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    stop_monitor = Queue()
    
    def temperature_monitor_thread(temperature_filename, stop_queue):
        """Runs in background thread, monitors temperature until stop_queue receives False."""
        cmd = ["sudo", "tegrastats", "--interval", "100"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        with open(temperature_filename, "w", newline="") as temp_f:
            writer = csv.writer(temp_f)
            writer.writerow(['timestamp', 'layers', 'gpu_freq_mhz', 'cpu_freq_mhz', 'gpu_temp_c', 'cpu_temp_c', 'tj_temp_c', 'vdd_gpu_soc_mw', 'vdd_cpu_cv_mw', 'vin_sys_5v0_mw', 'vddq_vdd2_1v8ao_mw'])
            
            current_layers = None
            
            try:
                while True:
                    if not stop_queue.empty():
                        signal = stop_queue.get()
                        if signal is False:
                            break
                        elif isinstance(signal, int):
                            current_layers = signal
                            continue
                    
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if not ready:
                        continue
                    
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    timestamp_val = time.time()
                    gpu_freq_mhz = None
                    match = re.search(r"GR3D_FREQ \d+%@\[(\d+),(\d+)\]", line)
                    if match:
                        gpu_freq_mhz = (int(match.group(1)) + int(match.group(2))) / 2

                    cpu_freq_mhz = None
                    match = re.search(r"CPU \[.*?(\d+)%@(\d+)", line)
                    if match:
                        cpu_freq_mhz = int(match.group(2))

                    gpu_temp = None
                    match = re.search(r"gpu@([\d.]+)C", line)
                    if match:
                        gpu_temp = float(match.group(1))
                    
                    cpu_temp = None
                    match = re.search(r"cpu@([\d.]+)C", line)
                    if match:
                        cpu_temp = float(match.group(1))
                    
                    tj_temp = None
                    match = re.search(r"tj@([\d.]+)C", line)
                    if match:
                        tj_temp = float(match.group(1))

                    vdd_gpu_soc = None
                    match = re.search(r"VDD_GPU_SOC (\d+)mW", line)
                    if match:
                        vdd_gpu_soc = int(match.group(1))
                    
                    vdd_cpu_cv = None
                    match = re.search(r"VDD_CPU_CV (\d+)mW", line)
                    if match:
                        vdd_cpu_cv = int(match.group(1))
                    
                    vin_sys_5v0 = None
                    match = re.search(r"VIN_SYS_5V0 (\d+)mW", line)
                    if match:
                        vin_sys_5v0 = int(match.group(1))
                    
                    vddq_vdd2_1v8ao = None
                    match = re.search(r"VDDQ_VDD2_1V8AO (\d+)mW", line)
                    if match:
                        vddq_vdd2_1v8ao = int(match.group(1))
        
                    writer.writerow([timestamp_val, current_layers, gpu_freq_mhz, cpu_freq_mhz, gpu_temp, cpu_temp, tj_temp, vdd_gpu_soc, vdd_cpu_cv, vin_sys_5v0, vddq_vdd2_1v8ao])
                    temp_f.flush()
            
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    
    # Write CSV header
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Start temperature monitor in background thread
    monitor_thread = threading.Thread(
        target=temperature_monitor_thread, 
        args=(temperature_filename, stop_monitor),
        daemon=True
    )
    monitor_thread.start()
    
    # Run inference loop with GRPO policy
    try:
        runner = Llama318BRunner(model_id="facebook/layerskip-llama3.2-1B")
        
        initialize_fan()
        time.sleep(60)
        
        print(f"\nRunning GRPO policy on {num_requests} requests...")
        
        # Initialize state tracking
        freq_bins = 11
        batch_sizes = [1, 7, 13, 19, 25, 31]
        num_batch_sizes = len(batch_sizes)
        early_exit_layers = list(range(9, 17))
        
        # Track current state from environment
        current_prefill_freq_bin = 5
        current_decode_freq_bin = 5
        current_batch_size = batch_sizes[2]  # Middle batch size
        current_early_exit_layer = 16  # Full model
        current_temperature = 50.0  # Celsius, initial estimate
        
        for request_idx in range(num_requests):
            print(f"Request {request_idx + 1}/{num_requests}")
            
            if request_idx == fan_off_at_step:
                print(f"Turning off fan at request {request_idx}")
                fan_off()
            
            # Build state from current environment state
            # State: [prefill_freq_bin_norm, decode_freq_bin_norm, batch_idx_norm, temperature_headroom, layers_norm]
            
            # Calculate temperature headroom (assuming max_temp = 100C, safe_temp = 80C)
            max_temp = 100.0
            safe_temp = 80.0
            headroom_ratio = max(0.0, min(1.0, (max_temp - current_temperature) / (max_temp - safe_temp)))
            
            # Find batch size index
            batch_idx = batch_sizes.index(current_batch_size) if current_batch_size in batch_sizes else 2
            
            # Create state tensor
            state = np.array([
                current_prefill_freq_bin / 10.0,  # Normalize to [0, 1]
                current_decode_freq_bin / 10.0,
                batch_idx / 5.0,  # 6 batch sizes: 0-5
                headroom_ratio,
                (current_early_exit_layer - 9) / 7.0  # Normalize layers 9-16 to [0, 1]
            ], dtype=np.float32)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action_idx, log_prob = policy.act(state_tensor, deterministic=False)
            
            # Decode action index to parameters
            action_idx = int(action_idx)
            
            current_prefill_freq_bin = (action_idx // (freq_bins * num_batch_sizes * 8))
            remainder = action_idx % (freq_bins * num_batch_sizes * 8)
            current_decode_freq_bin = (remainder // (num_batch_sizes * 8))
            current_batch_size = batch_sizes[(remainder // 8) % num_batch_sizes]
            current_early_exit_layer = early_exit_layers[remainder % 8]
            
            # Run inference with policy-selected action
            row = runner.step_env(
                prefill_freq_bin=current_prefill_freq_bin,
                decode_freq_bin=current_decode_freq_bin,
                batch_size=current_batch_size,
                early_exit_layer=current_early_exit_layer,
                logging=False,
                set_freq=True,
                max_new_tokens=100
            )
            
            row['timestamp'] = time.time_ns()
            row['layers'] = current_early_exit_layer
            row['prefill_freq_bin'] = current_prefill_freq_bin
            row['decode_freq_bin'] = current_decode_freq_bin
            
            # Update temperature estimate from latest measurements
            if 'prefill_GPU_TEMP' in row:
                current_temperature = row['prefill_GPU_TEMP']
            elif 'decode_GPU_TEMP' in row:
                current_temperature = row['decode_GPU_TEMP']
            # If neither, keep the previous estimate
            
            # Write to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            
            if (request_idx + 1) % 50 == 0:
                print(f"  Completed {request_idx + 1} requests...")
    
    except Exception as e:
        print(f"Error during inference loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Stopping temperature monitoring...")
        stop_monitor.put(False)
        monitor_thread.join(timeout=10)
        if monitor_thread.is_alive():
            print("Warning: Temperature monitor thread did not terminate cleanly")
        else:
            print("Temperature monitoring stopped successfully")
        
        print(f"\nResults saved to: {csv_file}")
        print(f"Temperature data saved to: {temperature_filename}")


if __name__ == "__main__":
    # test_baseline()
    # test_baseline()(prefill_freq_bin=7, decode_freq_bin=6, batch_size=32, fan_off_at_step=50)
    checkpoint_path = "/nvme/cs242-team7/grpo_early_exit/models/grpo_early_exit_model3.pth"
    test_grpo_checkpoint(checkpoint_path, num_requests=350, fan_off_at_step=50)

