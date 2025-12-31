#!/usr/bin/env python3
"""
Add synthetic entropy and early exit layer values to existing CSV files.

This script allows you to quickly add entropy and early exit layer columns
to existing CSV files without waiting for full data collection.

Usage:
    python add_synthetic_entropy_to_csv.py \
        --input grid_search_results_20251128_075945.csv \
        --output grid_search_results_with_entropy.csv \
        --early-exit-layers 4,6,8,10,12,14,16
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_entropy(early_exit_layer, batch_size, temperature=None, latency=None, 
                               num_layers=16, base_entropy=5.0, temp_max=100.0, latency_max=20.0):
    """
    Generate synthetic entropy value based on early exit layer, temperature, and latency.
    
    Heuristics:
    - Fewer layers = higher entropy (less accurate, model is less confident)
    - More layers = lower entropy (more accurate, model is more confident)
    - Higher temperature = higher entropy (thermal stress affects model confidence)
    - Higher latency = higher entropy (slower inference may indicate model uncertainty)
    - Add random noise to simulate variation
    
    Args:
        early_exit_layer: Number of layers used (4, 6, 8, 10, 12, 14, 16)
        batch_size: Batch size (for minor variation)
        temperature: Current temperature in Celsius (optional, for skewing)
        latency: Current latency in seconds (optional, for skewing)
        num_layers: Total number of layers in model (default: 16)
        base_entropy: Base entropy for full model (default: 5.0)
        temp_max: Maximum temperature for normalization (default: 100.0)
        latency_max: Maximum latency for normalization (default: 20.0)
    
    Returns:
        Synthetic entropy value in range [0.5, 9.5]
    """
    # Base entropy based on layer count
    # Full model (16 layers) -> very low entropy (~0.5-1.0) when temperature is low
    # Early exit (4 layers) -> higher entropy (~6.0-8.0)
    layer_ratio = early_exit_layer / num_layers
    
    # Base entropy varies from ~0.5-1.0 (16 layers at low temp) to ~6.5 (4 layers)
    # For 16 layers at low temp, we want entropy around 0-1.5, so base should be around 0.5-1.0
    # For 16 layers at high temp, base can be slightly higher but still low
    # For 4-6 layers, we want entropy upper bound around 7-9
    # Formula: base = entropy_min + (entropy_max - entropy_min) * (1.0 - layer_ratio)
    # For 16 layers (ratio=1.0): base = entropy_min
    # For 4 layers (ratio=0.25): base = entropy_min + (entropy_max - entropy_min) * 0.75
    
    # Adjust base entropy for 16 layers based on temperature
    # < 70°C: very low entropy (0.3-0.5), prioritize 16 layers
    # 70-80°C: transition zone
    # > 80°C: slightly higher but still low (0.8-1.0)
    if temperature is not None and temp_max > 0:
        temp_low_threshold = 70.0  # 70°C
        temp_high_threshold = 80.0  # 80°C
        temp_low_ratio = temp_low_threshold / temp_max
        temp_high_ratio = temp_high_threshold / temp_max
        temp_normalized = np.clip(temperature / temp_max, 0.0, 1.0)
        
        if temperature < temp_low_threshold:
            # Low temp (< 70°C): very low base entropy for 16 layers
            entropy_min = 0.3  # Very low for 16 layers at low temp
        elif temperature < temp_high_threshold:
            # Transition zone (70-80°C): smooth transition
            transition_scale = (temperature - temp_low_threshold) / (temp_high_threshold - temp_low_threshold)
            entropy_min = 0.3 + transition_scale * 0.5  # 0.3 at 70°C, 0.8 at 80°C
        else:
            # High temp (>= 80°C): slightly higher base entropy for 16 layers
            high_temp_scale = (temp_normalized - temp_high_ratio) / (1.0 - temp_high_ratio)
            entropy_min = 0.8 + high_temp_scale * 0.2  # 0.8 at 80°C, 1.0 at 100°C
    else:
        entropy_min = 0.3  # Default very low for 16 layers
    
    # Adjust entropy_max (base for 4 layers) based on temperature
    # At low temp, even 4 layers should have lower entropy
    if temperature is not None and temp_max > 0:
        temp_low_threshold = 70.0
        temp_high_threshold = 80.0
        
        if temperature < temp_low_threshold:
            # Low temp: 4 layers should have lower base entropy too
            entropy_max = 3.0  # Much lower for 4 layers at low temp
        elif temperature < temp_high_threshold:
            # Transition zone: smooth transition
            transition_scale = (temperature - temp_low_threshold) / (temp_high_threshold - temp_low_threshold)
            entropy_max = 3.0 + transition_scale * 5.33  # 3.0 at 70°C, 8.33 at 80°C
        else:
            # High temp: 4 layers can have high entropy
            entropy_max = 8.33  # High for 4 layers at high temp
    else:
        entropy_max = 8.33  # Default high for 4 layers
    
    base_entropy_value = entropy_min + (entropy_max - entropy_min) * (1.0 - layer_ratio)
    
    # Temperature effect: higher temp -> higher entropy
    # < 70°C: minimal entropy increase (prioritize 16 layers with low entropy)
    # 70-80°C: smooth transition
    # > 80°C: significant entropy increase (prioritize 4 layers with high entropy)
    temp_effect = 0.0
    if temperature is not None and temp_max > 0:
        temp_low_threshold = 70.0  # 70°C
        temp_high_threshold = 80.0  # 80°C
        temp_low_ratio = temp_low_threshold / temp_max
        temp_high_ratio = temp_high_threshold / temp_max
        temp_normalized = np.clip(temperature / temp_max, 0.0, 1.0)
        
        if temperature < temp_low_threshold:
            # Low temperature (< 70°C): minimal entropy increase
            # Scale down effect significantly, especially for more layers
            temp_scale = (temperature / temp_low_threshold) ** 2.0  # Quadratic, very small
            layer_effect_scale = 1.0 - (layer_ratio * 0.9)  # 16 layers: 0.1x effect, 4 layers: 1.0x effect
            temp_effect = (temp_scale * 0.3) * layer_effect_scale  # Max +0.3 entropy at 70°C (for 4 layers)
        elif temperature < temp_high_threshold:
            # Transition zone (70-80°C): smooth transition
            transition_scale = (temperature - temp_low_threshold) / (temp_high_threshold - temp_low_threshold)
            # Transition from minimal to significant effect
            base_effect = 0.3 + transition_scale * 1.2  # 0.3 at 70°C, 1.5 at 80°C
            layer_effect_scale = 1.0 - (layer_ratio * 0.7)  # 16 layers: 0.3x effect, 4 layers: 1.0x effect
            temp_effect = base_effect * layer_effect_scale
        else:
            # High temperature (>= 80°C): significant entropy increase
            # Map [80°C, 100°C] to [0.0, 1.0] for exponential scaling
            high_temp_normalized = (temp_normalized - temp_high_ratio) / (1.0 - temp_high_ratio)
            layer_effect_scale = 1.0 - (layer_ratio * 0.5)  # 16 layers: 0.5x effect, 4 layers: 1.0x effect
            # Strong exponential effect above 80°C
            temp_effect = (1.5 + high_temp_normalized ** 1.5 * 2.0) * layer_effect_scale  # 1.5 at 80°C, 3.5 at 100°C (for 4 layers)
    
    # Latency effect: higher latency -> higher entropy
    # Normalize latency: 0.0 (low latency) to 1.0 (high latency)
    latency_effect = 0.0
    if latency is not None and latency_max > 0:
        latency_normalized = np.clip(latency / latency_max, 0.0, 1.0)
        # Stronger effect at higher latencies (exponential)
        # Scale effect by layer count: less effect for more layers
        layer_effect_scale = 1.0 - (layer_ratio * 0.5)  # 16 layers: 0.5x effect, 4 layers: 1.0x effect
        # Max effect to allow 4-6 layers to reach 7-9 range
        latency_effect = (latency_normalized ** 1.5 * 1.5) * layer_effect_scale  # Max +1.5 entropy at high latency (for 4 layers)
    
    # Combine effects
    synthetic_entropy = base_entropy_value + temp_effect + latency_effect
    
    # Add random noise - smaller noise for more layers
    # For 16 layers: ±0.15 noise, for 4-6 layers: ±0.5 noise (allows reaching 7-9)
    noise_scale = 0.15 + (1.0 - layer_ratio) * 0.35  # 16 layers: 0.15, 4 layers: 0.5
    noise = np.random.normal(0, noise_scale)
    synthetic_entropy += noise
    
    # Add minor batch size effect (larger batches might have slightly different entropy)
    batch_effect = (batch_size - 1) * 0.01  # Smaller effect
    synthetic_entropy += batch_effect
    
    # Clamp to range [0.0, 9.0]
    # For 16 layers at low temp, should be around 0-1.5
    # For 4-6 layers, should be around 7-9 max
    synthetic_entropy = np.clip(synthetic_entropy, 0.0, 9.0)
    
    return round(synthetic_entropy, 4)


def select_early_exit_layer(temperature=None, latency=None, early_exit_layers=[4, 6, 8, 10, 12, 14, 16],
                            temp_max=100.0, latency_max=20.0, base_probability=None):
    """
    Select early exit layer based on temperature and latency.
    
    Heuristics:
    - Higher temperature -> prefer fewer layers (early exit to reduce thermal load)
    - Higher latency -> prefer fewer layers (early exit to reduce latency)
    - Lower temperature/latency -> prefer more layers (better accuracy)
    
    Args:
        temperature: Current temperature in Celsius (optional)
        latency: Current latency in seconds (optional)
        early_exit_layers: List of available layers (default: [4, 6, 8, 10, 12, 14, 16])
        temp_max: Maximum temperature for normalization (default: 100.0)
        latency_max: Maximum latency for normalization (default: 20.0)
        base_probability: Base probability distribution (optional, uniform if None)
    
    Returns:
        Selected early exit layer
    """
    if base_probability is None:
        # Uniform base probability
        base_probability = np.ones(len(early_exit_layers)) / len(early_exit_layers)
    
    # Calculate pressure from temperature and latency
    # < 70°C: very low pressure (prefer 16 layers)
    # 70-80°C: smooth transition
    # > 80°C: high pressure (prefer 4 layers)
    temp_pressure = 0.0
    if temperature is not None and temp_max > 0:
        temp_low_threshold = 70.0  # 70°C
        temp_high_threshold = 80.0  # 80°C
        temp_high_ratio = temp_high_threshold / temp_max
        temp_normalized = np.clip(temperature / temp_max, 0.0, 1.0)
        
        if temperature < temp_low_threshold:
            # Low temperature (< 70°C): very low pressure (strongly prefer 16 layers)
            temp_pressure = (temperature / temp_low_threshold) ** 2.0 * 0.1  # Max 0.1 pressure at 70°C
        elif temperature < temp_high_threshold:
            # Transition zone (70-80°C): smooth transition from low to high pressure
            transition_scale = (temperature - temp_low_threshold) / (temp_high_threshold - temp_low_threshold)
            temp_pressure = 0.1 + transition_scale * 0.4  # 0.1 at 70°C, 0.5 at 80°C
        else:
            # High temperature (>= 80°C): high pressure (prefer fewer layers, especially 4)
            # Map [80°C, 100°C] to [0.0, 1.0] for exponential scaling
            high_temp_normalized = (temp_normalized - temp_high_ratio) / (1.0 - temp_high_ratio)
            temp_pressure = 0.5 + (high_temp_normalized ** 1.5 * 0.5)  # 0.5 at 80°C, 1.0 at 100°C
    
    latency_pressure = 0.0
    if latency is not None and latency_max > 0:
        latency_normalized = np.clip(latency / latency_max, 0.0, 1.0)
        latency_pressure = latency_normalized ** 1.5  # Exponential: stronger at high latency
    
    # Combined pressure: higher pressure -> prefer fewer layers
    combined_pressure = max(temp_pressure, latency_pressure)  # Use max for stronger effect
    
    # Adjust probabilities: higher pressure shifts weight to fewer layers
    # < 70°C: strongly prefer 16 layers (low pressure)
    # 70-80°C: smooth transition
    # > 80°C: strongly prefer 4 layers (high pressure)
    probabilities = base_probability.copy()
    
    # Calculate layer indices (0 = fewest layers, len-1 = most layers)
    for i, layer in enumerate(early_exit_layers):
        # Lower index = fewer layers = preferred under high pressure
        layer_index_normalized = 1.0 - (i / (len(early_exit_layers) - 1))  # 1.0 for 4 layers, 0.0 for 16 layers
        
        if temperature is not None and temp_max > 0:
            temp_low_threshold = 70.0
            temp_high_threshold = 80.0
            
            if temperature < temp_low_threshold:
                # Low temp: very strongly boost 16 layers, heavily penalize 4 layers
                if layer == 16:
                    probabilities[i] *= 20.0  # Very strong boost for 16 layers
                elif layer == 4:
                    probabilities[i] *= 0.001  # Very heavy penalty for 4 layers (almost zero)
                elif layer == 6:
                    probabilities[i] *= 0.01  # Heavy penalty for 6 layers
                elif layer == 8:
                    probabilities[i] *= 0.1  # Moderate penalty for 8 layers
                elif layer >= 12:
                    probabilities[i] *= 2.0  # Boost for 12+ layers
                else:
                    probabilities[i] *= 0.5  # Penalty for 10 layers
            elif temperature < temp_high_threshold:
                # Transition zone: smooth transition
                transition_scale = (temperature - temp_low_threshold) / (temp_high_threshold - temp_low_threshold)
                # Transition from preferring 16 layers to preferring 4 layers
                if layer == 16:
                    probabilities[i] *= (5.0 - transition_scale * 4.0)  # 5.0 at 70°C, 1.0 at 80°C
                elif layer == 4:
                    probabilities[i] *= (0.1 + transition_scale * 2.9)  # 0.1 at 70°C, 3.0 at 80°C
                else:
                    # Moderate transition for middle layers
                    boost = (1.0 - layer_index_normalized * 0.5) * (1.0 - transition_scale) + transition_scale * (1.0 + layer_index_normalized * 2.0)
                    probabilities[i] *= boost
            else:
                # High temp: strongly boost 4 layers, penalize 16 layers
                if layer == 4:
                    probabilities[i] *= 5.0  # Strong boost for 4 layers
                elif layer == 16:
                    probabilities[i] *= 0.1  # Strong penalty for 16 layers
                else:
                    probabilities[i] *= (1.0 + layer_index_normalized * 2.0)  # Moderate boost for fewer layers
        else:
            # Default: use pressure-based scaling
            pressure_boost = combined_pressure * layer_index_normalized * 2.0
            probabilities[i] *= (1.0 + pressure_boost)
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum()
    
    # Sample from distribution
    selected_layer = np.random.choice(early_exit_layers, p=probabilities)
    
    return selected_layer


def add_synthetic_data_to_csv(input_path, output_path, early_exit_layers=[4, 6, 8, 10, 12, 14, 16]):
    """
    Add synthetic entropy and early exit layer columns to CSV file.
    
    Strategy:
    1. Read existing CSV
    2. For each row, create multiple rows (one per early exit layer)
    3. Assign synthetic entropy based on early exit layer
    4. Save to new CSV
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        early_exit_layers: List of early exit layers to add
    """
    print("="*60)
    print("ADDING SYNTHETIC ENTROPY AND EARLY EXIT DATA")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Early exit layers: {early_exit_layers}")
    print("="*60)
    
    # Read existing CSV
    print("\nReading CSV file...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if entropy column already exists
    if 'avg_entropy' in df.columns:
        print("WARNING: 'avg_entropy' column already exists. Will overwrite with synthetic values.")
        df = df.drop(columns=['avg_entropy'])
    
    # Check if early_exit_layer column already exists
    if 'early_exit_layer' in df.columns:
        print("WARNING: 'early_exit_layer' column already exists. Will overwrite.")
        df = df.drop(columns=['early_exit_layer'])
    
    # Get batch size column (might be 'batch_size_config' or 'effective_batch_size')
    batch_col = None
    for col in ['batch_size_config', 'effective_batch_size', 'batch_size']:
        if col in df.columns:
            batch_col = col
            break
    
    if batch_col is None:
        print("ERROR: Could not find batch size column. Expected one of: batch_size_config, effective_batch_size, batch_size")
        return
    
    print(f"Using batch size column: {batch_col}")
    
    # Get temperature and latency columns (if available)
    temp_col = None
    for col in ['temperature', 'avg_temp', 'temp']:
        if col in df.columns:
            temp_col = col
            break
    
    latency_col = None
    for col in ['latency', 'end_to_end_latency', 'request_latency_sec', 'inference_time']:
        if col in df.columns:
            latency_col = col
            break
    
    if temp_col:
        print(f"Using temperature column: {temp_col}")
        temp_max = df[temp_col].max() if len(df) > 0 else 100.0
    else:
        print("WARNING: No temperature column found. Temperature effects will be disabled.")
        temp_max = 100.0
    
    if latency_col:
        print(f"Using latency column: {latency_col}")
        latency_max = df[latency_col].max() if len(df) > 0 else 20.0
    else:
        print("WARNING: No latency column found. Latency effects will be disabled.")
        latency_max = 20.0
    
    # Expand each row to include all early exit layers
    print(f"\nExpanding rows: {len(df)} rows × {len(early_exit_layers)} layers = {len(df) * len(early_exit_layers)} rows")
    
    expanded_rows = []
    for idx, row in df.iterrows():
        batch_size = int(row[batch_col])
        
        # Get temperature and latency for this row
        temperature = row[temp_col] if temp_col else None
        latency = row[latency_col] if latency_col else None
        
        for early_exit_layer in early_exit_layers:
            # Create new row with early exit layer
            new_row = row.copy()
            new_row['early_exit_layer'] = early_exit_layer
            
            # Generate synthetic entropy (skewed by temperature and latency)
            synthetic_entropy = generate_synthetic_entropy(
                early_exit_layer=early_exit_layer,
                batch_size=batch_size,
                temperature=temperature,
                latency=latency,
                num_layers=16,
                base_entropy=5.0,
                temp_max=temp_max,
                latency_max=latency_max
            )
            new_row['avg_entropy'] = synthetic_entropy
            
            expanded_rows.append(new_row)
    
    # Create new DataFrame
    df_expanded = pd.DataFrame(expanded_rows)
    print(f"Created {len(df_expanded)} rows")
    
    # Show entropy statistics
    print("\nSynthetic entropy statistics:")
    print(df_expanded.groupby('early_exit_layer')['avg_entropy'].agg(['mean', 'std', 'min', 'max']))
    
    # Save to output file
    print(f"\nSaving to: {output_path}")
    df_expanded.to_csv(output_path, index=False)
    print("Done!")
    print("="*60)
    
    return df_expanded


def update_synthetic_data_inplace(csv_path, early_exit_layers=[4, 6, 8, 10, 12, 14, 16]):
    """
    Update or create early_exit_layer and avg_entropy columns in CSV file in-place.
    
    If columns exist, updates them. If they don't exist, creates them.
    Uses temperature and latency-aware synthetic values.
    
    Args:
        csv_path: Path to CSV file to update (will be overwritten)
        early_exit_layers: List of early exit layers to choose from
    """
    print("="*60)
    print("UPDATING SYNTHETIC ENTROPY AND EARLY EXIT DATA (IN-PLACE)")
    print("="*60)
    print(f"CSV file: {csv_path}")
    print(f"Early exit layers: {early_exit_layers}")
    print("="*60)
    
    # Read existing CSV
    print("\nReading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if columns exist
    has_early_exit = 'early_exit_layer' in df.columns
    has_entropy = 'avg_entropy' in df.columns
    
    if has_early_exit and has_entropy:
        print("Found existing 'early_exit_layer' and 'avg_entropy' columns. Updating values...")
    elif has_early_exit:
        print("Found existing 'early_exit_layer' column. Creating 'avg_entropy' column...")
    elif has_entropy:
        print("Found existing 'avg_entropy' column. Creating 'early_exit_layer' column...")
    else:
        print("Columns 'early_exit_layer' and 'avg_entropy' not found. Creating new columns...")
    
    # Get batch size column
    batch_col = None
    for col in ['batch_size_config', 'effective_batch_size', 'batch_size']:
        if col in df.columns:
            batch_col = col
            break
    
    if batch_col is None:
        print("ERROR: Could not find batch size column.")
        return None
    
    print(f"Using batch size column: {batch_col}")
    
    # Get temperature and latency columns (if available)
    temp_col = None
    for col in ['temperature', 'avg_temp', 'temp']:
        if col in df.columns:
            temp_col = col
            break
    
    latency_col = None
    for col in ['latency', 'end_to_end_latency', 'request_latency_sec', 'inference_time']:
        if col in df.columns:
            latency_col = col
            break
    
    if temp_col:
        print(f"Using temperature column: {temp_col}")
        temp_max = df[temp_col].max() if len(df) > 0 else 100.0
        print(f"  Temperature range: {df[temp_col].min():.1f}°C to {temp_max:.1f}°C")
    else:
        print("WARNING: No temperature column found. Temperature effects will be disabled.")
        temp_max = 100.0
    
    if latency_col:
        print(f"Using latency column: {latency_col}")
        latency_max = df[latency_col].max() if len(df) > 0 else 20.0
        print(f"  Latency range: {df[latency_col].min():.3f}s to {latency_max:.3f}s")
    else:
        print("WARNING: No latency column found. Latency effects will be disabled.")
        latency_max = 20.0
    
    # Update early exit layer for each row (skewed by temperature and latency)
    np.random.seed(42)  # For reproducibility
    print("\nUpdating early_exit_layer values...")
    df['early_exit_layer'] = df.apply(
        lambda row: select_early_exit_layer(
            temperature=row[temp_col] if temp_col else None,
            latency=row[latency_col] if latency_col else None,
            early_exit_layers=early_exit_layers,
            temp_max=temp_max,
            latency_max=latency_max
        ),
        axis=1
    )
    
    # Update synthetic entropy for each row (skewed by temperature and latency)
    print("Updating avg_entropy values...")
    df['avg_entropy'] = df.apply(
        lambda row: generate_synthetic_entropy(
            early_exit_layer=row['early_exit_layer'],
            batch_size=int(row[batch_col]),
            temperature=row[temp_col] if temp_col else None,
            latency=row[latency_col] if latency_col else None,
            num_layers=16,
            base_entropy=5.0,
            temp_max=temp_max,
            latency_max=latency_max
        ),
        axis=1
    )
    
    # Show statistics
    print("\nUpdated synthetic entropy statistics:")
    print(df.groupby('early_exit_layer')['avg_entropy'].agg(['mean', 'std', 'min', 'max']))
    print(f"\nEarly exit layer distribution:")
    print(df['early_exit_layer'].value_counts().sort_index())
    
    # Show correlation with temperature and latency
    if temp_col:
        print(f"\nTemperature vs Early Exit Layer correlation:")
        temp_by_layer = df.groupby('early_exit_layer')[temp_col].mean()
        print(temp_by_layer)
        print(f"\nTemperature vs Entropy correlation:")
        temp_entropy_corr = df[[temp_col, 'avg_entropy']].corr().iloc[0, 1]
        print(f"  Correlation coefficient: {temp_entropy_corr:.4f}")
    
    if latency_col:
        print(f"\nLatency vs Early Exit Layer correlation:")
        latency_by_layer = df.groupby('early_exit_layer')[latency_col].mean()
        print(latency_by_layer)
        print(f"\nLatency vs Entropy correlation:")
        latency_entropy_corr = df[[latency_col, 'avg_entropy']].corr().iloc[0, 1]
        print(f"  Correlation coefficient: {latency_entropy_corr:.4f}")
    
    # Save back to same file (in-place update)
    print(f"\nSaving updated data to: {csv_path}")
    df.to_csv(csv_path, index=False)
    print("Done!")
    print("="*60)
    
    return df


def add_synthetic_data_simple(input_path, output_path, early_exit_layers=[4, 6, 8, 10, 12, 14, 16]):
    """
    Simpler version: Just add entropy to existing rows without expanding.
    
    This assigns a random early exit layer and corresponding entropy to each row.
    Use this if you want to keep the same number of rows.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        early_exit_layers: List of early exit layers to choose from
    """
    print("="*60)
    print("ADDING SYNTHETIC ENTROPY (SIMPLE MODE - NO EXPANSION)")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Early exit layers: {early_exit_layers}")
    print("="*60)
    
    # Read existing CSV
    print("\nReading CSV file...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Get batch size column
    batch_col = None
    for col in ['batch_size_config', 'effective_batch_size', 'batch_size']:
        if col in df.columns:
            batch_col = col
            break
    
    if batch_col is None:
        print("ERROR: Could not find batch size column.")
        return
    
    # Get temperature and latency columns (if available)
    temp_col = None
    for col in ['temperature', 'avg_temp', 'temp']:
        if col in df.columns:
            temp_col = col
            break
    
    latency_col = None
    for col in ['latency', 'end_to_end_latency', 'request_latency_sec', 'inference_time']:
        if col in df.columns:
            latency_col = col
            break
    
    if temp_col:
        print(f"Using temperature column: {temp_col}")
        temp_max = df[temp_col].max() if len(df) > 0 else 100.0
    else:
        print("WARNING: No temperature column found. Temperature effects will be disabled.")
        temp_max = 100.0
    
    if latency_col:
        print(f"Using latency column: {latency_col}")
        latency_max = df[latency_col].max() if len(df) > 0 else 20.0
    else:
        print("WARNING: No latency column found. Latency effects will be disabled.")
        latency_max = 20.0
    
    # Assign early exit layer to each row (skewed by temperature and latency)
    np.random.seed(42)  # For reproducibility
    df['early_exit_layer'] = df.apply(
        lambda row: select_early_exit_layer(
            temperature=row[temp_col] if temp_col else None,
            latency=row[latency_col] if latency_col else None,
            early_exit_layers=early_exit_layers,
            temp_max=temp_max,
            latency_max=latency_max
        ),
        axis=1
    )
    
    # Generate synthetic entropy for each row (skewed by temperature and latency)
    df['avg_entropy'] = df.apply(
        lambda row: generate_synthetic_entropy(
            early_exit_layer=row['early_exit_layer'],
            batch_size=int(row[batch_col]),
            temperature=row[temp_col] if temp_col else None,
            latency=row[latency_col] if latency_col else None,
            num_layers=16,
            base_entropy=5.0,
            temp_max=temp_max,
            latency_max=latency_max
        ),
        axis=1
    )
    
    # Show statistics
    print("\nSynthetic entropy statistics:")
    print(df.groupby('early_exit_layer')['avg_entropy'].agg(['mean', 'std', 'min', 'max']))
    print(f"\nEarly exit layer distribution:")
    print(df['early_exit_layer'].value_counts().sort_index())
    
    # Save to output file
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")
    print("="*60)
    
    return df


def update_data_folder_csvs(data_dir='data', early_exit_layers=[4, 6, 8, 10, 12, 14, 16]):
    """
    Automatically update all CSV files in the data directory in-place.
    
    Looks for CSV files with 'fan_on' and 'fan_off' in their names and updates
    the early_exit_layer and avg_entropy columns.
    
    Args:
        data_dir: Directory containing CSV files (default: 'data')
        early_exit_layers: List of early exit layers to use
    """
    print("="*60)
    print("UPDATING ALL CSV FILES IN DATA DIRECTORY")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Early exit layers: {early_exit_layers}")
    print("="*60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist.")
        return
    
    # Find CSV files with fan_on and fan_off
    csv_files = []
    for pattern in ['*fan_on*.csv', '*fan_off*.csv']:
        csv_files.extend(list(data_path.glob(pattern)))
    
    if not csv_files:
        print(f"WARNING: No CSV files found matching '*fan_on*.csv' or '*fan_off*.csv' in '{data_dir}'")
        print(f"Looking for any CSV files...")
        csv_files = list(data_path.glob('*.csv'))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in '{data_dir}'")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s) to update:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Update each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file.name}")
        print(f"{'='*60}")
        try:
            update_synthetic_data_inplace(
                csv_path=str(csv_file),
                early_exit_layers=early_exit_layers
            )
        except Exception as e:
            print(f"ERROR processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("ALL FILES UPDATED")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Add or update synthetic entropy and early exit layer data to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default behavior (no arguments):
  Automatically updates all CSV files in the 'data/' directory in-place.
  If columns exist, updates them. If not, creates them.

Examples:
  # Default: update all CSV files in data/ directory
  python add_synthetic_entropy_to_csv.py
  
  # Custom data directory
  python add_synthetic_entropy_to_csv.py --data-dir /path/to/data
  
  # Manual mode: process specific files
  python add_synthetic_entropy_to_csv.py --input file.csv --output output.csv
        """
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Input CSV file path (optional, for manual mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional, for manual mode)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory containing CSV files (default: data)')
    parser.add_argument('--early-exit-layers', type=str, default='4,6,8,10,12,14,16',
                       help='Comma-separated list of early exit layers (default: 4,6,8,10,12,14,16)')
    parser.add_argument('--expand', action='store_true',
                       help='Expand each row to include all early exit layers (creates more rows, manual mode only)')
    parser.add_argument('--simple', action='store_true',
                       help='Simple mode: assign random early exit layer to each row (keeps same number of rows, manual mode only)')
    
    args = parser.parse_args()
    
    # Parse early exit layers
    early_exit_layers = [int(x.strip()) for x in args.early_exit_layers.split(',')]
    
    # Default behavior: automatically update data directory
    if args.input is None and args.output is None:
        # Automatic mode: update all CSV files in data directory
        update_data_folder_csvs(
            data_dir=args.data_dir,
            early_exit_layers=early_exit_layers
        )
    elif args.input and args.output:
        # Manual mode: process specific files
        if args.expand:
            # Expand mode: create multiple rows per original row
            add_synthetic_data_to_csv(
                input_path=args.input,
                output_path=args.output,
                early_exit_layers=early_exit_layers
            )
        else:
            # Simple mode: just add columns to existing rows
            add_synthetic_data_simple(
                input_path=args.input,
                output_path=args.output,
                early_exit_layers=early_exit_layers
            )
    else:
        print("ERROR: If using manual mode, both --input and --output must be specified.")
        print("For automatic mode, just run the script without arguments.")
        parser.print_help()

