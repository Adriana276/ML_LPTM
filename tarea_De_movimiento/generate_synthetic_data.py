"""
Generate synthetic accelerometer and gyroscope data for HAR
Simulates: Walking → Jumping → Falling → Lying → Walking
"""

import numpy as np
import pandas as pd
import os
from config import *

def generate_walking_data(duration, sampling_rate=50):
    """Generate walking pattern with periodic acceleration"""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Walking has periodic vertical and forward motion
    step_freq = 2  # Hz (2 steps per second)
    
    acc_x = 0.3 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.1, n_samples)  # Forward
    acc_y = 1.0 + 0.5 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.15, n_samples)  # Vertical (gravity + motion)
    acc_z = np.random.normal(0, 0.1, n_samples)  # Side to side
    
    gyro_x = np.random.normal(0, 5, n_samples)  # Slight rotation
    gyro_y = 10 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 3, n_samples)
    gyro_z = np.random.normal(0, 3, n_samples)
    
    labels = ['WAL'] * n_samples
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def generate_running_data(duration, sampling_rate=50):
    """Generate running pattern - faster and more intense than walking"""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Running has higher frequency and amplitude than walking
    step_freq = 3.5  # Hz (3.5 steps per second - faster than walking)
    
    acc_x = 0.7 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.2, n_samples)  # Forward momentum
    acc_y = 1.0 + 1.2 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.25, n_samples)  # Vertical (stronger than walking)
    acc_z = 0.3 * np.sin(2 * np.pi * step_freq * t + np.pi/4) + np.random.normal(0, 0.15, n_samples)  # Side to side
    
    gyro_x = np.random.normal(0, 10, n_samples)  # More rotation than walking
    gyro_y = 25 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 5, n_samples)
    gyro_z = np.random.normal(0, 5, n_samples)
    
    labels = ['RUN'] * n_samples
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def generate_jumping_data(duration, sampling_rate=50):
    """Generate jumping pattern with strong vertical acceleration"""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Jumping has strong vertical acceleration and brief airtime
    jump_freq = 1.5  # Hz (1.5 jumps per second)
    
    acc_x = np.random.normal(0, 0.2, n_samples)
    acc_y = 1.0 + 2.0 * np.abs(np.sin(2 * np.pi * jump_freq * t)) + np.random.normal(0, 0.3, n_samples)  # Strong vertical
    acc_z = np.random.normal(0, 0.15, n_samples)
    
    gyro_x = np.random.normal(0, 10, n_samples)
    gyro_y = 20 * np.sin(2 * np.pi * jump_freq * t) + np.random.normal(0, 5, n_samples)  # Strong rotation on landing
    gyro_z = np.random.normal(0, 8, n_samples)
    
    labels = ['JUM'] * n_samples
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def generate_falling_data(duration, sampling_rate=50):
    """Generate falling pattern with sudden change in all axes"""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Fall has sudden acceleration spike followed by impact
    fall_progress = t / duration  # 0 to 1
    
    acc_x = 2.0 * fall_progress + np.random.normal(0, 0.5, n_samples)  # Forward momentum
    acc_y = -9.8 * fall_progress + np.random.normal(0, 0.5, n_samples)  # Downward acceleration
    acc_z = 1.5 * fall_progress + np.random.normal(0, 0.4, n_samples)
    
    # High rotation during fall
    gyro_x = 100 * fall_progress + np.random.normal(0, 20, n_samples)
    gyro_y = 80 * fall_progress + np.random.normal(0, 15, n_samples)
    gyro_z = 60 * fall_progress + np.random.normal(0, 10, n_samples)
    
    labels = ['FALL'] * n_samples
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def generate_lying_data(duration, sampling_rate=50):
    """Generate lying pattern - minimal movement"""
    n_samples = int(duration * sampling_rate)
    
    # Lying down: mostly just gravity in one direction, minimal movement
    acc_x = np.random.normal(0, 0.05, n_samples)
    acc_y = np.random.normal(0, 0.05, n_samples)  # No vertical component when lying
    acc_z = np.random.normal(1.0, 0.05, n_samples)  # Gravity pointing sideways
    
    gyro_x = np.random.normal(0, 1, n_samples)
    gyro_y = np.random.normal(0, 1, n_samples)
    gyro_z = np.random.normal(0, 1, n_samples)
    
    labels = ['LYI'] * n_samples
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def generate_sequence():
    """
    Generate a complete sequence:
    Walking (5s) → Running (3s) → Jumping (2s) → Falling (0.5s) → Lying (3s) → Walking (5s)
    """
    sequences = []
    
    # Walking phase 1
    print("Generating walking phase 1...")
    seq1 = generate_walking_data(WALKING_DURATION)
    
    # Running phase
    print("Generating running phase...")
    seq2 = generate_running_data(RUNNING_DURATION)
    
    # Jumping phase
    print("Generating jumping phase...")
    seq3 = generate_jumping_data(JUMPING_DURATION)
    
    # Falling phase
    print("Generating falling phase...")
    seq4 = generate_falling_data(FALLING_DURATION)
    
    # Lying phase
    print("Generating lying phase...")
    seq5 = generate_lying_data(LYING_DURATION)
    
    # Walking phase 2 (recovery)
    print("Generating walking phase 2 (recovery)...")
    seq6 = generate_walking_data(WALKING_DURATION)
    
    # Combine all sequences
    all_seqs = [seq1, seq2, seq3, seq4, seq5, seq6]
    
    acc_x = np.concatenate([s[0] for s in all_seqs])
    acc_y = np.concatenate([s[1] for s in all_seqs])
    acc_z = np.concatenate([s[2] for s in all_seqs])
    gyro_x = np.concatenate([s[3] for s in all_seqs])
    gyro_y = np.concatenate([s[4] for s in all_seqs])
    gyro_z = np.concatenate([s[5] for s in all_seqs])
    labels = np.concatenate([s[6] for s in all_seqs])
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def save_to_csv(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels, filename):
    """Save generated data to CSV in the format expected by training code"""
    df = pd.DataFrame({
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'label': labels
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} samples to {filename}")
    return df

if __name__ == "__main__":
    # Create dataset directory
    os.makedirs('./dataset', exist_ok=True)
    
    print("=" * 50)
    print("Generating Synthetic HAR Dataset")
    print("=" * 50)
    
    # Generate multiple sequences for training
    num_sequences = 5
    all_data = []
    
    for i in range(num_sequences):
        print(f"\nGenerating sequence {i+1}/{num_sequences}...")
        data = generate_sequence()
        df = save_to_csv(*data, f'./dataset/synthetic_sequence_{i+1}.csv')
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('./dataset/all_synthetic_data.csv', index=False)
    
    print("\n" + "=" * 50)
    print("Dataset Generation Complete!")
    print("=" * 50)
    print(f"Total samples: {len(combined_df)}")
    print(f"Duration: ~{len(combined_df) / SAMPLING_RATE:.1f} seconds")
    print("\nActivity distribution:")
    print(combined_df['label'].value_counts())
    print("\nFiles saved in ./dataset/")
