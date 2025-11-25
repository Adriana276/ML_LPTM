"""
Visualize synthetic data to verify patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load one sequence
df = pd.read_csv('./dataset/synthetic_sequence_1.csv')

print(f"Loaded {len(df)} samples")
print(f"Duration: {len(df)/50:.1f} seconds at 50 Hz")
print("\nActivity distribution:")
print(df['label'].value_counts())

# Create time axis
time = np.arange(len(df)) / 50  # Convert to seconds

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot accelerometer data
ax = axes[0]
ax.plot(time, df['acc_x'], label='Acc X', alpha=0.7)
ax.plot(time, df['acc_y'], label='Acc Y', alpha=0.7)
ax.plot(time, df['acc_z'], label='Acc Z', alpha=0.7)
ax.set_ylabel('Acceleration (g)', fontsize=12)
ax.set_title('Accelerometer Data - Synthetic Sequence', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot gyroscope data
ax = axes[1]
ax.plot(time, df['gyro_x'], label='Gyro X', alpha=0.7)
ax.plot(time, df['gyro_y'], label='Gyro Y', alpha=0.7)
ax.plot(time, df['gyro_z'], label='Gyro Z', alpha=0.7)
ax.set_ylabel('Angular Velocity (deg/s)', fontsize=12)
ax.set_title('Gyroscope Data - Synthetic Sequence', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot activity labels
ax = axes[2]
label_mapping = {'WAL': 0, 'JUM': 1, 'FALL': 2, 'LYI': 3}
numeric_labels = [label_mapping[l] for l in df['label']]

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']  # Green, Blue, Red, Orange
activity_names = ['Walking', 'Jumping', 'Falling', 'Lying']

ax.fill_between(time, 0, numeric_labels, step='mid', alpha=0.6, color='gray')
ax.set_ylabel('Activity', fontsize=12)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_title('Activity Labels Over Time', fontsize=14, fontweight='bold')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(activity_names)
ax.grid(True, alpha=0.3, axis='x')

# Add activity regions as colored backgrounds
current_activity = df['label'].iloc[0]
start_idx = 0

for i, label in enumerate(df['label']):
    if label != current_activity or i == len(df) - 1:
        end_idx = i
        color_idx = label_mapping[current_activity]
        ax.axvspan(time[start_idx], time[end_idx], 
                  alpha=0.2, color=colors[color_idx])
        
        # Add text label in the middle of each section
        mid_time = (time[start_idx] + time[end_idx]) / 2
        ax.text(mid_time, label_mapping[current_activity], 
               current_activity, 
               ha='center', va='center', 
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        current_activity = label
        start_idx = i

plt.tight_layout()
plt.savefig('./results/synthetic_data_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to ./results/synthetic_data_visualization.png")
plt.close()

# Create a second figure showing magnitude
fig, ax = plt.subplots(figsize=(14, 5))

# Calculate acceleration magnitude
acc_magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

ax.plot(time, acc_magnitude, color='#2c3e50', linewidth=1.5, label='Acceleration Magnitude')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Magnitude (g)', fontsize=12)
ax.set_title('Total Acceleration Magnitude Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Add colored background for activities
current_activity = df['label'].iloc[0]
start_idx = 0

for i, label in enumerate(df['label']):
    if label != current_activity or i == len(df) - 1:
        end_idx = i
        color_idx = label_mapping[current_activity]
        ax.axvspan(time[start_idx], time[end_idx], 
                  alpha=0.15, color=colors[color_idx],
                  label=f'{current_activity}' if current_activity not in [df['label'].iloc[j] for j in range(i)] else None)
        current_activity = label
        start_idx = i

plt.tight_layout()
plt.savefig('./results/acceleration_magnitude.png', dpi=300, bbox_inches='tight')
print("Magnitude plot saved to ./results/acceleration_magnitude.png")

print("\nDone! Check the results folder for visualizations.")
