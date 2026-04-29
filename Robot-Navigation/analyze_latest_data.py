"""
Analyze latest Webots evaluation data and generate updated graphs.
Session: 20251219_035321
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load latest evaluation data
eval_folder = Path("semantic_vln_mcp/webots/controllers/semantic_vln_controller/evaluation_data/20251219_035321")
output_dir = Path("paper_graphs")
output_dir.mkdir(exist_ok=True)

print(f"Analyzing data from: {eval_folder}")

# Load all logs
with open(eval_folder / "tsg_logs.json") as f:
    tsg_data = json.load(f)

with open(eval_folder / "latency_logs.json") as f:
    latency_data = json.load(f)

with open(eval_folder / "slam_logs.json") as f:
    slam_data = json.load(f)

print(f"TSG entries: {len(tsg_data)}")
print(f"Latency entries: {len(latency_data)}")
print(f"SLAM entries: {len(slam_data)}")

# ============ 1. TSG ANALYSIS ============
print("\n" + "="*50)
print("TSG ANALYSIS")
print("="*50)

# Group by object class
objects_by_class = defaultdict(list)
for entry in tsg_data:
    objects_by_class[entry['object_class']].append(entry)

print(f"\nObjects tracked: {list(objects_by_class.keys())}")

for obj_class, entries in objects_by_class.items():
    visible = sum(1 for e in entries if e['is_visible'])
    occluded = sum(1 for e in entries if not e['is_visible'])
    avg_occl_time = np.mean([e['occlusion_time'] for e in entries if not e['is_visible']] or [0])
    print(f"  {obj_class}: {visible} visible, {occluded} occluded, avg occlusion: {avg_occl_time:.2f}s")

# ============ 2. TSG CONFIDENCE DECAY GRAPH (REAL DATA) ============
fig, ax = plt.subplots(figsize=(10, 6))

colors = {'refrigerator': '#4CAF50', 'person': '#E53935', 'bed': '#42A5F5'}
markers = {'refrigerator': 'o', 'person': 's', 'bed': '^'}

for obj_class in ['refrigerator', 'person', 'bed']:
    if obj_class not in objects_by_class:
        continue
    entries = objects_by_class[obj_class]
    occl_times = [e['occlusion_time'] for e in entries if e['occlusion_time'] > 0]
    confidences = [e['confidence'] * 100 for e in entries if e['occlusion_time'] > 0]
    
    if occl_times:
        ax.scatter(occl_times, confidences, label=f'{obj_class.title()}', 
                   alpha=0.7, c=colors.get(obj_class, 'gray'), marker=markers.get(obj_class, 'o'))

# Add theoretical decay lines
time_points = np.linspace(0, 5, 100)
decay_rates = {'Person (λ=0.5)': 0.5, 'Refrigerator (λ=0.1)': 0.1, 'Bed (λ=0.05)': 0.05}
for label, rate in decay_rates.items():
    ax.plot(time_points, 100 * np.exp(-rate * time_points), '--', alpha=0.5, label=f'Theory: {label}')

ax.axhline(y=20, color='gray', linestyle=':', label='Query Threshold (20%)')
ax.set_xlabel('Occlusion Time (seconds)', fontsize=12)
ax.set_ylabel('Confidence (%)', fontsize=12)
ax.set_title('TSG Confidence Decay - Real Data from Webots Session', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(output_dir / 'tsg_decay_real_data.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✓ Generated: tsg_decay_real_data.png")

# ============ 3. LATENCY BREAKDOWN ============
# Calculate average latencies (excluding outliers)
perception_times = [e['perception_ms'] for e in latency_data if e['perception_ms'] > 0]
slam_times = [e['slam_ms'] for e in latency_data if e['slam_ms'] < 300]  # Exclude spikes
gesture_times = [e['gesture_ms'] for e in latency_data if e['gesture_ms'] < 50]
planning_times = [e['planning_ms'] for e in latency_data if e['planning_ms'] > 0]
tsg_times = [e['tsg_ms'] for e in latency_data if e['tsg_ms'] > 0]

avg_latencies = {
    'Perception\n(YOLOv8)': np.mean(perception_times) if perception_times else 0,
    'SLAM\n(DeepLabV3+)': np.mean(slam_times) if slam_times else 0,
    'Gesture\n(MediaPipe)': np.mean(gesture_times) if gesture_times else 0,
    'Planning\n(A*)': np.mean(planning_times) if planning_times else 0,
    'TSG': np.mean(tsg_times) if tsg_times else 0
}

print("\nAverage Latencies:")
for comp, lat in avg_latencies.items():
    print(f"  {comp.replace(chr(10), ' ')}: {lat:.1f}ms")

fig, ax = plt.subplots(figsize=(10, 6))
components = list(avg_latencies.keys())
latencies = list(avg_latencies.values())
colors_lat = ['#FF7043', '#66BB6A', '#AB47BC', '#42A5F5', '#FFD54F']

bars = ax.barh(components, latencies, color=colors_lat)
ax.set_xlabel('Latency (ms)', fontsize=12)
ax.set_title('System Latency Breakdown - Real Webots Data', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(latencies) * 1.2)

for bar, val in zip(bars, latencies):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}ms', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'latency_real_data.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Generated: latency_real_data.png")

# ============ 4. SLAM MAP GROWTH ============
fig, ax = plt.subplots(figsize=(10, 6))
times = [e['timestamp'] for e in slam_data]
occupied = [e['occupied_cells'] for e in slam_data]
free = [e['free_cells'] for e in slam_data]

ax.plot(times, occupied, label='Occupied Cells', color='#E53935', linewidth=2)
ax.plot(times, free, label='Free Cells', color='#4CAF50', linewidth=2)
ax.set_xlabel('Simulation Time (seconds)', fontsize=12)
ax.set_ylabel('Number of Cells', fontsize=12)
ax.set_title('SLAM Occupancy Grid Growth Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'slam_growth_real_data.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Generated: slam_growth_real_data.png")

# ============ 5. MOTION PREDICTION ANALYSIS ============
print("\n" + "="*50)
print("MOTION PREDICTION ANALYSIS")
print("="*50)

# Check if person locations are changing over time (indicating motion prediction)
person_entries = objects_by_class.get('person', [])
if person_entries:
    # Group by approximate location to track same object
    location_changes = []
    prev_loc = None
    for e in sorted(person_entries, key=lambda x: x['timestamp']):
        loc = tuple(e['location'])
        # Filter out pixel coordinate errors
        if loc[0] < 100 and loc[1] < 100:  # Valid world coords
            if prev_loc is not None:
                dist = np.sqrt((loc[0] - prev_loc[0])**2 + (loc[1] - prev_loc[1])**2)
                if dist < 2:  # Same object, tracking over time
                    location_changes.append({
                        'time': e['timestamp'],
                        'loc': loc,
                        'prev_loc': prev_loc,
                        'distance': dist,
                        'occlusion': e['occlusion_time']
                    })
            prev_loc = loc
    
    print(f"\nPerson location changes tracked: {len(location_changes)}")
    if location_changes:
        print("Sample motion predictions:")
        for change in location_changes[:5]:
            print(f"  t={change['time']:.1f}s: moved {change['distance']:.3f}m during {change['occlusion']:.2f}s occlusion")

# ============ 6. WHY NAV_LOGS IS EMPTY ============
print("\n" + "="*50)
print("WHY NAV_LOGS IS EMPTY?")
print("="*50)
print("""
The nav_logs.json is empty because:
1. NO NAVIGATION COMMAND was issued during this session
2. The robot was moving (SLAM shows position changes)
3. But no "find the person" or "go to kitchen" command was given

To populate nav_logs:
1. Run the command_interface.py
2. Type: "find the person" or "go to kitchen"
3. The navigation will be logged
""")

# ============ 7. COMPARISON TABLE WITH REAL DATA ============
print("\n" + "="*50)
print("UPDATED COMPARISON TABLE")
print("="*50)

# Calculate real metrics
visible_count = sum(1 for e in tsg_data if e['is_visible'])
occluded_count = sum(1 for e in tsg_data if not e['is_visible'])
avg_visible_conf = np.mean([e['confidence'] for e in tsg_data if e['is_visible']]) * 100 if visible_count > 0 else 0
avg_2s_conf = np.mean([e['confidence'] for e in tsg_data if 1.5 < e['occlusion_time'] < 2.5]) * 100
avg_5s_conf = np.mean([e['confidence'] for e in tsg_data if e['occlusion_time'] > 4.5]) * 100

comparison = {
    "Metric": ["Objects Tracked", "Visible Detections", "Occluded Tracking", 
               "Avg Confidence (Visible)", "Avg Confidence (2s Hidden)", "Avg Confidence (5s Hidden)",
               "Total Latency (avg)"],
    "Value": [
        len(objects_by_class),
        visible_count,
        occluded_count,
        f"{avg_visible_conf:.1f}%",
        f"{avg_2s_conf:.1f}%",
        f"{avg_5s_conf:.1f}%",
        f"{np.mean([e['total_ms'] for e in latency_data]):.0f}ms"
    ]
}

print("\nReal Data Metrics:")
for m, v in zip(comparison["Metric"], comparison["Value"]):
    print(f"  {m}: {v}")

# Save as JSON
with open(output_dir / 'real_data_metrics.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("\n" + "="*50)
print("✓ All graphs generated in:", output_dir.absolute())
print("="*50)
