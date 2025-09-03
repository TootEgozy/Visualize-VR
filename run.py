from tkinter import filedialog
from datetime import datetime
import pyxdf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import pandas as pd
import os

now = datetime.now()
timestamp_excel = now.strftime("260825-%H%M")
timestamp_plot = now.strftime("260825-%H%M")
excel_filename = f"data-{timestamp_excel}.xlsx"
plot_filename = f"graph-{timestamp_plot}.png"


def select_valid_pupil(pupil_data):
    if pupil_data.shape[1] >= 2:
        left = pupil_data[:, 0]
        right = pupil_data[:, 1]
        if not np.all(left == -1):
            return left
        elif not np.all(right == -1):
            return right
        else:
            raise ValueError("Both pupils are invalid.")
    else:
        pupil = pupil_data[:, 0]
        if np.all(pupil == -1):
            raise ValueError("Pupil data is invalid.")
        return pupil


def average_before_event(pupil_time, pupil_data, event_time, samples=3, interval_ms=1, offset_ms=3):
    interval = interval_ms / 1000
    offset = offset_ms / 1000
    sample_times = [event_time - offset - i * interval for i in range(samples)]
    values = []
    for t in sample_times:
        idx = np.searchsorted(pupil_time, t)
        if 0 <= idx < len(pupil_data):
            val = pupil_data[idx]
            if val >= 0:
                values.append(val)
    return np.mean(values) if values else float('nan')


def average_min_after_event(pupil_time, pupil_data, start_time, end_time, min_samples=3):
    mask = (pupil_time >= start_time) & (pupil_time <= end_time)
    data = pupil_data[mask]
    data = data[data >= 0]
    if len(data) == 0:
        return float('nan')
    n = min(min_samples, len(data))
    min_vals = np.partition(data, n - 1)[:n]
    return np.mean(min_vals)


# Load XDF
xdf_path = filedialog.askopenfilename(
    title="Select XDF file",
    filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
)
if not xdf_path:
    raise FileNotFoundError("No file was selected.")

streams, header = pyxdf.load_xdf(xdf_path)

# Identify relevant streams
pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
marker_streams = [s for s in streams if "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

print("Found Streams:")
for s in streams:
    print(f"- {s['info']['name'][0]}")

pupil_data = np.array(pupil_streams[0]['time_series'])
pupil_time = np.array(pupil_streams[0]['time_stamps'])

chosen_pupil = select_valid_pupil(pupil_data)

if marker_streams:
    marker_labels = marker_streams[0]['time_series']
    marker_times = np.array(marker_streams[0]['time_stamps'])

    start_time = pupil_time[0]
    pupil_time -= start_time
    marker_times -= start_time
else:
    marker_labels = []
    marker_times = []

# Max/Min pupil points
max_idx = np.argmax(chosen_pupil)
min_idx = np.argmin(chosen_pupil)
max_point = (pupil_time[max_idx], chosen_pupil[max_idx])
min_point = (pupil_time[min_idx], chosen_pupil[min_idx])

# --- Build Event Infos (pairing Start/Stop) ---
event_infos = []
i = 0
while i < len(marker_labels) - 1:
    start_label = marker_labels[i][0] if isinstance(marker_labels[i], list) else marker_labels[i]
    end_label = marker_labels[i + 1][0] if isinstance(marker_labels[i + 1], list) else marker_labels[i + 1]

    if start_label.startswith("Start") and end_label.startswith("Stop"):
        event_infos.append({
            "label": start_label.replace("Start ", "").strip(),
            "start": marker_times[i],
            "end": marker_times[i + 1]
        })
        i += 2  # Skip to next event pair
    else:
        i += 1  # Skip unmatched or malformed events

# --- Compute Data for Excel ---
event_labels_with_times = []
pupil_before_events = []
min_pupil_after_events = []

for i, event in enumerate(event_infos):
    label = event["label"]
    start = event["start"]
    end = event["end"]

    event_labels_with_times.append(f"{label}\nStart: {start:.3f}s\nEnd: {end:.3f}s")

    avg_before = average_before_event(pupil_time, chosen_pupil, start)
    pupil_before_events.append(avg_before)

    next_start = event_infos[i + 1]["start"] if i + 1 < len(event_infos) else pupil_time[-1]
    avg_min_after = average_min_after_event(pupil_time, chosen_pupil, end, next_start)
    min_pupil_after_events.append(avg_min_after)

# --- Excel Output ---
excel_data = pd.DataFrame({
    "Event Label + Start/End": event_labels_with_times,
    "Avg Pupil Size Before Event": pupil_before_events,
    "Avg 3 Min Pupil Sizes After Event": min_pupil_after_events
})

extremes_data = pd.DataFrame({
    "Type": ["Maximum", "Minimum"],
    "Timestamp (s)": [max_point[0], min_point[0]],
    "Pupil Size": [max_point[1], min_point[1]]
})

results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, excel_filename)

with pd.ExcelWriter(output_path) as writer:
    excel_data.to_excel(writer, index=False, sheet_name="Events")
    extremes_data.to_excel(writer, index=False, sheet_name="Pupil Extremes")

print(f"Excel file saved to: {output_path}")

# --- Plot ---
plt.figure(figsize=(16, 6))
plt.plot(pupil_time, chosen_pupil, label="Chosen Pupil", color='navy')

marker_lines = []
for i, t in enumerate(marker_times):
    label = marker_labels[i][0] if isinstance(marker_labels[i], list) else marker_labels[i]
    line = plt.axvline(x=t, linestyle='--', color='gray', alpha=0.4)
    marker_lines.append((line, label))

cursor = mplcursors.cursor([line for line, _ in marker_lines], hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = [line for line, _ in marker_lines].index(sel.artist)
    sel.annotation.set_text(marker_lines[idx][1])
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.xlabel("Time (s)")
plt.ylabel("Pupil Size")
plt.title("Pupil Size Over Time with Event Markers")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(results_dir, plot_filename)
plt.savefig(plot_path, dpi=300)
plt.close()