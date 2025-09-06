from tkinter import filedialog
from datetime import datetime
import pyxdf
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import os
from openpyxl import Workbook

digits_fixed = 8 # how many digits to include in the floats
now = datetime.now()
timestamp_excel = now.strftime("260825-%H%M")
timestamp_plot = now.strftime("260825-%H%M")
excel_filename = f"data-{timestamp_excel}.xlsx"
plot_filename = f"graph-{timestamp_plot}.png"

# ----------------- Helper functions to restructure and clean data ---------------

def clean_event_name(event_name):
    s = event_name.lower()
    s = re.sub(r'\b(start|stop)\b', '', s)
    s = re.sub(r'for\s+\d+(\.\d+)?\s+seconds', '', s)
    s = ' '.join(s.split())
    s = s.rstrip('.')
    return s

def get_vector_from_name(event_name):
    match = re.search(r'\(([^)]+)\)', event_name)
    if match:
        return f"({match.group(1).strip()})"
    return None

def get_luminescence_from_name(event_name):
    return event_name.split(" in (")[0].strip()

def calc_event_duration(start_ts, stop_ts):
    duration = stop_ts - start_ts
    return f"{duration:.2f}"

def build_event_dict(stream):
    events = stream['time_series']
    timestamps = stream['time_stamps']
    event_dict = {}

    for i in range(len(events)):
        event = events[i]
        ts = timestamps[i]
        event_name = clean_event_name(event[0])
        event_vector = get_vector_from_name(event_name)
        event_luminescence = get_luminescence_from_name(event_name)
        last_event_ts = timestamps[-1]
        fixed_digits_ts = round(ts, digits_fixed)
        fixed_digits_next_event_ts = round(timestamps[i + 1] if i + 1 < len(timestamps) else last_event_ts, digits_fixed)

        if event_name not in event_dict:
            event_dict[event_name] = {
                "start": fixed_digits_ts,
                "stop": last_event_ts,
                "vector": event_vector,
                "luminescence": event_luminescence
            }
        else:
            event_dict[event_name]["stop"] = fixed_digits_ts
            event_dict[event_name]["next_event_start"] = fixed_digits_next_event_ts
            event_dict[event_name]["duration"] = calc_event_duration(event_dict[event_name].get("start"), fixed_digits_ts)

    return event_dict


def build_pupil_dict(stream, pupil_stream):
    timestamps = stream['time_stamps']
    pupil_dict = {}

    for time_stamp, pupil_size in zip(timestamps, pupil_stream):
        fixed_time_stamp = round(time_stamp, digits_fixed)
        pupil_dict[fixed_time_stamp] = pupil_size

    return pupil_dict



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


def get_closest_timestamp(timestamps_list, target_ts):
    idx = min(range(len(timestamps_list)), key=lambda i: abs(timestamps_list[i] - target_ts))
    return round(timestamps_list[idx], digits_fixed), idx


def get_timestamps_before_event(timestamps_list, pupil_map, event_ts, count=3, sampling_gap = 14):
    samples = []
    event_exact_ts, event_i = get_closest_timestamp(timestamps_list, event_ts)
    jump = 0.001 * sampling_gap
    items_left = count
    i = 1

    while items_left > 0:
        jump = jump * i
        i += 1
        ts, ts_i = get_closest_timestamp(timestamps_list, event_exact_ts - jump)

        pupil_size = pupil_map.get(ts)

        if pupil_size > 2.5: # valid sample, not a blink
            samples.append(pupil_size)
            items_left -= 1
        else: # a blink
            if i > 100:
                raise ValueError(f"Cannot find valid timestamps before the event which starts at {event_ts}")
    return samples

def get_average_size_before_event(pupil_dict_keys, pupil_dict, event_start):
    samples_before_start = get_timestamps_before_event(pupil_dict_keys, pupil_dict, event_start)
    average_pupil_size_before = sum(samples_before_start) / len(samples_before_start)
    return average_pupil_size_before


def get_average_size_after_event(pupil_dict_keys, pupil_dict, event_start, event_end, count=3):
    samples = [1000] * count
    start_ts, start_i = get_closest_timestamp(pupil_dict_keys, event_start)
    end_ts, end_i = get_closest_timestamp(pupil_dict_keys, event_end)

    if event_start == event_end:
        return pupil_dict.get(get_closest_timestamp(pupil_dict_keys, event_start))

    searching_range = pupil_dict_keys[start_i: end_i + 1]

    for ts in searching_range:
        pupil_size = pupil_dict.get(ts)
        if max(samples) > pupil_size > 2.5:
            idx = samples.index(max(samples))
            samples[idx] = pupil_size

    return sum(samples) / len(samples)


# --------------------------------------------------------------------------------


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

# --------------------------------- Compute Data for Excel -----------------------------------

# Collect data
event_dict = build_event_dict(marker_streams[0])
pupil_dict = build_pupil_dict(pupil_streams[0], chosen_pupil)

pupil_timestamps = list(pupil_dict.keys())

for event in event_dict:
    size_before = get_average_size_before_event(pupil_timestamps, pupil_dict, event_dict[event].get("start"))
    event_dict[event]["average_size_before"] = size_before

    event_end = round(event_dict[event].get("stop", marker_streams[0]["time_stamps"][-1]), digits_fixed)
    next_start = round(event_dict[event].get("next_event_start", marker_streams[0]["time_stamps"][-1]), digits_fixed)

    size_after = get_average_size_after_event(pupil_timestamps, pupil_dict, event_end, next_start)
    event_dict[event]["average_size_after"] = size_after

# Create excel
results_dir = os.path.join(os.getcwd(), "VR Processing Results")
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, excel_filename)


data = []
for event_name, info in event_dict.items():
    data.append([
        info.get('vector'),
        info.get('luminescence'),
        info.get('average_size_before'),
        info.get('average_size_after')
    ])

wb = Workbook()
ws = wb.active

headers = ["Vector", "Luminescence", "Avg Before", "Min Avg After"]

# merge headers
col = 1
for i, header in enumerate(headers):
    span = 3 if i == 1 else 2
    ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + span - 1)
    ws.cell(row=1, column=col, value=header)
    col += span

# fill data
for r, row in enumerate(data, 2):
    col = 1
    for i, value in enumerate(row):
        span = 3 if i == 1 else 2
        ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + span - 1)
        ws.cell(row=r, column=col, value=value)
        col += span
# Save
wb.save(output_path)

print(f"Excel file saved to: {output_path}")

# ------------------------------------ Plot ----------------------------------------
plt.figure(figsize=(16, 6))
pupil_line, = plt.plot(pupil_time, chosen_pupil, label="Chosen Pupil", color='navy')

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

cursor_pupil = mplcursors.cursor(pupil_line, hover=True)

@cursor_pupil.connect("add")
def on_add_pupil(sel):
    x, y = sel.target
    sel.annotation.set_text(f"t={x:.3f}, size={y:.3f}")
    sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)

plt.xlabel("Time (s)")
plt.ylabel("Pupil Size")
plt.title("Pupil Size Over Time with Event Markers")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(results_dir, plot_filename)
plt.savefig(plot_path, dpi=300)
plt.show()
plt.close()