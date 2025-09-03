from tkinter import filedialog
import pyxdf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

# Load the XDF file
# xdf_path = ""

xdf_path = filedialog.askopenfilename(
    title="Select XDF file",
    filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
)
if not xdf_path:
    raise FileNotFoundError("No file was selected.")

# streams: A list of dicts, each representing a different recorded data stream.
# header: Metadata about the recording session.
streams, header = pyxdf.load_xdf(xdf_path)

# Identify streams: one with pupil data, one with markers
pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
marker_streams = [s for s in streams if
                  "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

# Print stream names for verification
print("Found Streams:")
for s in streams:
    print(f"- {s['info']['name'][0]}")

# Extract pupil data
pupil_data = pupil_streams[0]['time_series']
pupil_time = pupil_streams[0]['time_stamps']
pupil_data = np.array(pupil_data)

# Determine if data is left and right pupil
if pupil_data.shape[1] >= 2:
    left_pupil = pupil_data[:, 0]
    right_pupil = pupil_data[:, 1]
else:
    left_pupil = pupil_data[:, 0]
    right_pupil = None

# Extract markers
# Gets the marker event labels and timestamps.
# Aligns the timestamps to start from 0 (relative time from experiment start).
marker_lines = []
if marker_streams:
    marker_labels = marker_streams[0]['time_series']
    marker_times = marker_streams[0]['time_stamps']

    # Convert to relative time (seconds since experiment started)
    start_time = pupil_time[0]
    pupil_time = pupil_time - start_time
    marker_times = marker_times - start_time

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(pupil_time, left_pupil, label="Left Pupil", color='navy')
if right_pupil is not None:
    plt.plot(pupil_time, right_pupil, label="Right Pupil", color='teal')

# Add vertical lines without labeling
for i, t in enumerate(marker_times):
    label = marker_labels[i][0] if isinstance(marker_labels[i], list) else marker_labels[i]
    line = plt.axvline(x=t, linestyle='--', color='gray', alpha=0.4)
    marker_lines.append((line, label))  # Save line and label for mplcursors

# Interactive cursor to show marker labels on hover
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
plt.show()