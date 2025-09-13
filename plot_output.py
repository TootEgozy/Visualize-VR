import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import os

class PlotOutput:
    def __init__(self, filename, marker_streams, pupil_streams, chosen_pupil):
        self.marker_streams = marker_streams
        self.pupil_streams = pupil_streams
        self.chosen_pupil = chosen_pupil
        self.marker_labels = marker_streams[0]['time_series']
        self.marker_times = np.array(marker_streams[0]['time_stamps'])
        self.pupil_data = np.array(pupil_streams[0]['time_series'])
        self.pupil_time = np.array(pupil_streams[0]['time_stamps'])

        if marker_streams:
            start_time = self.pupil_time[0]
            self.pupil_time -= start_time
            self.marker_times -= start_time
        else:
            self.marker_labels = []
            self.marker_times = []

        # Max/Min pupil points
        self.max_idx = np.argmax(chosen_pupil)
        self.min_idx = np.argmin(chosen_pupil)
        self.max_point = (self.pupil_time[max_idx], chosen_pupil[max_idx])
        self.min_point = (self.pupil_time[min_idx], chosen_pupil[min_idx])

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

results_dir = os.path.join(os.getcwd(), "VR Processing Results")
os.makedirs(results_dir, exist_ok=True)