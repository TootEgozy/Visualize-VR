from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import os


def compile_filename(subject_id, eeg_part):
    now = datetime.now()
    date_str = now.strftime("%d%m%Y")
    time_str = now.strftime("%H%M%S")
    return f"Graph-{subject_id}-{eeg_part}-{date_str}-{time_str}"

def plot_data(marker_streams, pupil_streams, chosen_pupil, results_dir_path, subject_id, eeg_part):
    pupil_time = np.array(pupil_streams[0]['time_stamps'])

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

    filename = compile_filename(subject_id, eeg_part)
    plot_path = os.path.join(results_dir_path, filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
