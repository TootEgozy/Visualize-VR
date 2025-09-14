import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial
import mplcursors

def process_waveform(t, y, first_dur=2.0, poly_deg=27,
                     sg_polyorder=2, sg_halfwidth=4):
    idx_first = t < first_dur
    idx_second = ~idx_first

    # poly fit first part
    if idx_first.sum() >= (poly_deg + 1):
        p = Polynomial.fit(t[idx_first], y[idx_first], deg=poly_deg).convert()
        y_first = p(t[idx_first])
    else:
        y_first = y[idx_first]

    # savgol second part
    window = 2 * sg_halfwidth + 1
    if window >= idx_second.sum():
        L = idx_second.sum()
        window = L if L % 2 == 1 else L - 1
    if idx_second.sum() and window >= 3:
        y_second = savgol_filter(y[idx_second], window_length=window,
                                 polyorder=sg_polyorder)
    else:
        y_second = y[idx_second]

    y_proc = y.copy()
    y_proc[idx_first] = y_first
    y_proc[idx_second] = y_second
    return y_proc


def plot_data(marker_streams, pupil_streams, chosen_pupil):
    pupil_time = np.array(pupil_streams[0]['time_stamps'])

    if marker_streams:
        marker_labels = marker_streams[0]['time_series']
        marker_times = np.array(marker_streams[0]['time_stamps'])
        start_time = pupil_time[0]
        pupil_time -= start_time
        marker_times -= start_time
    else:
        marker_labels, marker_times = [], []

    # processed version
    chosen_pupil = np.array(chosen_pupil)
    pupil_proc = process_waveform(pupil_time, chosen_pupil)

    # ---- plotting two subplots ----
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # ---------- RAW ----------
    pupil_line_raw, = axes[0].plot(pupil_time, chosen_pupil,
                                   color='navy', label="Raw Pupil")
    marker_lines_raw = []
    for i, t in enumerate(marker_times):
        label = marker_labels[i][0] if isinstance(marker_labels[i], list) else marker_labels[i]
        line = axes[0].axvline(x=t, linestyle='--', color='gray', alpha=0.4)
        marker_lines_raw.append((line, label))
    axes[0].set_title("Pupil Size (Raw)")
    axes[0].set_ylabel("Pupil Size")
    axes[0].set_ylim(-1, 4)   # << fixed y-axis range
    axes[0].grid(True)
    axes[0].legend(loc='upper right', fontsize='small')

    # ---------- PROCESSED ----------
    pupil_line_proc, = axes[1].plot(pupil_time, pupil_proc,
                                    color='darkgreen', label="Processed Pupil")
    marker_lines_proc = []
    for i, t in enumerate(marker_times):
        label = marker_labels[i][0] if isinstance(marker_labels[i], list) else marker_labels[i]
        line = axes[1].axvline(x=t, linestyle='--', color='gray', alpha=0.4)
        marker_lines_proc.append((line, label))
    axes[1].set_title("Pupil Size (Processed)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pupil Size")
    axes[1].set_ylim(-1, 4)   # << fixed y-axis range
    axes[1].grid(True)
    axes[1].legend(loc='upper right', fontsize='small')

    plt.tight_layout()

    # ---------- Cursors ----------
    # raw marker cursors
    cursor_raw = mplcursors.cursor([line for line, _ in marker_lines_raw], hover=True)
    @cursor_raw.connect("add")
    def on_add_raw(sel):
        idx = [line for line, _ in marker_lines_raw].index(sel.artist)
        sel.annotation.set_text(marker_lines_raw[idx][1])
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    cursor_pupil_raw = mplcursors.cursor(pupil_line_raw, hover=True)
    @cursor_pupil_raw.connect("add")
    def on_add_pupil_raw(sel):
        x, y = sel.target
        sel.annotation.set_text(f"t={x:.3f}, size={y:.3f}")
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)

    # processed marker cursors
    cursor_proc = mplcursors.cursor([line for line, _ in marker_lines_proc], hover=True)
    @cursor_proc.connect("add")
    def on_add_proc(sel):
        idx = [line for line, _ in marker_lines_proc].index(sel.artist)
        sel.annotation.set_text(marker_lines_proc[idx][1])
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    cursor_pupil_proc = mplcursors.cursor(pupil_line_proc, hover=True)
    @cursor_pupil_proc.connect("add")
    def on_add_pupil_proc(sel):
        x, y = sel.target
        sel.annotation.set_text(f"t={x:.3f}, size={y:.3f}")
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)

    # save + show
    plt.show()
    plt.close()
