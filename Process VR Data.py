from tkinter import filedialog
import pyxdf
import matplotlib
matplotlib.use('TkAgg')
import os
from excel_output import ExcelOutput
from plot_output import plot_data

print("-------------------------- Running Process VR Data --------------------------")

# Load XDF files
xdf_paths = filedialog.askopenfilenames(
    title="Select XDF files",
    filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
)
if not xdf_paths:
    raise FileNotFoundError("No files were selected.")

for xdf_path in xdf_paths:
    streams, header = pyxdf.load_xdf(xdf_path)
    file_name = os.path.basename(xdf_path)
    print(f"Reading file {file_name}")

    pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
    marker_streams = [s for s in streams if
                      "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

    print("Found Streams:")
    for s in streams:
        print(f"- {s['info']['name'][0]}")

    excel_output = ExcelOutput(file_name, marker_streams, pupil_streams)

    results_dir = os.path.join(os.getcwd(), "VR Processing Results")
    os.makedirs(results_dir, exist_ok=True)

    date_time = excel_output.get_date_time()

    sub_dir = os.path.join(results_dir, f"{date_time}")
    os.makedirs(results_dir, exist_ok=True)

    excel_output.create_excel_file(sub_dir)

    subject_id, eeg_part = excel_output.parse_filename(file_name)
    plot_data(marker_streams, pupil_streams, excel_output.get_chosen_pupil(), sub_dir, subject_id, eeg_part)






