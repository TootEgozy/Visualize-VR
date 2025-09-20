import pyxdf
import matplotlib
from openpyxl import Workbook
from gui import run_gui
matplotlib.use('TkAgg')
import os
from excel_output import ExcelOutput
from plot_output import plot_data

print("-------------------------- Running Process VR Data --------------------------")

excel_output = ExcelOutput()
OUTPUT_FOLDER_NAME = "VR Processing Results"
default_output_filename = excel_output.compile_filename()

gui_data = run_gui(default_output_filename)
xdf_paths = gui_data["xdf_paths"]
output_folder_path = gui_data["output_folder_path"]
output_filename = gui_data["output_filename"]

results_dir = None
if os.path.basename(output_folder_path) == OUTPUT_FOLDER_NAME:
    results_dir = output_folder_path
else:
    results_dir = os.path.join(output_folder_path, OUTPUT_FOLDER_NAME)
os.makedirs(results_dir, exist_ok=True)

if not xdf_paths:
    raise FileNotFoundError("No files were selected.")

new_file_path = os.path.join(results_dir, output_filename)
wb = Workbook()
wb.save(new_file_path)

for index, xdf_path in enumerate(xdf_paths):
    streams, header = pyxdf.load_xdf(xdf_path)
    stream_filename = os.path.basename(xdf_path)
    print(f"Reading file {stream_filename}")

    pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
    marker_streams = [s for s in streams if
                      "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

    print("Found Streams:")
    for s in streams:
        print(f"- {s['info']['name'][0]}")


    excel_output.write_to_excel(marker_streams, pupil_streams, results_dir, output_filename, stream_filename, index)

    subject_id, eeg_part = excel_output.parse_filename(stream_filename)
    chosen_pupil = excel_output.select_valid_pupil(pupil_streams[0]['time_series'])
    # plot_data(marker_streams, pupil_streams, chosen_pupil, sub_dir, subject_id, eeg_part)






