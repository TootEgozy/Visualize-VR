from tkinter import filedialog
import pyxdf
import matplotlib
from openpyxl import Workbook
from gui import ask_if_adding_to_existing_excel
matplotlib.use('TkAgg')
import os
from excel_output import ExcelOutput
from plot_output import plot_data
import easygui

print("-------------------------- Running Process VR Data --------------------------")

excel_output = ExcelOutput()

results_dir = os.path.join(os.getcwd(), "VR Processing Results")
os.makedirs(results_dir, exist_ok=True)

# Load XDF files
xdf_paths = filedialog.askopenfilenames(
    title="Select XDF files",
    filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
)
if not xdf_paths:
    raise FileNotFoundError("No files were selected.")

add_to_existing, output_filepath, output_filename = ask_if_adding_to_existing_excel()

if not add_to_existing: # create a new file
    output_filename = excel_output.compile_filename()
    wb = Workbook()
    wb.save(output_filename)

for xdf_path in xdf_paths:
    streams, header = pyxdf.load_xdf(xdf_path)
    stream_filename = os.path.basename(xdf_path)
    print(f"Reading file {stream_filename}")

    pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
    marker_streams = [s for s in streams if
                      "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

    print("Found Streams:")
    for s in streams:
        print(f"- {s['info']['name'][0]}")


    excel_output.write_to_excel(marker_streams, pupil_streams, results_dir, add_to_existing, output_filename, stream_filename)

    subject_id, eeg_part = excel_output.parse_filename(stream_filename)
    chosen_pupil = excel_output.select_valid_pupil(pupil_streams[0]['time_series'])
    # plot_data(marker_streams, pupil_streams, chosen_pupil, sub_dir, subject_id, eeg_part)






