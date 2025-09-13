from tkinter import filedialog
import pyxdf
import matplotlib
matplotlib.use('TkAgg')
import os

from excel_output import ExcelOutput
from plot_output import plot_data

# Load XDF
xdf_path = filedialog.askopenfilename(
    title="Select XDF file",
    filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
)
if not xdf_path:
    raise FileNotFoundError("No file was selected.")

streams, header = pyxdf.load_xdf(xdf_path)
file_name = os.path.basename(xdf_path)
print(f'Reading file {file_name}')

# Identify relevant streams
pupil_streams = [s for s in streams if "pupil" in s['info']['name'][0].lower()]
marker_streams = [s for s in streams if "marker" in s['info']['name'][0].lower() or "event" in s['info']['name'][0].lower()]

print("Found Streams:")
for s in streams:
    print(f"- {s['info']['name'][0]}")


# Create excel
results_dir = os.path.join(os.getcwd(), "VR Processing Results")
os.makedirs(results_dir, exist_ok=True)

excel_output = ExcelOutput(file_name, marker_streams, pupil_streams)
excel_output.create_excel_file(results_dir)

plot_data(marker_streams, pupil_streams, excel_output.get_chosen_pupil(), results_dir, excel_output.subject_id, excel_output.eeg_part)

