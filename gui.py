import tkinter as tk
from tkinter import filedialog

def run_gui(default_filename="output.xlsx"):
    result = {
        "xdf_paths": [],
        "output_folder_path": "",
        "output_filename": default_filename
    }

    def select_files():
        paths = filedialog.askopenfilenames(
            title="Select input files",
            filetypes=[("XDF files", "*.xdf"), ("All files", "*.*")]
        )
        if paths:
            result["xdf_paths"] = list(paths)
            files_var.set(", ".join(paths))

    def select_output_folder():
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            output_folder_var.set(path)

    def submit():
        result["output_folder_path"] = output_folder_var.get()
        result["output_filename"] = outfile_var.get()
        root.destroy()

    root = tk.Tk()
    root.title("Excel File Setup")

    default_font = ("TkDefaultFont", 12)
    pad_opts = {"padx": 10, "pady": 5}

    # ---- Input files ----
    tk.Label(root, text="Input files:", font=default_font).grid(row=0, column=0, sticky="w", **pad_opts)
    files_var = tk.StringVar()
    tk.Entry(root, textvariable=files_var, width=60, font=default_font).grid(row=0, column=1, **pad_opts)
    tk.Button(root, text="Browse", command=select_files, font=default_font).grid(row=0, column=2, **pad_opts)

    # ---- Output folder ----
    tk.Label(root, text="Output folder:", font=default_font).grid(row=1, column=0, sticky="w", **pad_opts)
    output_folder_var = tk.StringVar()
    tk.Entry(root, textvariable=output_folder_var, width=60, font=default_font).grid(row=1, column=1, **pad_opts)
    tk.Button(root, text="Browse", command=select_output_folder, font=default_font).grid(row=1, column=2, **pad_opts)

    # ---- Output filename ----
    tk.Label(root, text="Output filename:", font=default_font).grid(row=2, column=0, sticky="w", **pad_opts)
    outfile_var = tk.StringVar(value=default_filename)
    tk.Entry(root, textvariable=outfile_var, width=60, font=default_font).grid(row=2, column=1, **pad_opts)

    # ---- Submit ----
    tk.Button(root, text="OK", command=submit, font=default_font).grid(row=3, column=1, **pad_opts)

    # Center window
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 3) - (h // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return result
