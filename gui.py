import easygui
import os

def ask_if_adding_to_existing_excel():
    """
    Interactive dialog for Excel output configuration.

    Returns:
        add_to_existing (bool): True if user chose to add to an existing file.
        file_path (str | None): Full path if add_to_existing is True, else None.
        file_name (str | None): Filename if add_to_existing is True, else None.
    """
    file_path = None
    file_name = None

    while True:
        # Step 1: Yes/No
        choice = easygui.ynbox("Do you want to add to an existing file?", "Excel Options", ["Yes", "No"])

        if not choice:  # User picked "No" or closed
            return False, None, None

        # Step 2: File selection or Go Back
        buttons = ["Browse File", "Go Back"]
        choice = easygui.buttonbox("Select an existing Excel file or go back:", "Select File", choices=buttons)

        if choice == "Go Back":
            continue  # restart at Step 1

        if choice == "Browse File":
            file_path = easygui.fileopenbox("Select Excel file", filetypes=["*.xlsx"])
            if file_path and os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                return True, file_path, file_name
            else:
                easygui.msgbox("Invalid file path, please try again.")
