from openpyxl.styles import Border, Side

header_border = Border(
    left=Side(border_style="thin", color="808080"),
    right=Side(border_style="thin", color="808080"),
    top=Side(border_style="thin", color="808080"),
    bottom=Side(border_style="thin", color="808080")
)

right_border = Border(right=Side(border_style="thin", color="808080"))

bottom_border = Border(bottom=Side(border_style="thin", color="808080"))
