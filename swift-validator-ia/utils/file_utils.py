from pathlib import Path

def is_pdf(file_path: str) -> bool:
    return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]