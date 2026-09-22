from pathlib import Path

def list_files(folder: str, recursive: bool = True, extension: str = None) -> list[str]:
    """
    Scan a folder and return all file paths as a list of strings.
 
    Args:
        folder:     Path to the folder to scan.
        recursive:  If True, scan all subfolders too.
        extension:  Filter by file extension e.g. ".pdf". None = all files.
 
    Returns:
        List of absolute file path strings.
    """
    folder_path = Path(folder)
 
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
 
    pattern = "**/*" if recursive else "*"
    all_paths = [p for p in folder_path.glob(pattern) if p.is_file()]
 
    if extension:
        ext = extension if extension.startswith(".") else f".{extension}"
        all_paths = [p for p in all_paths if p.suffix.lower() == ext.lower()]
 
    return [str(p.resolve()) for p in sorted(all_paths)]

if __name__ == "__main__":
    # Example usage:
    pdf_files = list_files("papers", recursive=False, extension=".pdf")
    print(f"Found {len(pdf_files)} PDF files:")
    for f in pdf_files:
        print(f.split('\\')[-1].replace(".pdf", "")[:30])