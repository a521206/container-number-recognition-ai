"""One-off script to rename container image files to their numeric identifiers.

Usage:
    python scripts/rename_files.py
"""

import re
import os
from pathlib import Path

directory = Path("data/Container Images1")

pattern = re.compile(r"\((\d+)\)")

for file in directory.iterdir():
    if not file.is_file():
        continue

    match = pattern.search(file.name)
    if match:
        number = match.group(1)
        ext = file.suffix
        new_name = f"{number}{ext}"
        new_path = file.parent / new_name

        if new_path.exists() and new_path != file:
            print(f"Skipping (exists): {file.name} -> {new_name}")
            continue

        print(f"{file.name} -> {new_name}")
        file.rename(new_path)
