"""Command-line interface for container detection."""

import json
import logging
import os
import sys

from ..utils.csv_manager import update_results, save_results
from ..core.combined_pipeline import run_combined_extraction

log = logging.getLogger(__name__)


def main():
    """Main CLI function - uses combined OCR + Llama Extract pipeline."""
    file_paths = [a for a in sys.argv[1:] if os.path.isfile(a)]

    if file_paths:
        for filepath in file_paths:
            file_name = os.path.basename(filepath)
            result, _ = run_combined_extraction(filepath)
            result.file_name = file_name
            update_results(result)
            if result.error or not result.container_number:
                log.warning("Skipped %s: %s", filepath, result.error or "No container detected")
            else:
                print(json.dumps(result.to_dict(), indent=2))
        save_results()
    elif not os.path.exists("./data"):
        log.error("Data directory './data' not found")
        return
    else:
        for filename in os.listdir("./data"):
            filepath = os.path.join("./data", filename)
            if not os.path.isfile(filepath):
                continue
            if not filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                continue
            
            result, _ = run_combined_extraction(filepath)
            result.file_name = filename
            update_results(result)
            
            if result.error or not result.container_number:
                log.warning("Skipped %s: %s", filename, result.error or "No container detected")
            else:
                print(json.dumps(result.to_dict(), indent=2))
        save_results()


if __name__ == "__main__":
    main()