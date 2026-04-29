"""Command-line interface for container detection."""

import argparse
import json
import logging
import os

from ..utils.csv_manager import update_results, save_results
from ..core.combined_pipeline import run_combined_extraction
from ..utils.config import DATA_DIR as CONFIG_DATA_DIR

log = logging.getLogger(__name__)


def main():
    """Main CLI function - uses combined OCR + Llama Extract pipeline."""
    parser = argparse.ArgumentParser(description="Container number recognition from images.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Image file paths to process, or a directory containing images. "
             "If a directory is provided, all images in it will be processed."
    )
    parser.add_argument(
        "-d", "--data-dir",
        default=CONFIG_DATA_DIR,
        help="Data directory to read images from (default: ./data). "
             "Results CSV will be written to this directory."
    )
    
    args = parser.parse_args()

    # Dynamically update the module-level DATA_DIR in config
    import src.utils.config as config
    config.DATA_DIR = os.path.abspath(args.data_dir)

    # Check if specific file paths were provided
    file_paths = [p for p in args.paths if os.path.isfile(p)]

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
    elif args.paths:
        # Check if any path is a directory
        dir_paths = [p for p in args.paths if os.path.isdir(p)]
        if dir_paths:
            # Process all directories provided
            for dir_path in dir_paths:
                _process_directory(dir_path)
            save_results()
        elif not dir_paths:
            log.error("No valid files or directories found in: %s", args.paths)
            return
    else:
        # Use the data-dir argument
        if not os.path.exists(args.data_dir):
            log.error("Data directory '%s' not found", args.data_dir)
            return
        _process_directory(args.data_dir)
        save_results()


def _process_directory(dir_path):
    """Process all images in a directory."""
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
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


if __name__ == "__main__":
    main()