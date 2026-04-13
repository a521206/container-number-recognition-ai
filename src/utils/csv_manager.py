"""CSV results management for extraction outputs."""

import csv
import os
from datetime import datetime, timezone
from typing import Dict

from .config import DATA_DIR
from ..processing.extraction import ContainerResult

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")

RESULT_COLUMNS = [
    "file_name",
    "container_number",
    "container_type",
    "container_color",
    "valid",
    "reason",
    "tare_weight_kg",
    "tare_weight_lbs",
    "payload_weight_kg",
    "payload_weight_lbs",
    "max_gross_weight_kg",
    "max_gross_weight_lbs",
    "owner_operator_name",
    "owner_operator_location",
    "bounding_box",
    "timestamp",
]


def get_or_create_results_csv() -> Dict[str, Dict[str, str]]:
    """Load existing results.csv or create empty dict if not exists."""
    results = {}
    if not os.path.exists(RESULTS_CSV):
        return results

    with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name")
            if file_name:
                results[file_name] = row

    return results


def save_results_csv(results: Dict[str, Dict[str, str]]) -> None:
    """Save results dict to CSV with proper columns."""
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for file_name, row in sorted(results.items()):
            row["file_name"] = file_name
            writer.writerow(row)


def _container_result_to_dict(result: ContainerResult) -> Dict[str, str]:
    """Convert ContainerResult to flat dict for CSV storage."""
    d: Dict[str, str] = {}
    
    d["file_name"] = result.file_name or ""
    d["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if not result:
        return d
    
    # Always preserve extracted data, even if validation failed
    d["container_number"] = result.container_number or result.raw_container_number or ""
    d["container_type"] = result.container_type or result.raw_container_type or ""
    d["valid"] = str(result.valid) if result.valid is not None else ""
    d["reason"] = result.reason or ""
    
    if result.error:
        d["container_number"] = f"error: {result.error}"
    
    d["container_color"] = str(result.container_color) if result.container_color and result.container_color != [0, 0, 0] else ""
    d["bounding_box"] = str(result.bounding_box) if result.bounding_box and result.bounding_box != [0, 0, 0, 0] else ""
    
    if result.weights:
        d["tare_weight_kg"] = str(result.weights.tare_weight.kilograms) if result.weights.tare_weight.kilograms else ""
        d["tare_weight_lbs"] = str(result.weights.tare_weight.pounds) if result.weights.tare_weight.pounds else ""
        d["payload_weight_kg"] = str(result.weights.payload_weight.kilograms) if result.weights.payload_weight.kilograms else ""
        d["payload_weight_lbs"] = str(result.weights.payload_weight.pounds) if result.weights.payload_weight.pounds else ""
        d["max_gross_weight_kg"] = str(result.weights.maximum_gross_weight.kilograms) if result.weights.maximum_gross_weight.kilograms else ""
        d["max_gross_weight_lbs"] = str(result.weights.maximum_gross_weight.pounds) if result.weights.maximum_gross_weight.pounds else ""
    
    if result.owner_operator:
        d["owner_operator_name"] = result.owner_operator.name or ""
        d["owner_operator_location"] = result.owner_operator.location or ""
    
    return d


def merge_results(
    existing_results: Dict[str, Dict[str, str]],
    result: ContainerResult,
) -> Dict[str, Dict[str, str]]:
    """Merge ContainerResult into existing results."""
    file_name = result.file_name
    if not file_name:
        return existing_results

    if file_name not in existing_results:
        existing_results[file_name] = {col: "" for col in RESULT_COLUMNS}

    result_dict = _container_result_to_dict(result)
    for key, value in result_dict.items():
        if key in RESULT_COLUMNS:
            existing_results[file_name][key] = value

    return existing_results


class ResultsCSVManager:
    """Manages results.csv with efficient batch updates."""

    def __init__(self):
        self.results: Dict[str, Dict[str, str]] = {}
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.results = get_or_create_results_csv()
            self._loaded = True

    def update(self, result: ContainerResult) -> None:
        """Merge result without I/O."""
        self.ensure_loaded()
        merge_results(self.results, result)

    def save(self) -> None:
        """Write to disk."""
        self.ensure_loaded()
        save_results_csv(self.results)


_manager = ResultsCSVManager()


def update_results(result: ContainerResult) -> None:
    """Update results.csv with ContainerResult."""
    _manager.update(result)


def save_results() -> None:
    """Write pending changes to disk (call after batch processing)."""
    _manager.save()