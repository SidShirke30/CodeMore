from pathlib import Path
import csv


REQUIRED_COLUMNS = {
    "timestamp",
    "model_version",
    "prediction",
    "actual_label",
}


def validate_data(path):
    path = Path(path)

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Dataset is empty")

    missing = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    valid_rows = [
        row for row in rows
        if row["actual_label"] != ""
    ]

    return valid_rows


def version_dataset(rows, version="dataset-v1"):
    return {
        "version": version,
        "rows": rows,
        "count": len(rows),
    }
