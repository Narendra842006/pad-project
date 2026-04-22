"""Split a Kaggle ECG CSV into one CSV file per row.

The website backend expects a single ECG beat per upload file. This utility
converts a dataset CSV into many per-row CSV files that you can upload one at
a time.

Usage:
    python split_ecg_dataset.py input.csv output_dir

Optional flags:
    --strip-label   Remove the last column when a row has 141 values.
                    This is useful when the dataset includes a label column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _is_numeric_row(values: list[str]) -> bool:
    try:
        for value in values:
            float(value)
        return True
    except ValueError:
        return False


def split_dataset(input_path: Path, output_dir: Path, strip_label: bool = False) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    with input_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)

        for row_index, row in enumerate(reader, start=1):
            values = [cell.strip() for cell in row if cell.strip()]
            if not values:
                continue

            if not _is_numeric_row(values):
                if row_index == 1:
                    continue
                raise ValueError(
                    f"Row {row_index} contains non-numeric values. "
                    "Check the input CSV format."
                )

            if strip_label and len(values) == 141:
                values = values[:140]

            output_file = output_dir / f"ecg_row_{created + 1:05d}.csv"
            with output_file.open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(values)

            created += 1

    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a Kaggle ECG dataset CSV into one file per row."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the dataset CSV")
    parser.add_argument("output_dir", type=Path, help="Directory for row CSV files")
    parser.add_argument(
        "--strip-label",
        action="store_true",
        help="Remove the last column when a row has 141 values",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input file not found: {args.input_csv}")

    created = split_dataset(args.input_csv, args.output_dir, args.strip_label)
    print(f"Created {created} ECG CSV file(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())