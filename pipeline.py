"""Utility script for quick EDA and baseline submission generation.

The script performs two tasks:
1. Prints dataset insights (shape, missing counts, numeric stats).
2. Creates a simple baseline submission using the sample format.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


def is_missing(value: Optional[str]) -> bool:
    """Return True if the string value should be treated as missing."""
    if value is None:
        return True
    stripped = value.strip()
    return stripped == "" or stripped.lower() in {"na", "nan", "none"}


@dataclass
class StreamingStats:
    """Welford-style streaming statistics for numeric columns."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("nan")
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class DatasetSummary:
    columns: List[str]
    row_count: int
    missing_counts: Dict[str, int]
    numeric_stats: Dict[str, StreamingStats]
    preview_rows: List[Dict[str, str]]


def analyze_dataset(csv_path: str, preview_rows: int = 5) -> DatasetSummary:
    """Stream the CSV to compute shape, missing counts, and numeric stats."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        missing_counts = {col: 0 for col in columns}
        numeric_stats = {col: StreamingStats() for col in columns}
        numeric_flags: Dict[str, Optional[bool]] = {col: None for col in columns}
        preview: List[Dict[str, str]] = []
        row_count = 0

        for row in reader:
            row_count += 1
            if len(preview) < preview_rows:
                preview.append(row)

            for col, raw_value in row.items():
                if is_missing(raw_value):
                    missing_counts[col] += 1
                    continue

                if numeric_flags[col] is False:
                    continue

                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    numeric_flags[col] = False
                    numeric_stats.pop(col, None)
                    continue

                numeric_flags[col] = True
                numeric_stats[col].update(numeric_value)

    return DatasetSummary(
        columns=columns,
        row_count=row_count,
        missing_counts=missing_counts,
        numeric_stats=numeric_stats,
        preview_rows=preview,
    )


def print_summary(summary: DatasetSummary) -> None:
    """Print a readable summary of the dataset."""
    print("=== Dataset Shape ===")
    print(f"Rows: {summary.row_count}")
    print(f"Columns: {len(summary.columns)}")
    print("\n=== Columns ===")
    print(", ".join(summary.columns))

    print("\n=== Missing Values ===")
    for col in summary.columns:
        missing = summary.missing_counts.get(col, 0)
        pct = (missing / summary.row_count * 100) if summary.row_count else 0
        print(f"{col}: {missing} ({pct:.2f}% missing)")

    print("\n=== Numeric Columns Summary ===")
    for col in summary.columns:
        stats = summary.numeric_stats.get(col)
        if not stats:
            continue
        mean = stats.mean
        std = stats.std
        min_val = stats.minimum
        max_val = stats.maximum
        print(
            f"{col}: count={stats.count}, mean={mean:.4f}, std={std:.4f}, "
            f"min={min_val:.4f}, max={max_val:.4f}"
        )

    if summary.preview_rows:
        print("\n=== Preview (first few rows) ===")
        for idx, row in enumerate(summary.preview_rows, start=1):
            formatted = ", ".join(f"{k}={v}" for k, v in row.items())
            print(f"Row {idx}: {formatted}")


def compute_baseline_value(train_csv: str, target_column: str = "value") -> float:
    """Compute a simple baseline prediction (overall mean)."""
    stats = StreamingStats()
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_value = row.get(target_column)
            if is_missing(raw_value):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            stats.update(value)

    if stats.count == 0:
        raise ValueError(f"No numeric values found in column '{target_column}'.")
    return stats.mean


def create_submission(
    sample_submission_csv: str,
    output_csv: str,
    fill_value: float,
    target_column: str = "value",
) -> None:
    """Fill the sample submission with a constant prediction."""
    with open(sample_submission_csv, "r", newline="", encoding="utf-8") as src, \
            open(output_csv, "w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row[target_column] = str(fill_value)
            writer.writerow(row)



def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA and baseline generator")
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument(
        "--sample",
        default="sample_submission.csv",
        help="Path to sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        default="baseline_submission.csv",
        help="Where to save the baseline submission",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    summary = analyze_dataset(args.train)
    print_summary(summary)

    baseline_value = compute_baseline_value(args.train)
    print(f"\nBaseline prediction (mean of 'value'): {baseline_value:.4f}")

    create_submission(args.sample, args.output, baseline_value)
    print(f"Baseline submission saved to {args.output}")


if __name__ == "__main__":
    main()
