"""Utility script for quick EDA and submission generation.

Features:
1. Prints dataset insights (shape, missing counts, numeric stats, unique counts).
2. Produces multiple baseline submissions (mean, last-value, trend, blended) without
   external dependencies so you can iterate quickly even in restricted
   environments.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


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
class ColumnUniqueSummary:
    unique_values: int
    example_values: List[str]


@dataclass
class DatasetSummary:
    columns: List[str]
    row_count: int
    missing_counts: Dict[str, int]
    numeric_stats: Dict[str, StreamingStats]
    unique_counts: Dict[str, ColumnUniqueSummary]
    preview_rows: List[Dict[str, str]]


def analyze_dataset(csv_path: str, preview_rows: int = 5) -> DatasetSummary:
    """Stream the CSV to compute shape, missing counts, numeric stats, and uniques."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        missing_counts = {col: 0 for col in columns}
        numeric_stats = {col: StreamingStats() for col in columns}
        numeric_flags: Dict[str, Optional[bool]] = {col: None for col in columns}
        preview: List[Dict[str, str]] = []
        row_count = 0
        uniques: Dict[str, set] = {col: set() for col in columns}

        for row in reader:
            row_count += 1
            if len(preview) < preview_rows:
                preview.append(row)

            for col, raw_value in row.items():
                if is_missing(raw_value):
                    missing_counts[col] += 1
                    continue

                if len(uniques[col]) < 10:
                    uniques[col].add(raw_value)
                # maintain uniques up to a reasonable bound to avoid memory blow-up
                if len(uniques[col]) > 10000:
                    uniques[col].clear()  # mark as too many by clearing

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

    unique_summary: Dict[str, ColumnUniqueSummary] = {}
    for col in columns:
        examples = sorted(list(uniques[col]))
        unique_summary[col] = ColumnUniqueSummary(
            unique_values=len(examples),
            example_values=examples[:5],
        )

    return DatasetSummary(
        columns=columns,
        row_count=row_count,
        missing_counts=missing_counts,
        numeric_stats=numeric_stats,
        unique_counts=unique_summary,
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

    print("\n=== Unique Values (approx) ===")
    for col in summary.columns:
        unique_info = summary.unique_counts[col]
        examples = ", ".join(unique_info.example_values)
        print(
            f"{col}: ~{unique_info.unique_values} uniques, examples: {examples}"
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


def month_index(year: str, month: str) -> int:
    return int(year) * 12 + (int(month) - 1)


def load_item_series(train_csv: str, target_column: str = "value") -> Dict[str, List[Tuple[int, float]]]:
    """Return per-item ordered time series (month_index, value)."""
    series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_missing(row.get(target_column)):
                continue
            try:
                ts = month_index(row["year"], row["month"])
                val = float(row[target_column])
            except (Exception,):
                continue
            series[row["item_id"]].append((ts, val))

    for item_id, values in series.items():
        values.sort(key=lambda x: x[0])
    return series


def linear_trend_forecast(values: List[Tuple[int, float]]) -> Optional[float]:
    """Simple OLS trend extrapolation to the next step."""
    if len(values) < 2:
        return None
    xs = [idx for idx, _ in enumerate(values)]
    ys = [v for _, v in values]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    next_x = xs[-1] + 1
    return intercept + slope * next_x


def forecast_item(series: List[Tuple[int, float]], method: str, blend_weight: float) -> float:
    """Predict next value using the requested strategy."""
    values = [v for _, v in series]
    last_val = values[-1]
    mean_val = sum(values) / len(values)
    if method == "mean":
        return mean_val
    if method == "last":
        return last_val
    if method == "trend":
        trend = linear_trend_forecast(series)
        return trend if trend is not None else last_val
    if method == "blend":
        trend = linear_trend_forecast(series)
        if trend is None:
            return last_val
        return blend_weight * trend + (1 - blend_weight) * last_val
    raise ValueError(f"Unknown method: {method}")


def build_prediction_map(
    train_csv: str,
    method: str,
    blend_weight: float,
    target_column: str = "value",
) -> Tuple[Dict[str, float], float]:
    """Create per-item predictions and return global fallback mean."""
    item_series = load_item_series(train_csv, target_column)
    fallback_mean = compute_baseline_value(train_csv, target_column)
    predictions: Dict[str, float] = {}
    for item_id, series in item_series.items():
        if not series:
            continue
        predictions[item_id] = forecast_item(series, method, blend_weight)
    return predictions, fallback_mean


def create_submission(
    sample_submission_csv: str,
    output_csv: str,
    predictions: Dict[str, float],
    fallback_mean: float,
    target_column: str = "value",
    following_column: str = "following_item_id",
) -> None:
    """Fill the sample submission with item-level predictions."""
    with open(sample_submission_csv, "r", newline="", encoding="utf-8") as src, open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            key = row.get(following_column)
            pred = predictions.get(key, fallback_mean)
            row[target_column] = f"{pred:.4f}"
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
        "--output-prefix",
        default="submission",
        help="Filename prefix for generated submissions",
    )
    parser.add_argument(
        "--strategies",
        default="mean,last,trend,blend",
        help="Comma-separated strategies: mean,last,trend,blend",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.7,
        help="Weight for trend in blend strategy (0-1)",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of preview rows to show in EDA",
    )
    return parser.parse_args(argv)


def run_eda(train_path: str, preview_rows: int) -> None:
    summary = analyze_dataset(train_path, preview_rows=preview_rows)
    print_summary(summary)


def generate_submissions(args: argparse.Namespace) -> List[str]:
    outputs: List[str] = []
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for strategy in strategies:
        predictions, fallback = build_prediction_map(
            args.train, method=strategy, blend_weight=args.blend_weight
        )
        output_path = f"{args.output_prefix}_{strategy}.csv"
        create_submission(args.sample, output_path, predictions, fallback)
        outputs.append(output_path)
        print(
            f"Saved {strategy} submission to {output_path} "
            f"(fallback mean={fallback:.4f})"
        )
    return outputs


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_eda(args.train, preview_rows=args.preview_rows)
    print("\n=== Generating submissions ===")
    generate_submissions(args)


if __name__ == "__main__":
    main()
