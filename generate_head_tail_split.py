import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate head.txt and tail.txt from train-item usage."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names under --data-root, direct dataset paths, or 'all'.",
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent / "datasets"),
        help="Root directory that contains dataset folders.",
    )
    parser.add_argument(
        "--usage-threshold",
        type=float,
        default=0.80,
        help="Cumulative usage threshold for the head set, e.g. 0.80.",
    )
    parser.add_argument(
        "--train-file",
        default="train.txt",
        help="Training interaction filename inside each dataset directory.",
    )
    parser.add_argument(
        "--test-file",
        default="test.txt",
        help="Optional test interaction filename used to infer the full item range.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split summary without writing head.txt and tail.txt.",
    )
    parser.add_argument(
        "--write-summary-json",
        action="store_true",
        help="Also write head_tail_summary.json into each dataset directory.",
    )
    return parser.parse_args()


def resolve_dataset_dirs(dataset_args, data_root):
    resolved = []
    seen = set()

    def add_dataset_dir(dataset_dir):
        dataset_dir = dataset_dir.resolve()
        if dataset_dir in seen:
            return
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        seen.add(dataset_dir)
        resolved.append(dataset_dir)

    for dataset_arg in dataset_args:
        if dataset_arg.lower() == "all":
            for child in sorted(data_root.iterdir()):
                if child.is_dir() and (child / "train.txt").exists():
                    add_dataset_dir(child)
            continue

        dataset_path = Path(dataset_arg)
        if dataset_path.exists():
            add_dataset_dir(dataset_path)
            continue

        add_dataset_dir(data_root / dataset_arg)

    return resolved


def update_counts_and_max_item(file_path, item_counter=None):
    max_item_id = -1

    if not file_path.exists():
        return max_item_id

    with file_path.open("r", encoding="utf-8", errors="replace") as input_file:
        for raw_line in input_file:
            parts = raw_line.strip().split()
            if len(parts) <= 1:
                continue

            item_ids = [int(token) for token in parts[1:]]
            if not item_ids:
                continue

            max_item_id = max(max_item_id, max(item_ids))
            if item_counter is not None:
                item_counter.update(item_ids)

    return max_item_id


def load_item_usage(dataset_dir, train_filename, test_filename):
    train_path = dataset_dir / train_filename
    test_path = dataset_dir / test_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train_counter = Counter()
    max_item_id = update_counts_and_max_item(train_path, train_counter)
    max_item_id = max(max_item_id, update_counts_and_max_item(test_path))

    item_count = max_item_id + 1 if max_item_id >= 0 else 0
    item_usage = np.zeros(item_count, dtype=np.int64)
    for item_id, usage_count in train_counter.items():
        item_usage[item_id] = usage_count

    return item_usage


def split_items_by_usage(item_usage, usage_threshold):
    if not 0 < usage_threshold <= 1:
        raise ValueError("--usage-threshold must be in the range (0, 1].")

    item_count = int(item_usage.shape[0])
    total_usage = int(item_usage.sum())

    if item_count == 0:
        return {
            "head_items": [],
            "tail_items": [],
            "head_count": 0,
            "tail_count": 0,
            "total_usage": 0,
            "actual_head_usage": 0,
            "actual_head_usage_ratio": 0.0,
        }

    sorted_item_ids = np.argsort(-item_usage, kind="stable")

    if total_usage == 0:
        return {
            "head_items": [],
            "tail_items": sorted(sorted_item_ids.tolist()),
            "head_count": 0,
            "tail_count": item_count,
            "total_usage": 0,
            "actual_head_usage": 0,
            "actual_head_usage_ratio": 0.0,
        }

    cumulative_ratio = np.cumsum(item_usage[sorted_item_ids]) / total_usage
    head_count = int(np.searchsorted(cumulative_ratio, usage_threshold, side="left") + 1)

    head_items = sorted(sorted_item_ids[:head_count].tolist())
    tail_items = sorted(sorted_item_ids[head_count:].tolist())
    actual_head_usage = int(item_usage[head_items].sum()) if head_items else 0

    return {
        "head_items": head_items,
        "tail_items": tail_items,
        "head_count": head_count,
        "tail_count": item_count - head_count,
        "total_usage": total_usage,
        "actual_head_usage": actual_head_usage,
        "actual_head_usage_ratio": actual_head_usage / total_usage,
    }


def write_item_file(file_path, item_ids):
    content = "".join(f"{item_id}\n" for item_id in item_ids)
    file_path.write_text(content, encoding="ascii")


def write_summary_json(file_path, dataset_name, usage_threshold, split_result):
    summary = {
        "dataset": dataset_name,
        "usage_threshold": usage_threshold,
        "head_count": split_result["head_count"],
        "tail_count": split_result["tail_count"],
        "total_usage": split_result["total_usage"],
        "actual_head_usage": split_result["actual_head_usage"],
        "actual_head_usage_ratio": split_result["actual_head_usage_ratio"],
    }
    file_path.write_text(json.dumps(summary, indent=2), encoding="ascii")


def process_dataset(dataset_dir, usage_threshold, train_filename, test_filename, dry_run, write_summary):
    item_usage = load_item_usage(dataset_dir, train_filename, test_filename)
    split_result = split_items_by_usage(item_usage, usage_threshold)

    dataset_name = dataset_dir.name
    print(
        f"[{dataset_name}] items={item_usage.shape[0]} total_usage={split_result['total_usage']} "
        f"threshold={usage_threshold:.2f} head={split_result['head_count']} "
        f"tail={split_result['tail_count']} actual_head_usage={split_result['actual_head_usage_ratio'] * 100:.2f}%"
    )

    if dry_run:
        return

    write_item_file(dataset_dir / "head.txt", split_result["head_items"])
    write_item_file(dataset_dir / "tail.txt", split_result["tail_items"])

    if write_summary:
        write_summary_json(
            dataset_dir / "head_tail_summary.json",
            dataset_name,
            usage_threshold,
            split_result,
        )


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    dataset_dirs = resolve_dataset_dirs(args.datasets, data_root)

    for dataset_dir in dataset_dirs:
        process_dataset(
            dataset_dir=dataset_dir,
            usage_threshold=args.usage_threshold,
            train_filename=args.train_file,
            test_filename=args.test_file,
            dry_run=args.dry_run,
            write_summary=args.write_summary_json,
        )


if __name__ == "__main__":
    main()
