"""Check all transferred shards before paying for GPU training."""
import argparse
import json
from pathlib import Path

from training_core import validate_dataset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/gebco15s"))
    parser.add_argument("--allow-incomplete-splits", action="store_true", help="Inspect a partial delivery without declaring it trainable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, _, report = validate_dataset(args.data, require_splits=not args.allow_incomplete_splits)
    report["ready_for_training"] = all(report["splits"].values())
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
