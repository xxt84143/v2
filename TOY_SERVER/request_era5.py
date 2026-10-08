"""Plan, resume and download bulk ERA5 with at most five requests in flight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from era5_jobs import cached, download_all, download_lock, download_settings, effective_plans, requests_for
from era5_merge import merge_group, validate_download
from project import load_config, path_for, write_json

DEFAULT_ERA5_CONFIG = Path(__file__).resolve().with_name("era5_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ERA5_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; no network, token access or file writes")
    parser.add_argument("--download-only", action="store_true", help="Download validated chunks; do not merge canonical files")
    parser.add_argument("--merge-only", action="store_true", help="Merge validated downloads without CDS access")
    parser.add_argument("--overwrite", action="store_true", help="Replace canonical NetCDF files; keep completed raw chunks")
    parser.add_argument("--workers", type=int, help="Override simultaneous requests (1..5)")
    parser.add_argument("--verify-cache", action="store_true", help="Recompute completed raw-file SHA256 before reusing")
    args = parser.parse_args()
    if args.merge_only and args.download_only:
        parser.error("--merge-only and --download-only cannot be combined")
    config = load_config(args.config)
    if args.workers is not None:
        config["era5"].setdefault("download", {})["workers"] = args.workers
    settings = download_settings(config)
    plans = requests_for(config)
    print(f"{len(plans)} initial requests; workers={settings['workers']}; variables/request={settings['variables_per_request']}", flush=True)
    if args.dry_run:
        print(json.dumps({"dataset": config["era5"]["dataset"], "settings": settings, "plans": plans}, default=str, indent=2))
        return
    if not args.download_only:
        for group in ("wind", "waves"):
            if path_for(config, group).exists() and not args.overwrite:
                raise FileExistsError(f"{path_for(config, group)} exists; use --download-only or --overwrite")
    raw_root = Path(config["_base"]) / "data" / "raw"
    with download_lock(raw_root):
        write_json(raw_root / "plan.json", {"schema_version": "era5-download-plan-1", "settings": settings,
                                            "plans": [{**plan, "target": str(plan["target"])} for plan in plans]})
        if args.merge_only:
            leaves = effective_plans(plans)
            for plan in leaves:
                if not cached(plan, args.verify_cache):
                    raise FileNotFoundError(f"No validated completion marker: {plan['target']}")
                validate_download(plan, plan["target"])
        else:
            leaves = download_all(plans, settings, validate_download, args.verify_cache)
        write_json(raw_root / "manifest.json", {"schema_version": "era5-download-manifest-1",
                                               "plans": [{**plan, "target": str(plan["target"])} for plan in leaves]})
        if not args.download_only:
            for group in ("wind", "waves"):
                merge_group(config, leaves, group, args.overwrite, settings["merge_time_chunk"])


if __name__ == "__main__":
    main()
