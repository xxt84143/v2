"""Build independent upload archives from canonical source and shared contracts."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"__pycache__", ".git", ".venv", "cases_island", "dataset", "data", "inspection", "local_check", "runs"}


def files_for(package):
    for path in (ROOT / package).rglob("*"):
        if (path.is_file() and not (set(path.relative_to(ROOT / package).parts) & EXCLUDED)
                and path.suffix not in {".pyc", ".zip", ".pt", ".npz", ".nc"}
                and path.name not in {".env", ".cdsapirc", ".netrc", "coverage_plan.json"}):
            yield path.relative_to(ROOT / package).as_posix(), path.read_bytes()
    for path in (ROOT / "shared").glob("*.py"):
        yield f"shared/{path.name}", path.read_bytes()
    yield "README.md", (ROOT / "README.md").read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = []
    for package in ("TOY_SERVER", "TRAIN_SERVER"):
        payload = dict(files_for(package))
        for name, content in payload.items():
            if name.endswith(".py"):
                compile(ast.parse(content.decode("utf-8-sig")), name, "exec")
            elif name.endswith(".json"):
                json.loads(content)
        payload["CHECKSUMS.sha256"] = ("\n".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}" for name, content in sorted(payload.items()))+"\n").encode()
        archive_path = destination / f"{package}_v2.1.1.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(payload.items()):
                archive.writestr(f"{package}/{name}", content)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
            for name, content in payload.items():
                if archive.read(f"{package}/{name}") != content:
                    raise ValueError(f"Source/ZIP mismatch: {name}")
            if json.loads(archive.read(f"{package}/config.json"))["version"] != "2.1.1":
                raise ValueError("Unexpected package version")
            if any("era5" in name.lower() for name in payload):
                raise ValueError("Unexpected download file in upload archive")
        report.append({"file": archive_path.name, "files": len(payload),
                       "bytes": archive_path.stat().st_size,
                       "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                       "shared_included": True, "checksum_manifest_verified": True})
    output = ROOT / "checks/upload_packages.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
