"""Static-only v2.1.1 checks; never import or execute application modules."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "v2"


def main():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(f"Static check failed: {name}")

    sources = {}
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")  # compile only, never exec
        sources[path.relative_to(ROOT).as_posix()] = (source, tree)
    check("all_python_ast_and_compile", bool(sources))
    json_count = 0
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"))
        json_count += 1
    check("all_json_parse", json_count > 0)

    config = json.loads((ROOT / "TOY_SERVER/config.json").read_text(encoding="utf-8"))
    exp = config["experiment"]
    check("version_and_generator", config["version"] == "2.1.1" and exp["generator"] == "broad-coverage-1")
    sampling = exp["case_sampling"]
    target = sampling["target_effective_train_cases"]
    candidates = math.ceil(target * (1 + sampling["training_reserve_fraction"]))
    count = exp["terrain"]["count"]
    forcing_count = exp["forcing"]["count"]
    check("default_effective_target", target == 16384)
    check("default_candidates", candidates == 20480)
    check("unique_pair_capacity", candidates <= count["train"] * forcing_count["train"])
    n_eval = sampling["evaluation_cases_per_group"]
    for split in ("validation", "test"):
        capacity = min(count[split]*forcing_count["train"], count["train"]*forcing_count[split],
                       count[split]*forcing_count[split])
        check(f"{split}_capacity", n_eval <= capacity)
    check("default_total_candidates", candidates+6*n_eval == 23552)
    check("default_full_effective_dataset", target+6*n_eval == 19456)
    check("fixed_spectrum", config["boundary"] == {"period": "peak", "jonswap_gamma": 3.3,
                                                    "directional_spread_degrees": 30.0})
    check("same_forcing_spectrum", exp["forcing"]["jonswap_gamma"] == 3.3
          and exp["forcing"]["directional_spread_deg"] == 30.0)
    check("relative_depth", exp["normalization"]["terrain_channel"] == "relative_depth")
    check("record_only_default", sampling["require_converged"] is False)
    terrain = exp["terrain"]
    check("original_terrain_ranges", terrain["base_depth_m"] == 100.0
          and terrain["island_radius_km"] == [3.0, 8.0]
          and terrain["skirt_width_km"] == [5.0, 12.0]
          and terrain["shore_depth_m"] == [3.0, 15.0])
    check("ellipse_ratio", 1 < terrain["ellipse_axis_ratio"][0] <= terrain["ellipse_axis_ratio"][1])

    for name in ("island_core.py", "coverage_generator.py"):
        check(f"sync_island_toy_{name}", (ROOT / "TOY_SERVER" / name).read_bytes()
              == (ROOT / "island_toy" / name).read_bytes())
    check("sync_swan_inputs", (ROOT / "TOY_SERVER/swan_inputs.py").read_bytes()
          == (ROOT / "swan_inputs.py").read_bytes())
    toy = json.loads((ROOT / "island_toy/config.json").read_text(encoding="utf-8"))
    check("toy_config_sync", all(toy[key] == exp[key]
          for key in ("generator", "terrain", "forcing", "case_sampling")))

    # Feature implementation must match the supplied v2.1 baseline when available.
    base_present = BASE.is_dir()
    if base_present:
        for rel in ("TOY_SERVER/forcing_arrays.py", "TOY_SERVER/wave_geometry.py",
                    "TRAIN_SERVER/forcing_arrays.py", "TRAIN_SERVER/wave_geometry.py"):
            check(f"unchanged_{rel}", (ROOT / rel).read_text(encoding="utf-8")
                  == (BASE / rel).read_text(encoding="utf-8"))
        def function_dumps(relative):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse((BASE / relative).read_text(encoding="utf-8")).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        baseline = function_dumps("TOY_SERVER/swan_inputs.py")
        current = {n.name: ast.dump(n, include_attributes=False)
                   for n in sources["TOY_SERVER/swan_inputs.py"][1].body if isinstance(n, ast.FunctionDef)}
        check("unchanged_boundary_renderer", all(current[n] == baseline[n]
              for n in ("boundary_blocks", "render_input")))

    # Resolve new modules using AST, rather than importing numpy or the project.
    module_files = {p.stem for p in (ROOT / "TOY_SERVER").glob("*.py")}
    for rel in ("TOY_SERVER/coverage_generator.py", "TOY_SERVER/effective_cases.py",
                "TOY_SERVER/generate_cases.py", "TOY_SERVER/run_swan.py", "TOY_SERVER/prepare_dataset.py"):
        tree = sources[rel][1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in module_files:
                other = sources[f"TOY_SERVER/{node.module}.py"][1]
                definitions = {n.name for n in other.body
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                definitions.update(t.id for n in other.body if isinstance(n, ast.Assign)
                                   for t in n.targets if isinstance(t, ast.Name))
                check(f"local_import_{rel}_{node.module}", all(a.name in definitions for a in node.names))

    for rel in ("TOY_SERVER/generate_cases.py", "TOY_SERVER/inspect_cases.py",
                "island_toy/build_island_toy.py"):
        calls = [n for n in ast.walk(sources[rel][1]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "case_plan"]
        check(f"configured_case_plan_{rel}", bool(calls) and all(len(n.args) == 3 for n in calls))
    for rel, flag in (("TOY_SERVER/generate_cases.py", "--target-train-cases"),
                      ("TOY_SERVER/generate_cases.py", "--training-candidates"),
                      ("TOY_SERVER/generate_cases.py", "--plan-only"),
                      ("TOY_SERVER/prepare_dataset.py", "--allow-partial")):
        check(f"cli_{flag}", any(isinstance(n, ast.Constant) and n.value == flag
                                for n in ast.walk(sources[rel][1])))
    check("shared_runner_eligibility", "saved_case(root, row, require_converged)" in sources["TOY_SERVER/run_swan.py"][0])
    check("converter_preflight", "select_effective(root, rows, target" in sources["TOY_SERVER/prepare_dataset.py"][0])
    check("training_quota_metadata_validation", "target_effective_train_cases" in sources["TRAIN_SERVER/training_core.py"][0])

    report = {"version": "2.1.1", "mode": "static_only", "checks": checks,
              "passed": len(checks), "python_sources": len(sources), "json_files": json_count,
              "baseline_available": base_present, "runtime_tests_executed": False,
              "case_generation_executed": False, "swan_executed": False, "training_executed": False,
              "source_hashes": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                                for rel in sources if rel.startswith(("TOY_SERVER/", "island_toy/", "TRAIN_SERVER/"))}}
    destination = ROOT / "experiments/v2.1.1/static_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(checks)} static checks, {len(sources)} Python sources, {json_count} JSON files. "
          "No application modules imported or executed.")


if __name__ == "__main__":
    main()
