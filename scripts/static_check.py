"""Check syntax, contracts and dependency direction without importing the project."""
from __future__ import annotations
import ast
import builtins
import hashlib
import json
import math
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "_archive/v2.1.1_before_cleanup_20261012"


def main():
    checks = []
    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(f"Static check failed: {name}")

    sources, trees = {}, {}
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        trees[rel] = ast.parse(source, filename=rel)
        compile(trees[rel], rel, "exec")
        sources[rel] = source
    check("python_ast_and_compile", bool(sources))
    for rel, source in sources.items():
        table = symtable.symtable(source, rel, "exec")
        declared = {s.get_name() for s in table.get_symbols()
                    if s.is_assigned() or s.is_imported() or s.is_namespace()}
        allowed = declared | set(dir(builtins)) | {"__file__", "__name__", "__package__", "__doc__"}
        missing = set()
        def referenced_globals(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in allowed:
                    missing.add(symbol.get_name())
            for child in scope.get_children():
                referenced_globals(child)
        referenced_globals(table)
        check(f"defined_global_names:{rel}:{sorted(missing)}", not missing)
    configs = {p.relative_to(ROOT).as_posix(): json.loads(p.read_text(encoding="utf-8-sig"))
               for p in ROOT.rglob("*.json") if "checks" not in p.parts}
    check("json_parse", bool(configs))
    check("one_readme", {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.md")} == {"README.md"})
    removed = ("island_toy", "results", "experiments", "ERA5Downloading.py", "TOY_SERVER/era5_config.json",
               "TOY_SERVER/era5_bulk.example.json", "TOY_SERVER/request_era5.py", "TOY_SERVER/era5_jobs.py",
               "TOY_SERVER/era5_merge.py", "TOY_SERVER/island_core.py", "TOY_SERVER/coverage_generator.py",
               "TOY_SERVER/forcing_arrays.py", "TRAIN_SERVER/forcing_arrays.py")
    check("legacy_and_download_files_removed", all(not (ROOT / name).exists() for name in removed))
    forbidden = {"cdsapi", "xarray", "netCDF4", "era5_jobs", "era5_merge", "island_core", "coverage_generator",
                 "forcing_arrays", "wave_geometry"}
    imported = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
    check("obsolete_imports_removed", not (forbidden & imported))
    check("toy_dependencies_minimal", (ROOT / "TOY_SERVER/requirements.txt").read_text().splitlines()
          == ["numpy>=1.26,<3", "matplotlib>=3.8,<4"])
    config = configs["TOY_SERVER/config.json"]
    exp, sampling = config["experiment"], config["experiment"]["case_sampling"]
    check("download_config_removed", not ({"era5", "sampling"} & config.keys())
          and not ({"wind", "waves"} & config["paths"].keys()))
    target = sampling["target_effective_train_cases"]
    candidates = math.ceil(target*(1+sampling["training_reserve_fraction"]))
    check("default_target_preserved", target == 16384 and candidates == 20480)
    nt, nf = exp["terrain"]["count"], exp["forcing"]["count"]
    check("training_capacity", candidates <= nt["train"]*nf["train"])
    for split in ("validation", "test"):
        check(f"{split}_capacity", sampling["evaluation_cases_per_group"] <= min(
            nt[split]*nf["train"], nt["train"]*nf[split], nt[split]*nf[split]))
    check("fixed_spectrum", config["boundary"] == {"period": "peak", "jonswap_gamma": 3.3,
                                                    "directional_spread_degrees": 30.0})
    check("feature_schema_preserved", exp["normalization"]["terrain_channel"] == "relative_depth"
          and sampling["require_converged"] is False)
    check("bootstrap_copies_equal", (ROOT / "TOY_SERVER/_bootstrap.py").read_bytes()
          == (ROOT / "TRAIN_SERVER/_bootstrap.py").read_bytes())

    production = {r for r in sources if "tests/" not in r and not r.startswith("scripts/")}
    edges = {r: set() for r in production}
    def resolve(module, caller):
        for directory in ((ROOT / caller).parent, (ROOT / caller).parent.parent, ROOT):
            candidate = directory / module.replace(".", "/")
            for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                if path.is_file():
                    return path.relative_to(ROOT).as_posix()
        return None
    def symbols(tree):
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return names
    for rel, tree in trees.items():
        for node in ast.walk(tree):
            modules = [node.module] if isinstance(node, ast.ImportFrom) else (
                [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                destination = resolve(module, rel)
                if destination is None:
                    continue
                if isinstance(node, ast.ImportFrom):
                    check(f"imports:{rel}:{module}", all(a.name in symbols(trees[destination]) for a in node.names))
                if rel in production and destination in production:
                    edges[rel].add(destination)
    visiting, visited = set(), set()
    def visit(module):
        if module in visiting:
            raise ValueError(f"Circular production dependency: {module}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in edges[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)
    for module in edges:
        visit(module)
    check("production_dependency_graph_acyclic", True)
    check("shared_has_no_server_dependency", all(not any(d.startswith(("TOY_SERVER/", "TRAIN_SERVER/"))
          for d in dependencies) for r, dependencies in edges.items() if r.startswith("shared/")))
    check("servers_do_not_import_each_other", all(not any(d.startswith("TRAIN_SERVER/") for d in deps)
          for r, deps in edges.items() if r.startswith("TOY_SERVER/")) and all(
          not any(d.startswith("TOY_SERVER/") for d in deps) for r, deps in edges.items() if r.startswith("TRAIN_SERVER/")))
    check("runtime_does_not_import_renderer", "TOY_SERVER/swan_inputs.py" not in edges["TOY_SERVER/swan_runtime.py"])
    check("quota_does_not_import_generation", "TOY_SERVER/generation.py" not in edges["TOY_SERVER/effective_cases.py"])
    generator_tree = trees["TOY_SERVER/generate_cases.py"]
    calls = {n.func.id for n in ast.walk(generator_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("incremental_case_reuse_connected", {"reuse_case", "resume_sampling", "write_manifest"}.issubset(calls))
    reuse = next(n for n in trees["TOY_SERVER/swan_inputs.py"].body
                 if isinstance(n, ast.FunctionDef) and n.name == "reuse_case")
    check("case_reuse_does_not_write_results", not any(isinstance(n, ast.Attribute)
          and n.attr in {"write_bytes", "write_text", "mkdir", "unlink"} for n in ast.walk(reuse)))
    if BASE.is_dir():
        def functions(path):
            return {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        current = functions(ROOT / "shared/wave_features.py")
        for old in ("forcing_arrays.py", "wave_geometry.py"):
            for name, node in functions(BASE / "TOY_SERVER" / old).items():
                check(f"feature_algorithm_unchanged:{name}", ast.dump(node) == ast.dump(current[name]))
        current = functions(ROOT / "TOY_SERVER/generation.py")
        original = functions(BASE / "TOY_SERVER/coverage_generator.py")
        rename = {"generate_terrains": "generate_islands", "generate_conditions": "generate_forcings", "sparse_plan": "case_plan"}
        for name in ("bounds", "rng_for", "strata", "between", "validate_config", "generate_terrains",
                     "generate_conditions", "pair_indices", "sparse_plan", "coverage_summary"):
            node = current[rename.get(name, name)]
            node.name = name
            check(f"generation_algorithm_unchanged:{name}", ast.dump(original[name]) == ast.dump(node))
        for name in ("make_island_depth", "local_xy_m", "domain_size_m", "island_to_row", "forcing_to_row"):
            check(f"geometry_unchanged:{name}", ast.dump(functions(BASE / "TOY_SERVER/island_core.py")[name]) == ast.dump(current[name]))
    for entry in ("generate_cases", "run_swan", "prepare_dataset", "inspect_cases"):
        check(f"entry_present:{entry}", f"TOY_SERVER/{entry}.py" in sources)
    for entry in ("train", "evaluate", "check_dataset", "check_environment", "migrate_dataset"):
        check(f"entry_present:{entry}", f"TRAIN_SERVER/{entry}.py" in sources)
    report = {"version": "2.1.1", "mode": "static_only", "passed": len(checks), "checks": checks,
              "python_sources": len(sources), "configuration_json_files": len(configs),
              "baseline_available": BASE.is_dir(), "runtime_tests_executed": False,
              "case_generation_executed": False, "swan_executed": False, "training_executed": False,
              "dependency_graph": {r: sorted(d) for r, d in sorted(edges.items())},
              "source_hashes_lf": {r: hashlib.sha256(s.encode()).hexdigest() for r, s in sources.items()}}
    destination = ROOT / "checks/static_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Passed {len(checks)} static checks across {len(sources)} Python sources; no project code executed.")


if __name__ == "__main__":
    main()
