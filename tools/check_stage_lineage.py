#!/usr/bin/env python3
"""Validate public Stage discovery, declared file lineage, hashes, and imports."""
import argparse
import ast
import hashlib
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stage(directory):
    return tomllib.loads((directory / "STAGE.toml").read_text(encoding="utf-8"))


def check(root, verify_files):
    errors = []
    dirs = [root / "src/stages/01_mpp", root / "src/stages/02_circuit_level", root]
    stages = {}
    for directory in dirs:
        path = directory / "STAGE.toml"
        if not path.is_file():
            errors.append(f"Missing stage metadata: {path.relative_to(root)}")
            continue
        try:
            data = read_stage(directory)
            sid = str(data["id"]).zfill(2)
            stages[sid] = (directory, data)
        except Exception as exc:
            errors.append(f"Invalid stage metadata {path.relative_to(root)}: {exc}")
    if set(stages) != {"01", "02", "03"}:
        errors.append(f"Expected public stages 01, 02, 03; found {sorted(stages)}")
        return errors
    expected_parents = {"01": "", "02": "01_mpp", "03": "02_circuit_level"}
    for sid, (directory, data) in stages.items():
        source = directory / data.get("source_root", "")
        if not source.is_dir():
            errors.append(f"Stage {sid} source_root missing: {source}")
        if data.get("inherits", "") != expected_parents[sid]:
            errors.append(f"Stage {sid} must inherit {expected_parents[sid]!r}")
        if bool(data.get("is_canonical", sid == "03")) != (sid == "03"):
            errors.append(f"Stage {sid} is_canonical must be {sid == '03'}")
        child = source
        parent = None
        if sid != "01":
            parent_id = "01" if sid == "02" else "02"
            parent_directory, parent_data = stages[parent_id]
            parent = parent_directory / parent_data["source_root"]
        forbidden = data.get("forbidden_import_prefixes", [])
        for py in child.rglob("*.py"):
            if any(part.startswith(".") or part in {"__pycache__", ".venv"} for part in py.parts):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except (SyntaxError, UnicodeDecodeError):
                errors.append(f"Stage {sid} has unparsable Python: {py.relative_to(root)}")
                continue
            for node in ast.walk(tree):
                name = node.module if isinstance(node, ast.ImportFrom) else None
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    names = [name] if name else []
                for imported in names:
                    for prefix in forbidden:
                        prefix = prefix.rstrip(".")
                        if imported == prefix or imported.startswith(prefix + "."):
                            errors.append(f"{py.relative_to(root)}:{node.lineno} imports forbidden {imported}")
        if not verify_files:
            continue
        added, changed = set(data.get("added", [])), set(data.get("changed", []))
        unchanged, removed = set(data.get("unchanged", [])), set(data.get("removed", []))
        groups = [added, changed, unchanged, removed]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
            errors.append(f"Stage {sid} lineage declarations overlap")
        for rel in added:
            if not (child / rel).is_file():
                errors.append(f"Stage {sid} added file missing: {rel}")
            if parent is not None and (parent / rel).exists():
                errors.append(f"Stage {sid} added file already exists in parent: {rel}")
        for rel in changed:
            if parent is None or not (child / rel).is_file() or not (parent / rel).is_file():
                errors.append(f"Stage {sid} changed file missing in a generation: {rel}")
            elif digest(child / rel) == digest(parent / rel):
                errors.append(f"Stage {sid} changed file hash is identical to parent: {rel}")
        for rel in unchanged:
            if parent is None or not (child / rel).is_file() or not (parent / rel).is_file():
                errors.append(f"Stage {sid} unchanged file missing in a generation: {rel}")
            elif digest(child / rel) != digest(parent / rel):
                errors.append(f"Stage {sid} unchanged file hash differs from parent: {rel}")
        for rel in removed:
            if parent is None or (child / rel).exists() or not (parent / rel).exists():
                errors.append(f"Stage {sid} removed declaration is inconsistent: {rel}")
        actual = {str(p.relative_to(child)) for p in child.rglob("*") if p.is_file() and not any(x.startswith(".") or x in {"__pycache__", ".venv"} for x in p.parts)}
        undeclared = actual - added - changed - unchanged
        if undeclared:
            errors.append(f"Stage {sid} has undeclared source files: {sorted(undeclared)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    errors = check(args.repo_root.resolve(), args.verify_files)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Public stage lineage and import boundaries: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
