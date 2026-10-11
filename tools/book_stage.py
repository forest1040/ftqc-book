#!/usr/bin/env python3
"""List, inspect, test, and run the three public book stages."""
import argparse
import pathlib
import shutil
import subprocess
import sys
import tomllib


def repo_root():
    return pathlib.Path(__file__).resolve().parent.parent


def stages(root):
    result = {}
    for directory in [root, root / "src/stages/01_mpp", root / "src/stages/02_circuit_level"]:
        path = directory / "STAGE.toml"
        if path.exists():
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            result[str(data["id"]).zfill(2)] = (directory, data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=repo_root())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for command in ("info", "test"):
        sub = commands.add_parser(command)
        sub.add_argument("stage_id")
    run = commands.add_parser("run")
    run.add_argument("stage_id")
    run.add_argument("example_name")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    available = stages(root)
    if args.command == "list":
        for sid, (directory, data) in sorted(available.items()):
            print(f"{sid}  {directory.relative_to(root)}  {data.get('display_name', data.get('name', ''))}")
        return 0
    key = args.stage_id.zfill(2) if args.stage_id.isdigit() else args.stage_id
    match = next(((d, m) for sid, (d, m) in available.items() if sid == key or d.name == key), None)
    if match is None:
        print(f"Unknown public stage: {args.stage_id}", file=sys.stderr)
        return 2
    directory, data = match
    if args.command == "info":
        for name, value in data.items():
            print(f"{name}: {value}")
        return 0
    if args.command == "test":
        if shutil.which("uv"):
            command = ["uv", "run", "--locked", "pytest", "-q", "tests"]
        else:
            command = [sys.executable, "-m", "pytest", "-q", "tests"]
        return subprocess.run(command, cwd=directory).returncode
    script = directory / "examples" / args.example_name
    if not script.is_file():
        print(f"Example not found: {script}", file=sys.stderr)
        return 2
    if shutil.which("uv"):
        command = ["uv", "run", "--locked", "python", str(script)]
    else:
        command = [sys.executable, str(script)]
    return subprocess.run(command, cwd=directory).returncode


if __name__ == "__main__":
    raise SystemExit(main())
