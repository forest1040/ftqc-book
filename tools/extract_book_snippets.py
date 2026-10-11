#!/usr/bin/env python3
"""
tools/extract_book_snippets.py - Extract and verify named code snippets for book inclusion.

Parses source code or markdown/TeX files for snippet markers:
  book-snippet:NAME-start
  ...
  book-snippet:NAME-end
Supports verification mode (--verify), snippet output verification (--verify-output),
and prevents nesting/crossing/duplicate marker definitions.
Does not write files to disk unless --write is explicitly passed.
"""

import argparse
import os
import pathlib
import re
import sys
from typing import Dict, List, Tuple, Any, Optional

def find_repo_root(start_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Find repository root dynamically, prioritizing .git directory."""
    if start_path is None:
        start_path = pathlib.Path.cwd()
    curr = start_path.resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists():
            return parent
    for parent in [curr] + list(curr.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return pathlib.Path(__file__).resolve().parent.parent

START_PATTERN = re.compile(r'(?:#|//|%|<!--)\s*book-snippet:([a-zA-Z0-9_-]+)-start(?:\s*-->)?')
END_PATTERN = re.compile(r'(?:#|//|%|<!--)\s*book-snippet:([a-zA-Z0-9_-]+)-end(?:\s*-->)?')

def extract_snippets_from_file(file_path: pathlib.Path) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extract snippets from a single file.
    Enforces no nesting/crossing and detects duplicate snippet names even after closing.
    Returns:
      (snippets_dict: {name: lines}, errors: list_of_error_messages)
    """
    snippets = {}
    errors = []
    seen_names_in_file = set()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return {}, [f"Could not open file {file_path}: {e}"]

    # Active stack of snippets: list of (name, start_line_num, accumulated_lines)
    active_stack: List[Tuple[str, int, List[str]]] = []

    for idx, line in enumerate(lines, 1):
        start_match = START_PATTERN.search(line)
        end_match = END_PATTERN.search(line)

        if start_match:
            name = start_match.group(1)

            # Check if snippet was already defined earlier in this file
            if name in seen_names_in_file:
                errors.append(f"{file_path}:{idx}: Duplicate snippet definition for '{name}' (already defined previously in this file).")
            seen_names_in_file.add(name)

            # Check nesting: nesting is not permitted
            if active_stack:
                parent_name = active_stack[-1][0]
                errors.append(f"{file_path}:{idx}: Nested snippet '{name}' inside active snippet '{parent_name}' is not allowed.")

            active_stack.append((name, idx, []))
            continue

        if end_match:
            name = end_match.group(1)
            if not active_stack:
                errors.append(f"{file_path}:{idx}: Snippet end for '{name}' without any active open snippet.")
            else:
                top_name, top_line, content_lines = active_stack[-1]
                if top_name != name:
                    errors.append(f"{file_path}:{idx}: Mismatched/crossed snippet end tag '{name}' (expected end for '{top_name}' opened at line {top_line}).")
                else:
                    active_stack.pop()
                    snippets[name] = content_lines
            continue

        # Accumulate content lines into current active snippet
        if active_stack:
            for item in active_stack:
                item[2].append(line)

    for name, start_line, _ in active_stack:
        errors.append(f"{file_path}:{start_line}: Snippet '{name}' was opened but never closed.")

    return snippets, errors

def collect_scan_files(target_path: pathlib.Path) -> List[pathlib.Path]:
    """Collect text sources while pruning generated, cached, and VCS trees."""
    if target_path.is_file():
        return [target_path]
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "site-packages", "build", "dist"}
    files = []
    for directory, dirnames, filenames in os.walk(target_path):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))
        current = pathlib.Path(directory)
        for filename in filenames:
            if any(filename.endswith(ext) for ext in (".py", ".tex", ".md", ".toml", ".sh")):
                files.append(current / filename)
    return sorted(files)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and verify named code snippets for book inclusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", nargs="?", help="File or directory to scan/extract from (defaults to book and public source trees)")
    parser.add_argument("--repo-root", type=str, help="Path to repository root")
    parser.add_argument("--name", type=str, help="Filter for specific snippet name")
    parser.add_argument("--output-dir", type=str, help="Directory to save/verify extracted snippet files")
    parser.add_argument("--write", action="store_true", help="Explicitly write extracted snippets to output directory")
    parser.add_argument("--verify", "--check", action="store_true", help="Verify snippet markers for correctness across files")
    parser.add_argument("--verify-output", action="store_true", help="Verify that existing files in output-dir match freshly extracted snippets")

    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root) if args.repo_root else find_repo_root()

    if args.target:
        target_path = pathlib.Path(args.target)
        if not target_path.is_absolute():
            target_path = (pathlib.Path.cwd() / target_path).resolve()
    else:
        target_path = None

    if target_path is None:
        targets = [repo_root / name for name in ("book", "src", "examples")]
        missing = [path for path in targets if not path.is_dir()]
        if missing:
            print(f"Default snippet source directory missing: {missing[0]}", file=sys.stderr)
            return 1
        files_to_scan = sorted({f for path in targets for f in collect_scan_files(path)})
        display_target = repo_root
    else:
        if not target_path.exists():
            print(f"Error: Target path '{target_path}' does not exist.", file=sys.stderr)
            return 1
        files_to_scan = collect_scan_files(target_path)
        display_target = target_path

    all_snippets = {}
    all_errors = []

    for f in sorted(files_to_scan):
        snips, errs = extract_snippets_from_file(f)
        all_errors.extend(errs)
        for name, lines in snips.items():
            if name in all_snippets:
                all_errors.append(f"Global snippet name collision for '{name}' across files ({f.relative_to(repo_root)} vs {all_snippets[name][0].relative_to(repo_root)}).")
            else:
                all_snippets[name] = (f, lines)

    if args.verify:
        print(f"=== Verifying Snippet Markers in {display_target} ===")
        print(f"Scanned {len(files_to_scan)} files. Found {len(all_snippets)} unique snippets.")

        if all_errors:
            print(f"\n[SNIPPET VERIFICATION FAILED ({len(all_errors)} errors)]:", file=sys.stderr)
            for err in all_errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

        print("Snippet marker verification: PASSED.")
        return 0

    if args.verify_output:
        if not args.output_dir:
            print("Error: --verify-output requires --output-dir <directory>", file=sys.stderr)
            return 1

        out_dir = pathlib.Path(args.output_dir)
        print(f"=== Verifying Snippet Outputs in {out_dir} ===")

        if not out_dir.exists():
            print(f"[STALE/MISSING] Output directory {out_dir} does not exist.", file=sys.stderr)
            return 1

        stale_errors = []
        for name, (source_file, lines) in all_snippets.items():
            expected_content = "".join(lines)
            target_file = out_dir / f"{name}.snippet"

            if not target_file.exists():
                stale_errors.append(f"Missing snippet output file: {target_file.name}")
            else:
                with open(target_file, "r", encoding="utf-8") as tf:
                    actual_content = tf.read()
                if actual_content != expected_content:
                    stale_errors.append(f"Stale snippet output file (content mismatch): {target_file.name}")

        if stale_errors:
            print(f"\n[OUTPUT VERIFICATION FAILED ({len(stale_errors)} errors)]:", file=sys.stderr)
            for err in stale_errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

        print("Snippet output verification: PASSED (all files are up to date).")
        return 0

    # Extraction / Output mode
    if all_errors:
        for err in all_errors:
            print(f"Warning: {err}", file=sys.stderr)

    if args.name:
        if args.name not in all_snippets:
            print(f"Error: Snippet '{args.name}' not found.", file=sys.stderr)
            return 1
        source_file, lines = all_snippets[args.name]
        print(f"--- Snippet '{args.name}' from {source_file.relative_to(repo_root)} ---")
        sys.stdout.writelines(lines)
        return 0

    print(f"Found {len(all_snippets)} snippets:")
    for name, (source_file, lines) in sorted(all_snippets.items()):
        rel_src = source_file.relative_to(repo_root) if source_file.is_relative_to(repo_root) else source_file
        print(f"  - '{name}' ({len(lines)} lines) in {rel_src}")

        if args.output_dir:
            out_dir = pathlib.Path(args.output_dir)
            if args.write:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{name}.snippet"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"    Written to {out_file}")
            else:
                print(f"    (Use --write to save to {out_dir})")

    return 0

if __name__ == "__main__":
    sys.exit(main())
