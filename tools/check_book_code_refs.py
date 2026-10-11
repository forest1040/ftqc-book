#!/usr/bin/env python3
"""Check public book paths, chapter/stage mapping, TeX labels, and citations."""
import argparse
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH_RE = re.compile(r"(?<![\w])(?:src/(?:stages/\d{2}_[\w-]+/src/ftqc|ftqc)(?:/[\w.-]+)*/?)(?![\w])")
PRIVATE_PATH_RE = re.compile(r"(?<![\w])(?:code/(?:mpp|ls_s|d3|advanced)|ref/fable)/[^\s}\])]+")


def untex(value):
    return value.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")


def check(root):
    errors = []
    map_path = root / "book/code-map.toml"
    try:
        data = tomllib.loads(map_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Cannot read code map {map_path}: {exc}"]
    chapters = data.get("chapters", {})
    stages = data.get("stages", {})
    if set(chapters) != {str(n) for n in range(1, 26)}:
        errors.append("code-map must contain exactly chapters 1 through 25")
    if set(stages) != {"01_mpp", "02_circuit_level", "03_d3_integrated"}:
        errors.append("code-map stages must be exactly 01_mpp, 02_circuit_level, 03_d3_integrated")
    expected = {"01_mpp": (6, 13), "02_circuit_level": (14, 19), "03_d3_integrated": (20, 25)}
    for sid, (first, last) in expected.items():
        stage = stages.get(sid, {})
        source = stage.get("source_root", "")
        if not source or not (root / source).is_dir():
            errors.append(f"Stage {sid} source_root is missing: {source}")
        if stage.get("is_canonical") != (sid == "03_d3_integrated"):
            errors.append(f"Stage {sid} has incorrect is_canonical value")
        current = stage.get("current_chapters", [])
        if current != list(range(first, last + 1)):
            errors.append(f"Stage {sid} current_chapters must be {list(range(first, last + 1))}")
    book = root / "book"
    book_docs = sorted([*book.rglob("*.tex"), *book.rglob("*.md")])
    public_docs = [root / "README.md", *(root / "docs").rglob("*.md")]
    docs = sorted({*book_docs, *(path for path in public_docs if path.is_file())})
    labels, refs, citations = set(), [], []
    for doc in docs:
        content = doc.read_text(encoding="utf-8", errors="replace")
        if doc in book_docs:
            labels.update(re.findall(r"\\label\{([^}]+)\}", content))
            refs.extend((doc, m.group(1)) for m in re.finditer(r"\\(?:ref|autoref|eqref|pageref)\{([^}]+)\}", content))
            citations.extend((doc, key.strip()) for m in re.finditer(r"\\cite\w*\{([^}]+)\}", content) for key in m.group(1).split(","))
        for raw in PATH_RE.findall(untex(content)):
            raw = raw.rstrip(".,;:!?")
            if not (root / raw).exists():
                errors.append(f"{doc.relative_to(root)} references missing code path {raw}")
        for raw in PRIVATE_PATH_RE.findall(untex(content)):
            errors.append(f"{doc.relative_to(root)} retains private/retired code path {raw}")
    for doc, label in refs:
        if label not in labels:
            errors.append(f"{doc.relative_to(root)} has unresolved TeX reference {label}")
    bib_keys = set()
    for bib in book.rglob("*.bib"):
        bib_keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib.read_text(encoding="utf-8", errors="replace")))
    for doc in book_docs:
        bib_keys.update(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", doc.read_text(encoding="utf-8", errors="replace")))
    for doc, key in citations:
        if key not in bib_keys:
            errors.append(f"{doc.relative_to(root)} cites missing bibliography key {key}")
    for num in range(1, 26):
        entry = chapters.get(str(num), {})
        ch_file = entry.get("file", "")
        if not ch_file or not (root / ch_file).is_file():
            errors.append(f"Chapter {num} file missing: {ch_file}")
        wanted = "01_mpp" if 6 <= num <= 13 else "02_circuit_level" if 14 <= num <= 19 else "03_d3_integrated" if num >= 20 else ""
        if entry.get("stage", "") != wanted:
            errors.append(f"Chapter {num} must map to stage {wanted!r}")
        mapped_source = stages.get(wanted, {}).get("source_root", "") if wanted else ""
        for field in ("stage_modules", "canonical_modules"):
            for path in entry.get(field, []):
                prefix = mapped_source.rstrip("/") + "/" if field == "stage_modules" else "src/ftqc/"
                if (field == "stage_modules" and not wanted) or (field == "canonical_modules" and wanted != "03_d3_integrated"):
                    errors.append(f"Chapter {num} cannot declare {field}: {path}")
                    continue
                if not prefix or not path.startswith(prefix) or not (root / path).is_file():
                    errors.append(f"Chapter {num} {field} path invalid or missing: {path}")
        if entry.get("legacy_modules"):
            errors.append(f"Chapter {num} still maps private/legacy module paths")
    allowed_apps = {"aa", "ab", "ac", "ad"}
    apps = data.get("appendices", {})
    if set(apps) != allowed_apps:
        errors.append("code-map must contain only public appendices aa through ad")
    for key, app in apps.items():
        if not (root / app.get("file", "")).is_file():
            errors.append(f"Appendix {key} file missing: {app.get('file')}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--strict", action="store_true", help="fail on every invalid public code path")
    parser.add_argument("--check-code-map", action="store_true", help="validate chapter/stage map and TeX references")
    args = parser.parse_args()
    errors = check(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Public book code paths, chapter map, TeX references, and citations: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
