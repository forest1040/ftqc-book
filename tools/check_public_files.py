#!/usr/bin/env python3
"""Enforce the exact public file allowlist and scan tracked content for private residue."""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_PREFIX = r"https?://"
PRIVATE_HOST_PARTS = ("ftqc-", "tracker")
URL = re.compile(URL_PREFIX + r"[^\s]*(?:internal|corp|private|localhost|" + "".join(PRIVATE_HOST_PARTS) + r")", re.I)
LOCAL = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+(?:/[^\s\"']+)+")
SECRET = re.compile(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{20,}")
TOKEN = re.compile(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})")
FORBIDDEN_PATH = re.compile(
    r"(?:^|/)(?:\.claude|\.codex|\.agents|\.aws|\.ssh|archive|archives|logs?|cache|\.cache)(?:/|$)|"
    r"^(?:ref|plan|plans|code|stages)(?:/|$)|(?:^|/)chapter/(?:2[6-9]|[3-9][0-9]+)\.tex$",
    re.I,
)
SECRET_NAME = re.compile(r"(^|/)(\.env(?:\.|$)|id_(?:rsa|ed25519)$|.*\.(?:pem|key|p12|pfx)$)", re.I)


def tracked_entries(root):
    proc = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "-z"], capture_output=True, check=True)
    entries = {}
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        meta, name = raw.split(b"\t", 1)
        mode = meta.split(b" ", 1)[0].decode("ascii")
        entries[pathlib.PurePosixPath(name.decode()).as_posix()] = mode
    return entries


def check(root):
    manifest = root / "public-files.txt"
    if not manifest.is_file():
        return ["public-files.txt is missing"]
    listed = set()
    errors = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        path = pathlib.PurePosixPath(item)
        if path.is_absolute() or ".." in path.parts or "\\" in item or item.startswith(".git/"):
            errors.append(f"public-files.txt:{line_no}: unsafe path {item!r}")
        if item in listed:
            errors.append(f"public-files.txt:{line_no}: duplicate path {item}")
        listed.add(item)
    try:
        entries = tracked_entries(root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return errors + [f"Cannot enumerate Git tracked files: {exc}"]
    files = set(entries)
    for path in sorted(listed - files):
        errors.append(f"Manifest path is not a tracked file: {path}")
    for path in sorted(files - listed):
        errors.append(f"Tracked file is absent from public-files.txt: {path}")
    for path, mode in sorted(entries.items()):
        if FORBIDDEN_PATH.search(path):
            errors.append(f"Tracked private/retired path is forbidden: {path}")
        if SECRET_NAME.search(path):
            errors.append(f"Tracked secret-like filename: {path}")
        if mode == "120000":
            errors.append(f"Tracked symlink is forbidden: {path}")
        if mode == "160000":
            errors.append(f"Tracked submodule is forbidden: {path}")
        full = root / path
        try:
            payload = full.read_bytes()
        except OSError:
            continue
        if b"\0" in payload[:4096]:
            continue
        text = payload.decode("utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), 1):
            if URL.search(line):
                errors.append(f"Private/nonpublic URL candidate: {path}:{number}")
            if LOCAL.search(line):
                errors.append(f"Absolute local path candidate: {path}:{number}")
            if SECRET.search(line) or TOKEN.search(line) or ("-----BEGIN " + "PRIVATE KEY-----") in line:
                errors.append(f"Credential-like value candidate: {path}:{number}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    errors = check(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Public tracked-file allowlist and residue scan: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
