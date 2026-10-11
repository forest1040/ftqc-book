"""Behavioral checks for the public repository's layout and validation tools."""
import importlib.util
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_stage_lineage_and_import_boundaries():
    checker = load_tool("check_stage_lineage")
    assert checker.check(ROOT, verify_files=False) == []


def test_stage_lineage_finds_undeclared_file_in_stage_03_fixture():
    checker = load_tool("check_stage_lineage")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        dirs = {
            "01": root / "src/stages/01_mpp",
            "02": root / "src/stages/02_circuit_level",
            "03": root,
        }
        metas = {
            "01": {"id": "01", "name": "mpp", "inherits": "", "added": ["base.py"], "changed": [], "unchanged": [], "removed": []},
            "02": {"id": "02", "name": "circuit_level", "inherits": "01_mpp", "added": ["new.py"], "changed": [], "unchanged": ["base.py"], "removed": []},
            "03": {"id": "03", "name": "integrated", "inherits": "02_circuit_level", "is_canonical": True, "added": [], "changed": [], "unchanged": ["base.py", "new.py"], "removed": []},
        }
        roots = {}
        for sid, directory in dirs.items():
            src = directory / "src/ftqc"
            src.mkdir(parents=True)
            roots[sid] = src
            meta = {"source_root": "src/ftqc", "forbidden_import_prefixes": [], **metas[sid]}
            lines = [f"{key} = {json.dumps(value)}" for key, value in meta.items()]
            (directory / "STAGE.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (roots["01"] / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
        (roots["02"] / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
        (roots["02"] / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
        (roots["03"] / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
        (roots["03"] / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
        (roots["03"] / "unlisted.py").write_text("VALUE = 3\n", encoding="utf-8")
        errors = checker.check(root, verify_files=True)
    assert any("Stage 03 has undeclared source files" in error for error in errors)


def test_book_code_map_points_to_public_chapters_and_sources():
    checker = load_tool("check_book_code_refs")
    assert checker.check(ROOT) == []


def test_snippet_parser_rejects_crossed_markers():
    extractor = load_tool("extract_book_snippets")
    with tempfile.TemporaryDirectory() as tmp:
        fixture = pathlib.Path(tmp) / "crossed.py"
        fixture.write_text(
            "# book-snippet:first-start\n"
            "value = 1\n"
            "# book-snippet:second-end\n",
            encoding="utf-8",
        )
        snippets, errors = extractor.extract_snippets_from_file(fixture)
    assert snippets == {}
    assert any("Mismatched/crossed snippet end tag 'second'" in error for error in errors)
    assert any("never closed" in error for error in errors)


def test_stage_discovery_does_not_fall_back_to_private_layout():
    stage_cli = load_tool("book_stage")
    found = stage_cli.stages(ROOT)
    assert set(found) == {"01", "02", "03"}
    assert found["01"][0] == ROOT / "src/stages/01_mpp"
    assert found["02"][0] == ROOT / "src/stages/02_circuit_level"
    assert found["03"][0] == ROOT


def _git_fixture(root):
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True, capture_output=True)


def test_public_manifest_rejects_extra_tracked_file():
    checker = load_tool("check_public_files")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "README.md").write_text("Public project.\n", encoding="utf-8")
        (root / "public-files.txt").write_text("README.md\npublic-files.txt\n", encoding="utf-8")
        _git_fixture(root)
        assert checker.check(root) == []
        (root / "extra.txt").write_text("Unlisted file.\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "extra.txt"], check=True, capture_output=True)
        assert any("absent from public-files.txt: extra.txt" in error for error in checker.check(root))


def test_public_manifest_rejects_dangerous_path_and_residue():
    checker = load_tool("check_public_files")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "README.md").write_text("Public project.\n", encoding="utf-8")
        private_dir = root / ".codex"
        private_dir.mkdir()
        (private_dir / "settings.json").write_text("{}\n", encoding="utf-8")
        credential = "ghp" + "_" + "A" * 36
        local_path = "/Users/" + "reader" + "/work3/private.txt"
        (root / "notes.txt").write_text(f"credential = '{credential}'\npath = '{local_path}'\n", encoding="utf-8")
        (root / "public-files.txt").write_text("README.md\npublic-files.txt\n.codex/settings.json\nnotes.txt\n", encoding="utf-8")
        _git_fixture(root)
        errors = checker.check(root)
    assert any("private/retired path" in error for error in errors)
    assert any("Credential-like value" in error for error in errors)
    assert any("Absolute local path" in error for error in errors)
