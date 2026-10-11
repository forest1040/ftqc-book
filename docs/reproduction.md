# 再現手順

Python 3.12 以降と uv を用意し、リポジトリのルートで Stage 03 を同期します。`--locked` により `uv.lock` と一致しない解決を防ぎます。

```bash
uv sync --locked
uv run --locked pytest -q tests
uv run --locked python examples/ch25_d3_benchmark.py
```

12件に絞った Stage 03 の設計不変条件・API契約テストは次のコマンドで実行できます。現在の実測記録は12 passed、4.62秒です。

```bash
uv run --locked pytest -q \
  tests/test_ls_invariants.py \
  tests/test_magic_state.py \
  tests/test_runtime_reproducibility.py \
  tests/test_tsim_ideal.py
```

Stage 01 と Stage 02 は別々の環境で同期・実行してください。

```bash
uv sync --directory src/stages/01_mpp --locked
uv run --directory src/stages/01_mpp --locked pytest -q
uv run --directory src/stages/01_mpp --locked python examples/ch06_surface_patch.py

uv sync --directory src/stages/02_circuit_level --locked
uv run --directory src/stages/02_circuit_level --locked pytest -q
uv run --directory src/stages/02_circuit_level --locked python examples/ch16_circuit_merge.py
```

公開ファイル、書籍コード参照、Stage 系譜、snippet を確認し、wheel と sdist をビルドするコマンド:

```bash
uv run --locked python tools/check_public_files.py
uv run --locked python tools/check_book_code_refs.py --strict --check-code-map
uv run --locked python tools/check_stage_lineage.py --verify-files
uv run --locked python tools/extract_book_snippets.py --verify
uv build
```

サンプルと確率シミュレーションでは shot 数と seed が結果に影響します。コマンドライン引数がある場合は例の `--help` を確認し、実行条件を保存してください。tsim 経路は Stage 03 の依存に含まれる tsim を使用します。

書籍をビルドするには LuaLaTeX と latexmk が必要です。

```bash
cd book
latexmk -lualatex main.tex
```

実行レポートを生成するにはルートから次を実行します。Stage 03 は `t_qec_mode=memory` のみをサポートします。

```bash
uv run --locked python src/ftqc/viz/ftqc_run_report.py \
  --gates "H0 T0 CNOT0,1" --n-logical 2 \
  --cnot-mode lattice_surgery --t-qec-mode memory \
  --out-html /tmp/ftqc-run-report.html
```

第24章の SVG/HTML と PDF を再生成するには、追加の変換依存を同期します。

```bash
uv sync --locked --group figures
uv run --locked --group figures python book/figures/gen_ch24_timeline.py --pdf
```

再現結果を報告する際は Python/uv/OS、実行した環境(Stage)、正確なコマンド、終了コード、seed/shots を併記します。失敗や未実行の項目を合格と記載しません。
