# Stage 01: MPP（第6〜13章）

このプロジェクトは、表面符号パッチ、現象論的 MPP syndrome 測定、論理 Clifford 操作、lattice-surgery 幾何、T ゲートの魔法状態蒸留、QPD、最小ハイブリッドランタイムを扱います。

## 実行

Python 3.12 以降と uv が必要です。このディレクトリを作業場所として環境を作成します。

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python examples/ch06_surface_patch.py
```

この Stage の import 元はこのディレクトリの `src/ftqc/` です。ルート Stage 03 と Stage 02 にも `ftqc` パッケージがあるため、同じ環境へ複数 Stage をインストールしないでください。

## モジュール

- `surface_patch`, `clifford_ft_full`: パッチと論理 Clifford
- `geom_lattice_surgery`, `lattice_surgery`, `ls_handoff_shells`: 幾何と phenomenological MPP shell
- `t_gate`, `qpd_backend`: 蒸留模型と符号付き Clifford 分解
- `hybrid_ftqc_runtime`: 基礎的な統合ランタイム

Stage 01 は circuit-level ancilla noise、hook error の回路、4-step CX、回路レベル merge/CNOT を対象にしません。これらは第14〜19章の教材で扱います。
