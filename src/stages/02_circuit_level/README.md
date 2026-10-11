# Stage 02: Circuit Level（第14〜19章）

Stage 01 の phenomenological MPP から進み、ancilla syndrome extraction、4-step CX スケジュール、hook error の伝播、circuit-level merge、lattice-surgery CNOT、handoff shell を学びます。

## 実行

Python 3.12 以降と uv が必要です。このディレクトリの独立環境を使用します。

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked python examples/ch16_circuit_merge.py
```

import 元はこのディレクトリ内の `src/ftqc/` です。ルートと Stage 01 にも同名の `ftqc` があるため、他 Stage と環境を共有しないでください。

## 主なモジュール

- `circuit_level_merge`: seam ancilla による回路レベル merge
- `ls_cnot_ft`: 回路レベル LS-CNOT と T teleportation shell
- `ls_handoff_shells`: detector 付き handoff 回路と decoder
- `hybrid_ftqc_runtime`: Stage 01 から受け継いだ runtime に回路レベル経路を接続

`circuit_level_ls` は旧 API の退役スタブです。Stage 02 では任意距離一般化を行わず、Stage 03 の統合 trajectory/backend/レポート機能も含みません。
