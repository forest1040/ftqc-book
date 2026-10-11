# アーキテクチャと API

## 3つの実装段階

ルート `src/ftqc/` は第20〜25章の Stage 03 スナップショットです。第6〜13章の Stage 01 と第14〜19章の Stage 02 は `src/stages/` 以下の別プロジェクトに置き、各自の `pyproject.toml` と lock file で依存を管理します。いずれも `ftqc` という import 名を使うため、実行環境を混ぜません。

## Stage 03 の構成

`ftqc.hybrid_ftqc_runtime.HybridFTQCRuntime` が入力回路を解析し、Clifford チャンク、特殊ゲート、lattice-surgery CNOT イベントに分割します。Clifford チャンクは `CliffordFTCompiler` が Stim 回路にし、検出器サンプルをデコードします。特殊ゲートでは状態バックエンド上で論理状態を更新し、対応する QEC shell の残差を軌跡へ反映します。CNOT は設定に応じて transversal または circuit-level lattice surgery の経路を通ります。

主な公開 API と入出力は次のとおりです。

| API | 主な引数・戻り値 | 用途・制約 |
|---|---|---|
| `HybridFTQCRuntime(n_logical, p_phys=..., backend=..., qec_mode=..., nonclifford_mode=..., cnot_mode=...)` | `plan(gates) → PlanEvent[]`; `run(gates, shots, seed) → HybridRunResult` | Clifford+T のイベント計画と Monte Carlo 軌跡。QEC は `required/optional/off`、非 Clifford は `statevector/qpd`、CNOT は `transversal/lattice_surgery`。 |
| `CliffordFTCompiler(n_logical, p, tail_rounds)` | `compile(layers, noise) → (stim.Circuit, observable_records)`; `decode(circ, shots) → (logical_error_rates, raw_rates, decoder)` | Clifford ゲート列を detector 付き回路へ変換。graphlike DEM は MWPM、非対応 DEM は利用可能なら BP+OSD に渡します。 |
| `build_ls_s_shell`, `build_ls_cnot_shell`, `build_ls_t_injection_shell` | noise/確率・merge/split rounds 等を受け `ShellCircuit` を返す。`decode_shell(shell, shots) → ShellDecodeResult` | handoff 用回路の構築と残差観測量の復号。gauge detector を許可しません。 |
| `LSCnotFT().build(...)`, `LSTeleportFT(...).build_teleport(...)` | `(circuit, records, ...)` | ancilla ベースの lattice-surgery CNOT とテレポート骨格を作成します。 |
| `build_15to1()`, `code_properties(HX, HZ)`, `distill(HX, HZ, p, shots, seed)` | 検査行列、符号特性、受理率と出力誤り率 | 15-to-1 の組合せ・確率モデル。 |
| `run_tsim(runtime, gates, shots, seed)`, `run_tsim_fast(...)` | `HybridRunResult` | tsim サンプラーを runtime に接続します。呼び出し中は backend factory を一時的に差し替えるため、同一プロセスでの並行呼び出しには使わないでください。 |
| `tsim_logical_distribution(gates, n_logical, shots, seed)` | bit tuple → 頻度の辞書 | QEC を介さない理想回路分布の比較に使います。 |
| `build_ls_assignment(builder)`, `build_mpp_assignment(...)`, `write_bundle(...)` | `AssignmentManifest` と HTML/JSON/SVG | `ftqc.viz.ftqc_mapping_visualizer` の対応済み builder は circuit-level merge、LS-CNOT、LS-T teleportation です。 |
| `build_run_report_html(gates, n_logical, ..., t_qec_mode="memory")` | HTML 文字列 | `ftqc.viz` 内の可視化 API を束ねます。Stage 03 では `t_qec_mode="memory"` と `t_s_correction=True` のみ受け付け、未対応値は `ValueError` です。 |

API の統計値はショット推定です。seed は確率 generator に渡されますが、再現性テストが確認するのは `qec_mode="off"` の指定回路だけです。QEC ありの全経路・異なる依存バージョン間での bitwise 一致は保証しません。

## データの流れ

```text
論理ゲート列
  → plan: Clifford chunk / 特殊ゲート / LS-CNOT event
  → Stim の d=3 QEC 回路をサンプルし、検出器をデコード
  → 復号残差を論理状態へ適用し、状態バックエンドでゲートを進める
  → trajectory と HybridRunResult（分布、イベント、復号情報）
```

Clifford 状態バックエンドは Stim、一般状態ベクトル計算は NumPy または Qulacs を選べます。tsim はサンプラーなので状態ベクトルを返しません。QPD は T/T† を符号付き Clifford 分岐へ展開して再構成分布や期待値を推定します。裸の S/SDG は個別 LS-S 経路ではなく、一般の memory handoff shell を使います。

この設計は単一の物理回路シミュレーションではなく、物理 QEC 回路から得た残差を論理状態更新へ渡すハイブリッド模型です。詳しい仮定は [ノイズモデルとデコード](noise-and-decoding.md) と [制限](limitations.md) を参照してください。
