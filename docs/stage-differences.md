# Stage 01〜03 の違い

| Stage | 対応章 | モデルと主題 | 実行場所 |
|---|---|---|---|
| 01 MPP | 6–13 | phenomenological MPP syndrome、符号パッチ、論理演算、QPD | `src/stages/01_mpp/` |
| 02 Circuit Level | 14–19 | ancilla syndrome extraction、4-step CX、hook error、回路レベル merge/CNOT | `src/stages/02_circuit_level/` |
| 03 d=3 Integrated | 20–25 | ランタイム統合、trajectory、Statevector/Stim/tsim、レポートと対応範囲内の物理マッピング可視化 | ルート |

Stage 02 は Stage 01 の教材を継承し、個々の回路レベル操作へ進みます。Stage 03 はそれらの機能を含む統合スナップショットです。Stage 03 は d=3 固定で、より一般の距離や後続段階の API を提供しません。

公開パッケージのマッピング可視化は、Stage 03 に存在する `CircuitLevelMerge`、`LSCnotFT`、`LSTeleportFT` builder に限って lattice-surgery レイアウトを再構築します。裸の S/SDG は専用 LS-S gadget ではなく generic memory shell を使い、統合レポートの `t_qec_mode` は `memory` のみです。他の値は未対応機能として `ValueError` になります。

それぞれに独立した `ftqc` パッケージがあるため、同じ Python 環境へ複数 Stage をインストールすると import 元が曖昧になります。表のプロジェクトディレクトリごとに `uv` コマンドを実行してください。
