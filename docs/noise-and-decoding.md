# ノイズモデルとデコード

## Stage によるノイズ表現

Stage 01 は ancilla を含む物理回路を展開せず、MPP による syndrome 測定と phenomenological noise を扱います。Stage 02 は syndrome extraction ancilla、4-step CX スケジュール、ゲート・測定を含む circuit-level noise を導入します。Stage 03 の Clifford compiler と lattice-surgery shell も Stim の回路レベルモデルを用います。

回路生成器の `p` は物理誤り確率、`noise=False` は無雑音参照回路です。具体的な誤り位置・確率は各 builder と compiler の実装が定義します。すべてのゲートで一様な物理モデルだと解釈せず、対象回路の生成コードを確認してください。

## 検出器と復号

Stim は検出器サンプルと detector error model (DEM) を提供します。コードは graphlike DEM では PyMatching の MWPM を使い、非 graphlike DEM では `stimbposd` の BP+OSD へ切り替えます。Stage 03 の `stimbposd` は通常依存に含まれます。Shell API は `allow_gauge_detectors=False` を使います。無雑音時の非決定論的検出器はモデルの不整合として例外になります。

`shortest_graphlike_error` は DEM の graphlike 誤り集合についての指標です。`ignore_ungraphlike_errors=True` を指定する経路では非 graphlike な機構を無視して計算するため、その値だけで全 fault の距離や fault tolerance を証明できません。検証とその読み方は [validation.md](validation.md) を参照してください。

## 論理状態への残差反映

Stage 03 の基本 compiler が返す観測量反転は `X_L` 残差として状態へ適用されます。`mirror_z_residuals=True` は同じ発生率を使う独立な近似 `Z_L` チャネルを加えるストレス設定で、物理モデルから導出された相関誤りではありません。`Y_L` と完全な Pauli 相関残差フレームは実装されていません。

T ゲートの状態更新には NumPy/Qulacs の状態ベクトル、または QPD の符号付き Clifford 分岐を用います。Stim の安定化子回路内で魔法状態そのものを表現するものではありません。蒸留誤り模型と QEC shell の誤りは別にモデル化されます。
