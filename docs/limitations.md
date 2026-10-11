# 適用範囲と制限

- Stage 03 は符号距離 d=3、各パッチ9 data qubits を対象とします。任意の d へ一般化した API ではありません。
- ベースの residual model は `X_L` チャネルです。オプションの `mirror_z_residuals` は同率の独立 `Z_L` を加える近似で、Y-only の物理チャネルや X/Z 相関を表しません。
- Stim は安定化子シミュレータです。魔法状態の非安定化子状態更新は状態ベクトル backend で行い、Stim は handoff 周辺の QEC 残差を供給します。従って end-to-end の物理 injection simulation ではありません。
- `run_tsim` と `run_tsim_fast` は呼出し中に runtime module の `make_backend` を一時的に差し替え、終了時に復元する仕組みです。並行呼び出しや割り込み時の隔離に注意が必要な実装上の制限があります。
- 統合 HTML report API は `t_qec_mode="memory"` と `t_s_correction=True` だけを実装します。未対応値は明示的に `ValueError` になります。裸の S/SDG ゲートは LS-S gadget ではなく一般 memory shell を使います。
- LS CNOT の runtime shell には corner の制限があり、コード内注記では circuit distance 2 とされています。別経路で報告される graphlike 値3と混同しないでください。
- `shortest_graphlike_error` の値は graphlike 機構に関する上界・整合性チェックです。ungraphlike 機構を無視する指定や有限範囲の探索を、全誤りに対する厳密な距離証明と読み替えないでください。
- T 蒸留の `35p^3` は理想化した低誤り率の leading-order 関係です。有限 shots の数値は統計推定であり、近似の適用域を超える保証ではありません。
- Stage 01 は phenomenological noise、Stage 02/03 の対象回路は circuit-level noise を扱います。結果を同じノイズ配置・誤り確率の比較として直接並べられない場合があります。

この制限一覧は公開された Stage 03 ソースの挙動に基づきます。文書中に記述のない機能は提供を約束しません。
