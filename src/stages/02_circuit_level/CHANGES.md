# Stage 02 の構成

第14〜19章に対応する回路レベル教材を、独立した `ftqc` パッケージとして収録しています。Stage 01 のパッチ・論理演算を引き継ぎ、seam ancilla による circuit-level merge と LS-CNOT、detector 付き handoff shell を加えます。

`circuit_level_ls` は互換用の退役スタブで、実装は `circuit_level_merge` と `ls_cnot_ft` にあります。Stage 03 と共通環境を作らないでください。環境構築、実行方法、モデルの範囲は [README](README.md) を参照してください。
