# 検証範囲と実行結果

## 最新の確認結果

| 対象 | 結果 |
|---|---|
| ルート Stage 03 テスト | 38 passed、5.41秒 |
| Stage 01 テスト | 15 passed、6.93秒 |
| Stage 02 テスト | 13 passed、6.94秒 |
| 書籍参照・コード対応・Stage lineage・公開ファイル scan・snippet 検査 | 成功 |
| 第24章 Stim 図の PDF 再生成 | 成功 |
| Python wheel と sdist のビルド | 成功 |
| 書籍本体の PDF ビルド | 成功、278ページ。未解決参照・引用なし |

Stage 03 で追加した4ファイルの12件に絞った検証も、12 passed、4.62秒で合格しています。これは下記の詳細ケースを個別に確認するための対象 suite です。

再実行コマンド:

```bash
uv run --locked pytest -q \
  tests/test_ls_invariants.py \
  tests/test_magic_state.py \
  tests/test_runtime_reproducibility.py \
  tests/test_tsim_ideal.py
```

| テストファイル / ケース | 実際に確認すること | 確認しないこと |
|---|---|---|
| `test_ls_invariants.py`: `test_ls_cnot_is_deterministic_and_distance_three` | 無雑音 CNOT 回路の detector 決定論性、生成 DEM の `shortest_graphlike_error()` が3 | ungraphlike 機構を含む厳密な距離・FT 証明 |
| 同: `test_ls_cnot_computational_truth_table` | 無雑音時の4つの計算基底入力に対する CNOT 真理値表 | 任意の重ね合わせ・雑音下の論理誤り率 |
| 同: `test_handoff_shells_are_gauge_free_and_distance_three` | S/CNOT/T-injection shell の無雑音 gauge-free 性と graphlike shortest error 3 | 全 fault 集合に対する fault-tolerance 証明 |
| `test_magic_state.py`: `test_15_to_1_code_has_distance_three_and_35_weight_three_logicals` | 15-to-1 検査行列の CSS 性、論理量子ビット1個、weight 1/2 logical 0個と weight 3 logical 35個 | 確率的蒸留の出力誤り率を有限 shots で推定すること |
| 同: `test_ideal_t_teleportation_matches_t_gate` | 理想状態ベクトル上の T teleportation が成功を返すこと | 物理注入回路・ノイズ下のテレポート性能 |
| `test_runtime_reproducibility.py`: `test_runtime_repeats_for_same_seed_and_reports_all_shots` | `qec_mode="off"` の H-T-H、同一 seed/shot 数で測定カウントが一致し、総数600となること | QEC 有効時や全バックエンド、異なる依存版での bitwise 再現性 |
| 同: `test_report_builder_returns_complete_html` | Report builder が HTML 文書を返し入力ゲート名を含むこと | ブラウザ描画・視覚的正しさ |
| 同: `test_report_rejects_unimplemented_t_qec_modes` | report builder が `t_qec_mode="lattice_surgery"` を `ValueError` で拒否 | 他の無効入力すべてのエラー文面 |
| 同: `test_bare_s_report_uses_the_runtime_memory_handoff` | 裸の S が LS builder なしの generic memory shell として report 化されること | S が専用 lattice-surgery shell を通ること |
| 同: `test_tsim_wrapper_restores_runtime_and_backend_factory` | 正常終了時の `run_tsim` が factory と runtime の設定を復元し、backend を tsim と報告すること | 例外時・並列呼び出し時の全復元条件 |
| `test_tsim_ideal.py`: `test_ideal_hth_distribution` | tsim の H-T-H の0出力確率が解析値から0.04以内 | 高精度な分布推定 |
| 同: `test_ideal_bell_correlations_survive_t_and_cnot` | tsim の T/CNOT 回路で Bell 相関が0.97超、00確率が0.5から0.04以内 | ノイズ付き QEC runtime の Bell fidelity |

距離項目は Stim の graphlike DEM に対する指標です。該当テストは `ignore_ungraphlike_errors=True` を指定していませんが、それでも最短 graphlike error の確認だけで全 fault 集合を尽くすものではありません。統計的な誤り率には shot 数と信頼区間が必要です。

書籍は `latexmk -g -lualatex main.tex` でビルドしました。第6章に1.44ptのoverfull警告が残ります。これは組版の警告で、未解決の参照・引用やビルドエラーはありません。
