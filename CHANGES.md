# 変更履歴

## 公開版 第1〜25章

- Stage 01〜03 の独立した配置、実行環境、章対応を説明。
- 公開 Stage 03 の実装範囲、データフロー、ノイズ/復号モデル、限界、再現手順を文書化。
- 公開用の出典・引用・貢献ガイドを追加。
- Stage 03 は d=3 固定。距離一般化や完全な X/Y/Z 相関残差の実装を含めない。
- 公開 Stage 03 の可視化コードを flat `ftqc.viz` package import に統一し、欠けていた LSCnotFT/LSTeleportFT/CircuitLevelMerge のマッピング対応を復旧。
- 統合 report の未実装 `t_qec_mode` と optional T correction を `ValueError` で明示的に拒否。裸の S/SDG は generic memory shell に対応。
- 公開パッケージ検査、各 Stage のテスト、図・Python 配布物を検証。書籍 PDF は修正中。

検証の実測値と未完了の書籍 PDF ビルドは [docs/validation.md](docs/validation.md) に記録しています。
