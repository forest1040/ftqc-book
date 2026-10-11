# 貢献方法

報告や変更は、第1〜25章と Stage 01〜03 の公開範囲に沿ってください。再現可能な不具合報告には、Stage、Python/uv の版、実行コマンド、seed と shots、期待値と実際の結果を含めてください。秘密情報や非公開資料は投稿しないでください。

## 開発環境

Stage 03:

```bash
uv sync --locked
uv run --locked pytest -q tests
```

Stage 01・02 はそれぞれのプロジェクトで実行します。

```bash
uv sync --directory src/stages/01_mpp --locked
uv run --directory src/stages/01_mpp --locked pytest -q
uv sync --directory src/stages/02_circuit_level --locked
uv run --directory src/stages/02_circuit_level --locked pytest -q
```

書籍を変更した場合は `book/` で `latexmk -lualatex main.tex` を実行し、未解決引用・参照を確認してください。結果を説明するときは、未実行の確認を成功と表現せず、変更に関連するコマンドと結果を記載してください。

## 変更の範囲

Stage 01・02 の同名 `ftqc` パッケージは独立環境を維持します。Stage 03 の距離制約、残差モデル、ノイズ/復号上の制限を暗黙に広げず、API やモデルを変える場合は対応する本文・docs・例も更新してください。新しい外部依存を追加するときは、用途とライセンスを明記し lock file を更新してください。

貢献を提出することで、本リポジトリに適用されるライセンスの下で配布する権利がある素材を提供することを確認してください。
