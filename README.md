# FTQC 教科書・第1〜25章

誤り耐性量子計算（FTQC）を、表面符号の基礎から回路レベル lattice surgery、d=3 統合シミュレータまで段階的に学ぶ教材です。書籍本文は `book/`、実装と実行例は各 Stage にあります。

ビルド済みの [書籍 PDF（第1〜25章・付録A〜D）](book/main.pdf) を収録しています。

## Stage と章

| 対応章 | Stage | 内容 | Python プロジェクト |
|---|---|---|---|
| 1–5 | 基礎 | FTQC・誤り訂正の理論 | — |
| 6–13 | 01 MPP | 現象論的 Pauli product 測定、論理演算、QPD | `src/stages/01_mpp/` |
| 14–19 | 02 Circuit Level | ancilla syndrome 回路、hook error、回路レベル merge | `src/stages/02_circuit_level/` |
| 20–25 | 03 d=3 Integrated | 統合ランタイム、Clifford+T/QPD、Stim/tsim、可視化 | ルート |

各 Stage は独立した Python 環境です。3つの環境には同名の `ftqc` パッケージがあるため、Stage 01・02 の環境をルート環境へ同時にインストールしないでください。Stage 03 は符号距離 d=3 と9 data qubits に固定です。

## 必要な環境

- Python 3.12 以降
- [uv](https://docs.astral.sh/uv/)（`uv.lock` による依存固定）
- 書籍 PDF の生成には LuaLaTeX と `latexmk`

## 実行

ルートの Stage 03:

```bash
uv sync --locked
uv run --locked pytest -q tests
uv run --locked python examples/ch25_d3_benchmark.py
```

Stage 01・02 はそれぞれ独立して実行します:

```bash
uv run --directory src/stages/01_mpp --locked pytest -q
uv run --directory src/stages/01_mpp --locked python examples/ch06_surface_patch.py

uv run --directory src/stages/02_circuit_level --locked pytest -q
uv run --directory src/stages/02_circuit_level --locked python examples/ch16_circuit_merge.py
```

書籍のビルド:

```bash
cd book
latexmk -lualatex main.tex
```

最新の記録では、ルート Stage 03 は38件（5.41秒）、Stage 01 は15件（6.93秒）、Stage 02 は13件（6.94秒）が合格しています。書籍 PDF は278ページでビルド成功し、未解決の参照・引用はありません。その他の検査と各テストの対象は[検証記録](docs/validation.md)を参照してください。

### 書籍図の依存

Stage 03 の通常環境には、Stim 回路図を PDF 化する追加依存は含めません。図を再生成する場合だけ figures group を追加します。

```bash
uv sync --locked --group figures
uv run --locked --group figures python book/figures/gen_ch24_timeline.py --pdf
```

## 設計・検証

- [アーキテクチャと公開 API](docs/architecture.md)
- [ノイズモデルとデコード](docs/noise-and-decoding.md)
- [Stage 間の違い](docs/stage-differences.md)
- [検証項目と結果の読み方](docs/validation.md)
- [適用範囲と制限](docs/limitations.md)
- [再現手順](docs/reproduction.md)
- [参考文献](docs/references.md)
- [出典と権利に関する注記](NOTICE.md)

## ライセンス

コードと書籍本文のライセンスは、それぞれ `LICENSE` と `LICENSE-BOOK` を参照してください。コードは MIT、本文と教材図は CC BY 4.0 です。引用文献と外部依存はそれぞれの権利者・ライセンスに従います。
