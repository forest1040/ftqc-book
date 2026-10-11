# FTQC 教科書の編集とビルド

書籍本文は `main.tex` と `chapter/` にあります。`code-map.toml` は公開版の章・付録と実装 Stage の対応を記録します。コード例はリポジトリルートの Stage 03、または `src/stages/01_mpp/` と `src/stages/02_circuit_level/` の独立環境に対応します。

LuaLaTeX と latexmk を用意し、このディレクトリでビルドします。

```bash
latexmk -lualatex main.tex
```

本文は第I部から第V部までの第1〜25章と、付録A〜Dで構成します。Stage 01 と Stage 02 は個別の依存環境を持ちます。Stage 03 はリポジトリ直下の `src/ftqc` を正本とします。
