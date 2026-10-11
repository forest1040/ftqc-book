#!/usr/bin/env python3
"""第24章の Stim タイムライン図 (fig:ch24_stim_timeline) を生成する。

実行方法（リポジトリルートから）:
    uv run --locked python book/figures/gen_ch24_timeline.py
    uv run --locked --group figures python book/figures/gen_ch24_timeline.py --pdf

生成物: book/figures/ch24_stim_timeline.pdf

図の中身は stim_timeline_visualizer.py の実出力そのものです。加工はしていません。
（以前は要約行が tick 番号と重なるため SVG から取り除いていましたが、
render_timeline_svg の top_h を 54 -> 68 に直して重なりを解消したため不要になりました。）
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
VIS = ROOT / "src" / "ftqc" / "viz" / "stim_timeline_visualizer.py"

TITLE = "MPP 2-patch timeline: H, CNOT, H, CNOT"


def build_circuit(stim_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from ftqc.clifford_ft_full import CliffordFTCompiler

    compiler = CliffordFTCompiler(n_logical=2, p=0.005, tail_rounds=1)
    layers = [[("H", 0, None)], [("CNOT", 0, 1)],
              [("H", 0, None)], [("CNOT", 1, 0)]]
    circuit, _ = compiler.compile(layers)
    stim_path.write_text(str(circuit), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true",
                    help="svglib + reportlab で SVG を PDF へ変換する")
    args = ap.parse_args()

    stim_path = OUT_DIR / "ch24_stim_timeline.stim"
    svg_path = OUT_DIR / "ch24_stim_timeline.svg"

    build_circuit(stim_path)
    subprocess.run(
        [sys.executable, str(VIS),
         "--stim-file", str(stim_path),
         "--tick-policy", "auto",
         "--stride", "9",
         "--title", TITLE,
         "--out-svg", str(svg_path),
         "--out-html", str(OUT_DIR / "ch24_stim_timeline.html"),
         "--out-json", str(OUT_DIR / "ch24_stim_timeline.json")],
        check=True,
    )

    if args.pdf:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(OUT_DIR / "ch24_stim_timeline.pdf"))
        print(f"wrote {OUT_DIR / 'ch24_stim_timeline.pdf'}")


if __name__ == "__main__":
    main()
