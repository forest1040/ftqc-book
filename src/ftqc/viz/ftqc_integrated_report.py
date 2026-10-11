#!/usr/bin/env python3
"""
FTQC integrated report — 論理 → 物理 → 実行時系列 を 1 ページに束ねる.

For a single lattice-surgery operation, this assembles one self-contained HTML
page with three stacked views that drill from the abstract to the concrete:

  1. アーキテクチャ / 依存グラフ  (the 7-file stack; the files exercised by the
     chosen operation are highlighted).  -- mirrors chapter 18 fig:deps.
  2. 論理 → 物理 マッピング        (ftqc_mapping_visualizer, ls model):
     which physical data / ancilla / seam qubits implement the operation.
  3. 実行時系列 (tick)             (stim_timeline_visualizer): what each physical
     qubit does, tick by tick.

It reuses the two existing visualizers as libraries and introspects the real
builders, so the three views describe exactly the same circuit.

Example:
  python ftqc_integrated_report.py --ls-kind cnot --src-path ./src \
      --out-html ftqc_report_cnot.html
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 7-file dependency graph (chapter 18 fig:deps), with per-operation highlight.
# ---------------------------------------------------------------------------

# node id -> (label, chapter, x, y)  (grid coords; rendered top-down)
_DEP_NODES: Dict[str, Tuple[str, str, float, float]] = {
    "rt": ("hybrid_ftqc_runtime.py\n(オーケストレータ)", "18-20", 2.0, 0.0),
    "cf": ("clifford_ft_full.py\n(生成子追跡)", "9", 0.0, 1.0),
    "sh": ("ls_handoff_shells.py\n(シェルAPI・decode_shell)", "21", 2.0, 1.0),
    "tg": ("t_gate.py\n(蒸留・Tテレポート)", "14-16", 4.0, 1.0),
    "cn": ("ls_cnot_ft.py\n(LSCnotFT/LSTeleportFT)", "12,13,17", 1.1, 2.0),
    "cm": ("circuit_level_merge.py\n(CircuitLevelMerge)", "11", 2.9, 2.0),
    "gm": ("geom_lattice_surgery.py\n(幾何・GeomMerge)", "7,10", 2.0, 3.0),
}
_DEP_EDGES = [("rt", "cf"), ("rt", "sh"), ("rt", "tg"),
              ("sh", "cn"), ("sh", "cm"), ("cn", "gm"), ("cm", "gm")]

# Which files each operation exercises (for highlighting).
_OP_HIGHLIGHT: Dict[str, Set[str]] = {
    "merge_zz": {"sh", "cm", "gm"},
    "merge_xx": {"sh", "cm", "gm"},
    "cnot": {"sh", "cn", "gm"},
    "t": {"sh", "cn", "tg", "gm"},
    "t_nos": {"sh", "cn", "tg", "gm"},
}


def build_dependency_svg(highlight: Optional[Set[str]] = None, scale: float = 150.0) -> str:
    highlight = highlight or set()
    pad = 30
    box_w, box_h = 230.0, 50.0
    xs = [n[2] for n in _DEP_NODES.values()]
    ys = [n[3] for n in _DEP_NODES.values()]
    W = (max(xs) - min(xs)) * scale + box_w + 2 * pad
    H = (max(ys) - min(ys)) * scale + box_h + 2 * pad

    def cx(gx: float) -> float:
        return pad + (gx - min(xs)) * scale + box_w / 2.0

    def cy(gy: float) -> float:
        return pad + (gy - min(ys)) * scale + box_h / 2.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.0f} {H:.0f}" class="dep-svg">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 z" fill="#64748b"/></marker></defs>',
    ]
    for a, b in _DEP_EDGES:
        x1, y1 = cx(_DEP_NODES[a][2]), cy(_DEP_NODES[a][3]) + box_h / 2.0
        x2, y2 = cx(_DEP_NODES[b][2]), cy(_DEP_NODES[b][3]) - box_h / 2.0
        on = a in highlight and b in highlight
        parts.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{"#ea580c" if on else "#94a3b8"}" stroke-width="{2.4 if on else 1.4}" '
            f'marker-end="url(#arrow)"/>'
        )
    for nid, (label, chap, gx, gy) in _DEP_NODES.items():
        x = cx(gx) - box_w / 2.0
        y = cy(gy) - box_h / 2.0
        on = nid in highlight
        fill = "#fff7ed" if on else "#f8fafc"
        stroke = "#ea580c" if on else "#64748b"
        sw = 2.6 if on else 1.3
        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{box_w:.0f}" height="{box_h:.0f}" rx="9" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        name, sub = label.split("\n", 1)
        parts.append(
            f'<text x="{cx(gx):.0f}" y="{y+20:.0f}" text-anchor="middle" '
            f'font-size="12" font-weight="700" fill="#0f172a">{html.escape(name)}</text>'
        )
        parts.append(
            f'<text x="{cx(gx):.0f}" y="{y+36:.0f}" text-anchor="middle" '
            f'font-size="10" fill="#475569">{html.escape(sub)}  · 第{chap}章</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Builder construction and the three view sources
# ---------------------------------------------------------------------------

def _load_builder(ls_kind: str):
    if ls_kind in ("merge_zz", "merge_xx"):
        from ftqc.circuit_level_merge import CircuitLevelMerge
        return CircuitLevelMerge("ZZ" if ls_kind == "merge_zz" else "XX"), \
            ("CircuitLevelMerge " + ("ZZ" if ls_kind == "merge_zz" else "XX"))
    if ls_kind == "cnot":
        from ftqc.ls_cnot_ft import LSCnotFT
        return LSCnotFT(), "LS CNOT (LSCnotFT)"
    if ls_kind in ("t", "t_nos"):
        from ftqc.ls_cnot_ft import LSTeleportFT
        return LSTeleportFT(with_s=(ls_kind == "t")), \
            ("LS-T injection (with S)" if ls_kind == "t" else "LS-T injection (no S)")
    raise SystemExit(f"unknown --ls-kind: {ls_kind}")


def _builder_circuit(builder, p: float = 0.005):
    """Return a stim.Circuit for the builder, regardless of its build API."""
    if hasattr(builder, "build_teleport"):
        out = builder.build_teleport(p=p, noise=True)
    else:
        out = builder.build(p=p, noise=True)
    return out[0] if isinstance(out, tuple) else out


def _timeline_svg_and_stats(circuit_text: str, tick_policy: str = "auto") -> Tuple[str, Dict[str, int]]:
    from ftqc.viz import stim_timeline_visualizer as tl
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "circuit.stim"
        path.write_text(circuit_text, encoding="utf-8")
        manifest, _meta = tl.build_manifest(str(path), tick_policy=tick_policy)
        svg = tl.render_timeline_svg(manifest)
    stats = {
        "ticks": manifest.num_ticks,
        "qubits": len(manifest.used_qubits),
        "measurements": sum(t.measurement_count for t in manifest.ticks),
        "detectors": sum(t.detector_count for t in manifest.ticks),
        "observables": sum(t.observable_count for t in manifest.ticks),
    }
    return svg, stats


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_HTML_CSS = """
:root { color-scheme: light; }
body { font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       margin:0; color:#1f2937; background:#f1f5f9; }
header { background:#0f172a; color:#fff; padding:26px 34px; }
header h1 { margin:0 0 6px; font-size:24px; }
header p { margin:3px 0; color:#cbd5e1; font-size:14px; }
main { max-width:1280px; margin:0 auto; padding:24px; }
.flow { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:6px 0 0; font-size:13px; color:#cbd5e1; }
.flow b { color:#fff; } .flow .arrow { color:#f59e0b; font-weight:700; }
.card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:18px 20px;
        box-shadow:0 1px 4px rgba(0,0,0,.05); margin:18px 0; }
.card h2 { margin:0 0 4px; font-size:19px; }
.card .lead { color:#475569; font-size:13.5px; margin:0 0 12px; line-height:1.55; }
.step { display:inline-block; width:24px; height:24px; line-height:24px; text-align:center;
        border-radius:50%; background:#0f172a; color:#fff; font-weight:700; margin-right:8px; font-size:13px; }
.stats { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 0; }
.pill { background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px; padding:4px 10px; font-size:12.5px; }
.pill.warn { background:#fff7ed; border-color:#fed7aa; }
.scroll { overflow:auto; border:1px solid #eef2f7; border-radius:10px; padding:10px; background:#fff; }
.legend { display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:12.5px; color:#334155; margin-top:8px; }
.sw { display:inline-block; width:14px; height:14px; border-radius:50%; vertical-align:-2px; margin-right:4px; border:2px solid #222; }
.sw.sq { border-radius:3px; }
.dep-svg, .ftqc-svg { background:#fff; }
footer { text-align:center; color:#94a3b8; font-size:12px; padding:24px; }
a { color:#1d4ed8; }
"""


def build_report_html(ls_kind: str, p: float = 0.005, title: Optional[str] = None) -> str:
    from ftqc.viz import ftqc_mapping_visualizer as fmv

    builder, label = _load_builder(ls_kind)
    title = title or f"FTQC 統合レポート — {label}"

    # (2) logical -> physical mapping
    manifest = fmv.build_ls_assignment(builder, label=label)
    map_svg = fmv.render_svg(manifest)
    md = manifest.metadata

    # (3) execution timeline
    circ = _builder_circuit(builder, p=p)
    tl_svg, tl_stats = _timeline_svg_and_stats(str(circ))

    # (1) architecture / dependency graph
    dep_svg = build_dependency_svg(_OP_HIGHLIGHT.get(ls_kind, set()))

    map_stats = "".join(
        f'<span class="pill">{html.escape(k)}={v}</span>'
        for k, v in (("patches", len(md.get("patches", []))), ("data", md.get("n_data")),
                     ("ancilla", md.get("n_ancilla")), ("seam ancilla", md.get("n_seam_ancilla")))
    )
    tl_stats_html = "".join(
        f'<span class="pill">{k}={v}</span>' for k, v in tl_stats.items()
    )

    map_legend = (
        '<div class="legend">'
        + "".join(
            f'<span><span class="sw" style="background:{fmv.LS_PATCH_COLORS.get(p_,"#e5e7eb")};border-color:#444"></span>patch {html.escape(str(p_))}</span>'
            for p_ in md.get("patches", [])
        )
        + '<span><span class="sw" style="border-color:#1d4ed8"></span>X_L</span>'
        + '<span><span class="sw" style="border-color:#c2410c"></span>Z_L</span>'
        + '<span><span class="sw sq" style="background:#dbeafe;border-color:#555"></span>X ancilla</span>'
        + '<span><span class="sw sq" style="background:#ffedd5;border-color:#555"></span>Z ancilla</span>'
        + '<span><span class="sw sq" style="background:#fee2e2;border-color:#b91c1c"></span>seam ancilla</span>'
        + "</div>"
    )

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>同一の格子手術回路を 3 つの視点で俯瞰します（物理エラー率 p={p}）。</p>
  <div class="flow"><b>論理</b><span class="arrow">→</span><b>物理</b><span class="arrow">→</span><b>実行時系列</b>
    <span>（アーキテクチャ → どの物理量子ビット → tick ごとの動き）</span></div>
</header>
<main>

  <section class="card">
    <h2><span class="step">1</span>アーキテクチャ / 依存グラフ</h2>
    <p class="lead">この操作が使うファイルを橙色で強調しています（7 ファイルの依存は第18章 fig:deps に対応）。
       下層の幾何から上層のオーケストレータへ積み上がります。</p>
    <div class="scroll">{dep_svg}</div>
  </section>

  <section class="card">
    <h2><span class="step">2</span>論理 → 物理 マッピング</h2>
    <p class="lead">論理操作を実装する物理量子ビットの配置です。パッチごとに色分けし、データ（円）・
       シンドロームアンシラ（四角）・seam アンシラ（赤・パッチをまたぐ）を区別します。</p>
    <div class="stats">{map_stats}</div>
    {map_legend}
    <div class="scroll" style="margin-top:10px">{map_svg}</div>
  </section>

  <section class="card">
    <h2><span class="step">3</span>実行時系列（tick）</h2>
    <p class="lead">同じ回路を Stim の timeslice 風に時間方向へ展開したものです。各行が物理量子ビット、
       各列が tick で、初期化・CX・測定・ノイズなどの活動を示します。</p>
    <div class="stats">{tl_stats_html}</div>
    <div class="scroll" style="margin-top:10px">{tl_svg}</div>
  </section>

</main>
<footer>generated by ftqc_integrated_report.py — mapping × timeline × dependency graph</footer>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="論理→物理→実行時系列 を 1 ページに束ねる FTQC 統合レポート")
    ap.add_argument("--ls-kind", choices=["merge_zz", "merge_xx", "cnot", "t", "t_nos"], default="cnot")
    ap.add_argument("--p", type=float, default=0.005, help="物理エラー率")
    ap.add_argument("--src-path", type=str, default=None, help="シミュレータ src/ ディレクトリ")
    ap.add_argument("--vis-path", type=str, default=None,
                    help="2 つの可視化モジュールがあるディレクトリ（既定: このスクリプトと同じ場所）")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out-html", type=str, default="ftqc_integrated_report.html")
    args = ap.parse_args(argv)

    vis_path = args.vis_path or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, vis_path)
    if args.src_path:
        sys.path.insert(0, str(Path(args.src_path).resolve()))

    html_text = build_report_html(args.ls_kind, p=args.p, title=args.title)
    Path(args.out_html).write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
