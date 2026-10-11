#!/usr/bin/env python3
"""
ftqc_run_report — 1 つの run を 4 視点で 1 ページに束ねる統合レポート.

ゲート列と runtime 設定を受け取り、次を 1 つの HTML にまとめます:

  1. アーキテクチャ / 依存グラフ   : この run の全イベントが使うファイルを強調（第18章 fig:deps）。
  2. 論理 → 物理（ベース）         : n_logical 個のデータパッチ（mpp_d3）。
  3. イベント別（論理→物理 + tick）: plan の各イベントについて、LS なら多パッチ
     マッピング、そして実行時系列（tick）を並べる。
  4. T の分岐（QPD）               : T を含む場合、準確率分解の分岐ツリー。

既存の 3 可視化（ftqc_mapping_visualizer / stim_timeline_visualizer /
t_branch_visualizer）と橋渡し 2 モジュール（ftqc_integrated_report /
ftqc_run_visualizer）をライブラリとして合成します。すべて同一の runtime/plan
から生成するので、4 視点は同じ run を指します。

例:
  python ftqc_run_report.py --gates "H0 T0 CNOT0,1" --n-logical 2 \
      --cnot-mode lattice_surgery --t-qec-mode memory \
      --src-path ./src --out-html run_report.html
"""
from __future__ import annotations

import argparse
import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Gate = Tuple[str, int, Optional[int]]


def _tbranch_svg(gates: List[Gate], n_logical: int, shots: int = 20000) -> Optional[str]:
    """t_branch_visualizer で QPD 分岐 SVG を生成し、文字列で返す（T が無ければ None）。"""
    from ftqc.viz import t_branch_visualizer as tb
    from ftqc.viz.ftqc_run_visualizer import has_t, gates_to_tbranch_text
    if not has_t(gates):
        return None
    with tempfile.TemporaryDirectory() as d:
        svg = Path(d) / "tb.svg"
        tb.main(["--gates", gates_to_tbranch_text(gates), "--n-logical", str(n_logical),
                 "--shots", str(shots), "--out-svg", str(svg),
                 "--out-html", str(Path(d) / "tb.html"), "--out-json", str(Path(d) / "tb.json")])
        return svg.read_text(encoding="utf-8") if svg.exists() else None


_CSS = """
body{font-family:ui-sans-serif,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f1f5f9;color:#1f2937;}
header{background:#0f172a;color:#fff;padding:24px 32px;}
header h1{margin:0 0 6px;font-size:23px;} header p{margin:2px 0;color:#cbd5e1;font-size:13.5px;}
.flow{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px;color:#cbd5e1;}
.flow b{color:#fff;} .flow .a{color:#f59e0b;font-weight:700;}
main{max-width:1300px;margin:0 auto;padding:22px;}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;margin:18px 0;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.card h2{margin:0 0 4px;font-size:19px;} .card .lead{color:#475569;font-size:13.5px;margin:0 0 12px;line-height:1.55;}
.badge{display:inline-block;width:25px;height:25px;line-height:25px;text-align:center;border-radius:50%;background:#0f172a;color:#fff;font-weight:700;margin-right:8px;font-size:13px;}
.kind{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:2px 9px;font-size:12px;color:#3730a3;margin-left:6px;vertical-align:2px;}
.event{border:1px solid #e8edf3;border-radius:12px;padding:14px 16px;margin:14px 0;background:#fcfdff;}
.event h3{margin:0 0 8px;font-size:15.5px;}
.sub{font-size:12.5px;font-weight:700;color:#334155;margin:10px 0 4px;}
.stats{display:flex;flex-wrap:wrap;gap:7px;margin:6px 0;}
.pill{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:999px;padding:3px 9px;font-size:12px;}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0;}
.chip{background:#0f172a;color:#fff;border-radius:8px;padding:3px 9px;font-size:12px;}
.scroll{overflow:auto;border:1px solid #eef2f7;border-radius:9px;padding:8px;background:#fff;}
.legend{display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:12px;color:#334155;margin-top:6px;}
.sw{display:inline-block;width:13px;height:13px;border-radius:50%;vertical-align:-2px;margin-right:4px;border:2px solid #222;}
.sw.sq{border-radius:3px;}
footer{text-align:center;color:#94a3b8;font-size:12px;padding:22px;}
"""


def build_run_report_html(
    gates: List[Gate],
    *,
    n_logical: int,
    p_phys: float = 0.005,
    cnot_mode: str = "transversal",
    t_qec_mode: str = "memory",
    nonclifford_mode: str = "statevector",
    t_s_correction: bool = True,
    tick_policy: str = "auto",
    shots: int = 20000,
    title: Optional[str] = None,
) -> str:
    from ftqc.viz import ftqc_mapping_visualizer as fmv
    from ftqc.viz.ftqc_integrated_report import build_dependency_svg
    from ftqc.viz.ftqc_run_visualizer import event_circuits, _timeline_svg_and_stats, has_t, gates_to_tbranch_text
    from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

    if t_qec_mode != "memory":
        raise ValueError("Stage 03 supports t_qec_mode='memory' only")
    if not t_s_correction:
        raise ValueError("Stage 03 does not expose an optional T correction shell")
    rt = HybridFTQCRuntime(n_logical, p_phys=p_phys, cnot_mode=cnot_mode,
                           nonclifford_mode=nonclifford_mode)
    evs = event_circuits(rt, gates)
    title = title or "FTQC run レポート — 1 つの run を 4 視点で"
    gate_str = gates_to_tbranch_text(gates)
    cfg = (f"n={n_logical}, p_phys={p_phys}, cnot_mode={cnot_mode}, "
           f"t_qec_mode={t_qec_mode}, nonclifford_mode={nonclifford_mode}")

    # (1) dependency graph: union of per-event highlights
    highlight = set().union(*[e["highlight"] for e in evs]) if evs else set()
    dep_svg = build_dependency_svg(highlight)

    # (2) base logical -> physical (mpp_d3 data patches)
    cnot_pairs = [(a, b) for (nm, a, b) in gates if nm.upper() == "CNOT" and b is not None]
    base_manifest = fmv.build_mpp_assignment(n_logical, stride=fmv.NDATA, cnot_pairs=cnot_pairs)
    base_map_svg = fmv.render_svg(base_manifest)

    # plan summary chips
    chips = "".join(f'<span class="chip">{html.escape(e["label"])}</span>' for e in evs)

    # (3) per-event cards: LS mapping (if any) + timeline
    cards = []
    for i, e in enumerate(evs, 1):
        parts = [f'<div class="event"><h3><span class="badge">{i}</span>{html.escape(e["label"])}'
                 f'<span class="kind">{html.escape(e["kind"])}</span></h3>']
        if e["ls_builder"] is not None:
            man = fmv.build_ls_assignment(e["ls_builder"], label=e["label"])
            md = man.metadata
            ls_svg = fmv.render_svg(man)
            mp = "".join(f'<span class="pill">{k}={v}</span>' for k, v in
                         (("patches", len(md.get("patches", []))), ("data", md.get("n_data")),
                          ("ancilla", md.get("n_ancilla")), ("seam", md.get("n_seam_ancilla"))))
            leg = ("".join(
                f'<span><span class="sw sq" style="background:{fmv.LS_PATCH_COLORS.get(pp,"#e5e7eb")};border-color:#444"></span>{html.escape(str(pp))}</span>'
                for pp in md.get("patches", []))
                + '<span><span class="sw sq" style="background:#fee2e2;border-color:#b91c1c"></span>seam</span>')
            parts.append('<div class="sub">論理 → 物理（マッピング）</div>'
                         f'<div class="stats">{mp}</div><div class="legend">{leg}</div>'
                         f'<div class="scroll" style="margin-top:6px">{ls_svg}</div>')
        tl_svg, st = _timeline_svg_and_stats(str(e["circuit"]), tick_policy=tick_policy)
        tlp = "".join(f'<span class="pill">{k}={v}</span>' for k, v in st.items())
        parts.append('<div class="sub">実行時系列（tick）</div>'
                     f'<div class="stats">{tlp}</div>'
                     f'<div class="scroll" style="margin-top:6px">{tl_svg}</div></div>')
        cards.append("".join(parts))
    events_html = "\n".join(cards)

    # (4) QPD branch
    qpd_section = ""
    tb_svg = _tbranch_svg(gates, n_logical, shots=shots)
    if tb_svg:
        qpd_section = (
            '<section class="card"><h2><span class="badge">4</span>T の分岐（QPD）</h2>'
            '<p class="lead">nonclifford の各 T を準確率分解（½ I + (1/√2) S − ((√2−1)/2) Z）した'
            '符号付きクリフォード分岐です。S 分岐は LS-S シェルへ回されます。</p>'
            f'<div class="scroll">{tb_svg}</div></section>'
        )

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>gates: <code>{html.escape(gate_str)}</code></p>
  <p>{html.escape(cfg)}</p>
  <div class="flow"><b>論理</b><span class="a">→</span><b>物理</b><span class="a">→</span><b>実行時系列</b>
    <span class="a">→</span><b>分岐</b><span>（同一 run を 4 視点で）</span></div>
</header>
<main>

  <section class="card">
    <h2><span class="badge">1</span>アーキテクチャ / 依存グラフ</h2>
    <p class="lead">この run の全イベントが使うファイルを橙色で強調しています（第18章 fig:deps）。</p>
    <div class="scroll">{dep_svg}</div>
  </section>

  <section class="card">
    <h2><span class="badge">2</span>論理 → 物理（ベース）</h2>
    <p class="lead">{n_logical} 個の論理量子ビットのデータパッチ（mpp\\_d3, 各 9 データ）。各イベントの
       格子手術レイアウトは下のイベント別カードに示します。</p>
    <div class="scroll">{base_map_svg}</div>
  </section>

  <section class="card">
    <h2><span class="badge">3</span>イベント別（論理→物理 + 実行時系列）</h2>
    <p class="lead">plan が分解したイベントです。LS イベントは多パッチのマッピングと tick を、
       クリフォード/メモリは tick のみを示します。</p>
    <div class="chips">{chips}</div>
    {events_html}
  </section>

  {qpd_section}

</main>
<footer>generated by ftqc_run_report.py — dependency × mapping × timeline × T-branch</footer>
</body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="1 つの run を 4 視点で 1 ページに束ねる FTQC 統合レポート")
    ap.add_argument("--gates", required=True, help="ゲート列。例: 'H0 T0 CNOT0,1'")
    ap.add_argument("--n-logical", type=int, default=None)
    ap.add_argument("--p-phys", type=float, default=0.005)
    ap.add_argument("--cnot-mode", choices=["transversal", "lattice_surgery"], default="transversal")
    ap.add_argument("--t-qec-mode", choices=["memory"], default="memory")
    ap.add_argument("--nonclifford-mode", choices=["statevector", "qpd"], default="statevector")
    ap.add_argument("--tick-policy", choices=["auto", "explicit", "instruction"], default="auto")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--src-path", type=str, default=None)
    ap.add_argument("--vis-path", type=str, default=None,
                    help="可視化/橋渡しモジュールの場所（既定: このスクリプトと同じ場所）")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out-html", type=str, default="ftqc_run_report.html")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.vis_path or os.path.dirname(os.path.abspath(__file__)))
    if args.src_path:
        sys.path.insert(0, str(Path(args.src_path).resolve()))

    from ftqc.viz.ftqc_run_visualizer import _parse_gates
    gates = _parse_gates(args.gates)
    n = args.n_logical or (max((max(a, b if b is not None else 0) for _n, a, b in gates), default=0) + 1)

    doc = build_run_report_html(
        gates, n_logical=n, p_phys=args.p_phys, cnot_mode=args.cnot_mode,
        t_qec_mode=args.t_qec_mode, nonclifford_mode=args.nonclifford_mode,
        tick_policy=args.tick_policy, shots=args.shots, title=args.title,
    )
    Path(args.out_html).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
