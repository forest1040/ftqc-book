#!/usr/bin/env python3
"""
ftqc_run_visualizer — 現行シミュレータの run/plan 出力を既存の可視化に配線する薄いブリッジ.

HybridFTQCRuntime は各イベントの Stim 回路を内部で構築して即サンプル・デコードし、
回路自体は QECReport に保持しません。本ブリッジは *非侵襲* に、

  1. runtime.plan(gates) でイベント列（CliffordChunk / LSCnotEvent / SpecialGate）を得て、
  2. 各イベントの Stim 回路を、runtime と同じ公開ビルダ（CliffordFTCompiler /
     build_ls_cnot_shell / build_ls_s_shell / build_ls_t_injection_shell）で、
     runtime の設定（n, p_phys, tail_rounds, cnot_mode, t_qec_mode, t_s_correction）
     に従って再構築し、
  3. それを stim_timeline_visualizer に渡してイベントごとの実行時系列（tick）を描く

という流れで「論理ゲート列 → 各イベントが実際に流す回路 → tick」を一望にします。
あわせて、T を含む場合は t_branch_visualizer に同じゲート列を渡して QPD 分岐も描きます。

依存: stim_timeline_visualizer.py / t_branch_visualizer.py（同じ場所 or --vis-path）と
      シミュレータ src/（--src-path）。

例:
  python ftqc_run_visualizer.py --gates "H0 T0 CNOT0,1" --n-logical 2 \
      --cnot-mode lattice_surgery --t-qec-mode lattice_surgery \
      --src-path ./src --out-timeline run_timeline.html --out-tbranch run_tbranch.html
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


# --------------------------------------------------------------------------- #
# 1. イベント → Stim 回路 の再構築（runtime と同じ公開ビルダを使う）
# --------------------------------------------------------------------------- #

def _clifford_circuit(gates: List[Gate], n: int, p: float, tail_rounds: int):
    """CliffordChunk（空なら memory shell）に対応する Stim 回路。"""
    from ftqc.clifford_ft_full import CliffordFTCompiler, layer_gates
    comp = CliffordFTCompiler(n, p=p, tail_rounds=tail_rounds)
    layers = layer_gates(list(gates), n)
    circ, _ = comp.compile(layers, noise=True)
    return circ


def event_circuits(runtime: Any, gates: List[Gate]) -> List[Dict[str, Any]]:
    """plan(gates) の各イベントを、runtime と同じ公開ビルダで Stim 回路に再構築する。

    返り値: [{'label','kind','circuit','ls_builder','highlight'} , ...]（plan 順）。
      - 'circuit'    : timeline 用の Stim 回路（runtime と同じシェル回路）
      - 'ls_builder' : 論理→物理マッピング用のビルダ実体（LS イベントのみ。非 LS は None）
      - 'highlight'  : 依存グラフで強調するファイル id の集合
    """
    from ftqc.hybrid_ftqc_runtime import CliffordChunk, LSCnotEvent, SpecialGate
    from ftqc.ls_handoff_shells import build_ls_cnot_shell

    n = int(getattr(runtime, "n"))
    p = float(getattr(runtime, "p_phys", 0.005))
    tail = int(getattr(runtime, "tail_rounds", 1))
    out: List[Dict[str, Any]] = []
    for ev in runtime.plan(gates):
        label = getattr(ev, "label", type(ev).__name__)
        ls_builder = None
        highlight: set = set()
        if isinstance(ev, CliffordChunk):
            circ = _clifford_circuit(ev.gates, n, p, tail)
            kind = "clifford_chunk" + ("" if ev.gates else " (memory shell)")
            highlight = {"cf"}
        elif isinstance(ev, LSCnotEvent):
            circ = build_ls_cnot_shell(p=p, noise=True).circuit
            kind = "ls_cnot_shell"
            from ftqc.ls_cnot_ft import LSCnotFT
            ls_builder = LSCnotFT()
            highlight = {"sh", "cn", "gm"}
        elif isinstance(ev, SpecialGate):
            name = ev.name.upper()
            if name in ("S", "SDG"):
                circ = _clifford_circuit([], n, p, tail)
                kind = f"special:{name} (memory shell)"
                highlight = {"cf"}
            elif name in ("T", "TDG", "T_DAG"):
                circ = _clifford_circuit([], n, p, tail)
                kind = "T memory shell"
                highlight = {"cf", "tg"}
            else:
                circ = _clifford_circuit([], n, p, tail)
                kind = f"special:{name} (memory shell)"
                highlight = {"cf"}
        else:
            continue
        out.append({"label": label, "kind": kind, "circuit": circ,
                    "ls_builder": ls_builder, "highlight": highlight})
    return out


# --------------------------------------------------------------------------- #
# 2. timeline への配線
# --------------------------------------------------------------------------- #

def _timeline_svg_and_stats(circuit_text: str, tick_policy: str = "auto") -> Tuple[str, Dict[str, int]]:
    from ftqc.viz import stim_timeline_visualizer as tl
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "event.stim"
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


_CSS = """
body{font-family:ui-sans-serif,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f1f5f9;color:#1f2937;}
header{background:#0f172a;color:#fff;padding:22px 30px;}
header h1{margin:0 0 6px;font-size:22px;} header p{margin:2px 0;color:#cbd5e1;font-size:13.5px;}
main{max-width:1280px;margin:0 auto;padding:22px;}
.event{background:#fff;border:1px solid #e2e8f0;border-radius:13px;padding:16px 18px;margin:16px 0;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.event h2{margin:0 0 2px;font-size:17px;}
.event .kind{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:2px 9px;font-size:12px;color:#3730a3;margin-bottom:8px;}
.badge{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;border-radius:50%;background:#0f172a;color:#fff;font-weight:700;margin-right:8px;font-size:13px;}
.stats{display:flex;flex-wrap:wrap;gap:7px;margin:6px 0;}
.pill{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:999px;padding:3px 9px;font-size:12px;}
.scroll{overflow:auto;border:1px solid #eef2f7;border-radius:9px;padding:8px;background:#fff;}
footer{text-align:center;color:#94a3b8;font-size:12px;padding:20px;}
.note{color:#475569;font-size:13px;margin:2px 0 10px;}
"""


def run_timeline_report(
    runtime: Any,
    gates: List[Gate],
    out_html: str = "run_timeline.html",
    tick_policy: str = "auto",
    title: Optional[str] = None,
) -> str:
    """plan(gates) の各イベント回路を timeline で描き、1 ページに束ねる。"""
    evs = event_circuits(runtime, gates)
    title = title or "FTQC run timeline — 論理ゲート列 → イベント回路 → tick"
    cfg = (f"n={getattr(runtime,'n','?')}, p_phys={getattr(runtime,'p_phys','?')}, "
           f"cnot_mode={getattr(runtime,'cnot_mode','?')}, t_qec_mode={getattr(runtime,'t_qec_mode','?')}, "
           f"nonclifford_mode={getattr(runtime,'nonclifford_mode','?')}")
    sections = []
    for i, ev in enumerate(evs, 1):
        svg, st = _timeline_svg_and_stats(str(ev["circuit"]), tick_policy=tick_policy)
        pills = "".join(f'<span class="pill">{k}={v}</span>' for k, v in st.items())
        sections.append(
            f'<section class="event"><h2><span class="badge">{i}</span>{html.escape(ev["label"])}</h2>'
            f'<span class="kind">{html.escape(ev["kind"])}</span>'
            f'<div class="stats">{pills}</div>'
            f'<div class="scroll">{svg}</div></section>'
        )
    body = "\n".join(sections) if sections else "<p class='note'>イベントがありません。</p>"
    doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<header><h1>{html.escape(title)}</h1>
<p>{html.escape(cfg)}</p>
<p>plan が分解した各イベントについて、runtime と同じビルダで再構築した Stim 回路を tick 展開しています。</p></header>
<main>{body}</main>
<footer>generated by ftqc_run_visualizer.py — runtime.plan × stim_timeline_visualizer</footer>
</body></html>"""
    Path(out_html).write_text(doc, encoding="utf-8")
    return out_html


# --------------------------------------------------------------------------- #
# 3. t_branch への配線（同じゲート列の QPD 分岐）
# --------------------------------------------------------------------------- #

def gates_to_tbranch_text(gates: List[Gate]) -> str:
    """[('H',0,None),('T',0,None),('CNOT',0,1)] -> 'H0 T0 CNOT0,1'."""
    toks = []
    for name, a, b in gates:
        toks.append(f"{name}{a}" if b is None else f"{name}{a},{b}")
    return " ".join(toks)


def has_t(gates: List[Gate]) -> bool:
    return any(name.upper() in ("T", "TDG") for name, _a, _b in gates)


def t_branch_report(
    gates: List[Gate],
    out_html: str = "run_tbranch.html",
    n_logical: Optional[int] = None,
    shots: int = 20000,
    out_json: Optional[str] = None,
    out_svg: Optional[str] = None,
) -> Optional[str]:
    """同じゲート列を t_branch_visualizer に渡し、T の QPD 分岐ツリーを描く。

    T/TDG が無い場合は None を返す（QPD 分岐が無いため）。
    """
    if not has_t(gates):
        return None
    from ftqc.viz import t_branch_visualizer as tb
    argv = ["--gates", gates_to_tbranch_text(gates), "--shots", str(shots), "--out-html", out_html]
    if n_logical is not None:
        argv += ["--n-logical", str(n_logical)]
    if out_json:
        argv += ["--out-json", out_json]
    if out_svg:
        argv += ["--out-svg", out_svg]
    tb.main(argv)
    return out_html


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_gates(text: str) -> List[Gate]:
    from ftqc.viz import t_branch_visualizer as tb
    return list(tb.parse_gate_text(text))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="現行シミュレータの run/plan を timeline / t_branch に配線する薄いブリッジ")
    ap.add_argument("--gates", required=True, help="ゲート列。例: 'H0 T0 CNOT0,1 T1'")
    ap.add_argument("--n-logical", type=int, default=None, help="論理量子ビット数（既定: ゲートから推定）")
    ap.add_argument("--p-phys", type=float, default=0.005)
    ap.add_argument("--cnot-mode", choices=["transversal", "lattice_surgery"], default="transversal")
    ap.add_argument("--t-qec-mode", choices=["memory"], default="memory")
    ap.add_argument("--nonclifford-mode", choices=["statevector", "qpd"], default="statevector")
    ap.add_argument("--tick-policy", choices=["auto", "explicit", "instruction"], default="auto")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--src-path", type=str, default=None)
    ap.add_argument("--vis-path", type=str, default=None,
                    help="可視化モジュールのある場所（既定: このスクリプトと同じ場所）")
    ap.add_argument("--out-timeline", type=str, default="run_timeline.html")
    ap.add_argument("--out-tbranch", type=str, default="run_tbranch.html")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.vis_path or os.path.dirname(os.path.abspath(__file__)))
    if args.src_path:
        sys.path.insert(0, str(Path(args.src_path).resolve()))

    gates = _parse_gates(args.gates)
    n = args.n_logical or (max((max(a, b if b is not None else 0) for _n, a, b in gates), default=0) + 1)

    from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime
    rt = HybridFTQCRuntime(n, p_phys=args.p_phys, cnot_mode=args.cnot_mode,
                           nonclifford_mode=args.nonclifford_mode)

    tl_path = run_timeline_report(rt, gates, out_html=args.out_timeline, tick_policy=args.tick_policy)
    print(f"wrote {tl_path}")
    tb_path = t_branch_report(gates, out_html=args.out_tbranch, n_logical=n, shots=args.shots,
                              out_json=str(Path(args.out_tbranch).with_suffix(".json")),
                              out_svg=str(Path(args.out_tbranch).with_suffix(".svg")))
    if tb_path:
        print(f"wrote {tb_path}")
    else:
        print("(no T/TDG gates -> t_branch report skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
