#!/usr/bin/env python3
"""
Visualize T-gate execution branches for the hybrid FTQC simulator.

Focus: QPD/gate-cutting view of T/Tdag.  Each T is expanded into a signed
Clifford-channel branch.  The S/SDG term is explicitly marked as routed to the
LS-S shell, while I/Z terms remain ordinary Clifford/QEC branches.

Outputs:
  * HTML: interactive-looking static report with timeline, branch tree, tables.
  * SVG : branch tree only.
  * JSON: machine-readable manifest for later integration with runtime traces.

The script is self-contained and does not require stim/qulacs.  It uses a small
NumPy state-vector only to validate/reconstruct noiseless Z-basis distributions.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import zipfile
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Gate = Tuple[str, int, Optional[int]]

I2 = np.eye(2, dtype=complex)
X1 = np.array([[0, 1], [1, 0]], dtype=complex)
Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S1 = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG1 = np.array([[1, 0], [0, -1j]], dtype=complex)
T1 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG1 = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
_SINGLE = {
    "I": I2,
    "X": X1,
    "Y": Y1,
    "Z": Z1,
    "H": H1,
    "S": S1,
    "SDG": SDG1,
    "T": T1,
    "T_DAG": TDG1,
}

SQRT2 = math.sqrt(2.0)
T_QPD_TERMS: Tuple[Tuple[str, float], ...] = (
    ("I", 0.5),
    ("S", 1.0 / SQRT2),
    ("Z", -(SQRT2 - 1.0) / 2.0),
)
TDG_QPD_TERMS: Tuple[Tuple[str, float], ...] = (
    ("I", 0.5),
    ("SDG", 1.0 / SQRT2),
    ("Z", -(SQRT2 - 1.0) / 2.0),
)


@dataclass
class QPDTEvent:
    event_id: int
    gate_index: int
    gate: str
    target: int
    gamma: float
    terms: List[Dict[str, object]]


@dataclass
class LSRoute:
    t_event_id: int
    gate_index: int
    target: int
    branch_term: str
    shell_label: str
    route: str = "LS-S shell"
    note: str = "QPD S/SDG term is a Clifford phase gate implemented through LS-S."


@dataclass
class ExpandedBranch:
    branch_id: int
    term_path: Tuple[str, ...]
    coefficient: float
    abs_coefficient: float
    sign: int
    sampling_probability: float
    estimator_weight: float
    gates: Tuple[Gate, ...]
    ls_routes: Tuple[LSRoute, ...]


# ---------------------------------------------------------------------------
# Parsing and state-vector utilities
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    name = str(name).upper()
    return {
        "CX": "CNOT",
        "S_DAG": "SDG",
        "S†": "SDG",
        "TDG": "T_DAG",
        "T†": "T_DAG",
    }.get(name, name)


def parse_gate_json(s: str) -> List[Gate]:
    raw = json.loads(s)
    if not isinstance(raw, list):
        raise ValueError("--gates-json must be a JSON list")
    out: List[Gate] = []
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name")
            a = item.get("target", item.get("a"))
            b = item.get("b", item.get("control_target", item.get("target2")))
            if normalize_name(name) == "CNOT":
                # For CNOT dicts, allow control/target keys.
                a = item.get("control", item.get("a", a))
                b = item.get("target", item.get("b", b))
            if name is None or a is None:
                raise ValueError(f"Bad gate dict: {item!r}")
            out.append((normalize_name(name), int(a), None if b is None else int(b)))
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                name, a = item
                out.append((normalize_name(name), int(a), None))
            elif len(item) == 3:
                name, a, b = item
                out.append((normalize_name(name), int(a), None if b is None else int(b)))
            else:
                raise ValueError(f"Bad gate tuple: {item!r}")
        else:
            raise ValueError(f"Bad gate entry: {item!r}")
    return out


def parse_gate_text(s: str) -> List[Gate]:
    """Parse lightweight syntax such as 'H0 T0 H1 CNOT0,1 T1 S0'."""
    gates: List[Gate] = []
    for token in s.replace(";", " ").split():
        token = token.strip()
        if not token:
            continue
        upper = token.upper()
        if upper.startswith("CNOT") or upper.startswith("CX"):
            name = "CNOT"
            rest = token[4:] if upper.startswith("CNOT") else token[2:]
            rest = rest.strip("()")
            if "," not in rest:
                raise ValueError(f"CNOT token must look like CNOT0,1: {token!r}")
            a, b = rest.split(",", 1)
            gates.append((name, int(a), int(b)))
        else:
            # Greedy known gate prefix.
            matched = None
            for prefix in ("T_DAG", "TDG", "SDG", "S_DAG", "H", "T", "S", "X", "Y", "Z", "I"):
                if upper.startswith(prefix):
                    matched = prefix
                    break
            if matched is None:
                raise ValueError(f"Unknown gate token: {token!r}")
            rest = token[len(matched):].strip("()")
            gates.append((normalize_name(matched), int(rest), None))
    return gates


def infer_n(gates: Sequence[Gate], n_logical: Optional[int] = None) -> int:
    if n_logical is not None:
        return int(n_logical)
    m = 0
    for _, a, b in gates:
        m = max(m, a if b is None else max(a, b))
    return m + 1


def apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    st = state.reshape([2] * n)
    st = np.tensordot(U, st, axes=([1], [q]))
    return np.moveaxis(st, 0, q).reshape(-1)


def apply_cnot(state: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    st = state.reshape([2] * n)
    sl = [slice(None)] * n
    sl[c] = 1
    sub = st[tuple(sl)]
    sub = np.flip(sub, axis=(t if t < c else t - 1))
    st[tuple(sl)] = sub
    return st.reshape(-1)


def apply_gate(state: np.ndarray, gate: Gate, n: int) -> np.ndarray:
    name, a, b = gate
    name = normalize_name(name)
    if name == "I":
        return state
    if name == "CNOT":
        if b is None:
            raise ValueError("CNOT requires target")
        return apply_cnot(state, a, b, n)
    if name not in _SINGLE:
        raise ValueError(f"Unsupported statevector gate {name}")
    return apply_1q(state, _SINGLE[name], a, n)


def z_distribution_from_gates(gates: Sequence[Gate], n: int) -> Dict[Tuple[int, ...], float]:
    st = np.zeros(2 ** n, dtype=complex)
    st[0] = 1.0
    for g in gates:
        st = apply_gate(st, g, n)
    probs = np.abs(st) ** 2
    out: Dict[Tuple[int, ...], float] = {}
    for idx, p in enumerate(probs):
        if abs(float(p)) < 1e-14:
            continue
        key = tuple((idx >> (n - 1 - q)) & 1 for q in range(n))
        out[key] = out.get(key, 0.0) + float(p)
    return out


# ---------------------------------------------------------------------------
# QPD expansion
# ---------------------------------------------------------------------------

def terms_for_gate(name: str) -> Tuple[Tuple[str, float], ...]:
    name = normalize_name(name)
    if name == "T":
        return T_QPD_TERMS
    if name == "T_DAG":
        return TDG_QPD_TERMS
    raise ValueError(f"No QPD terms for {name}")


def make_t_events(gates: Sequence[Gate]) -> List[QPDTEvent]:
    events: List[QPDTEvent] = []
    for gi, (name, a, b) in enumerate(gates):
        name = normalize_name(name)
        if name not in ("T", "T_DAG"):
            continue
        terms = []
        gamma = sum(abs(c) for _, c in terms_for_gate(name))
        for term, coeff in terms_for_gate(name):
            route = "LS-S shell" if term in ("S", "SDG") else "ordinary Clifford/QEC branch"
            terms.append({
                "term": term,
                "coefficient": coeff,
                "abs_coefficient": abs(coeff),
                "sampling_probability": abs(coeff) / gamma,
                "estimator_factor": gamma * (1 if coeff >= 0 else -1),
                "route": route,
            })
        events.append(QPDTEvent(len(events), gi, name, a, gamma, terms))
    return events


def expand_branches(gates: Sequence[Gate]) -> Tuple[List[ExpandedBranch], List[QPDTEvent]]:
    t_events = make_t_events(gates)
    total_gamma = math.prod(ev.gamma for ev in t_events) if t_events else 1.0
    branches: List[Tuple[float, Tuple[str, ...], Tuple[Gate, ...], Tuple[LSRoute, ...]]] = [(1.0, tuple(), tuple(), tuple())]
    t_event_by_gate_index = {ev.gate_index: ev for ev in t_events}

    for gi, (name, a, b) in enumerate(gates):
        name = normalize_name(name)
        if name in ("T", "T_DAG"):
            ev = t_event_by_gate_index[gi]
            new = []
            for coeff0, path0, gates0, routes0 in branches:
                for term, coeff in terms_for_gate(name):
                    new_gates = gates0 if term == "I" else gates0 + ((term, a, None),)
                    routes = routes0
                    if term in ("S", "SDG"):
                        routes = routes0 + (LSRoute(
                            t_event_id=ev.event_id,
                            gate_index=gi,
                            target=a,
                            branch_term=term,
                            shell_label=f"T{ev.event_id}_gate{gi}_q{a}_{term}_ls_s_shell",
                        ),)
                    new.append((float(coeff0) * float(coeff), path0 + (term,), new_gates, routes))
            branches = new
        else:
            branches = [(c, p, gs + ((name, a, b),), r) for c, p, gs, r in branches]

    out: List[ExpandedBranch] = []
    for bid, (coeff, path, bgates, routes) in enumerate(branches):
        sign = 1 if coeff >= 0 else -1
        # If branch is sampled from product abs(coeff_i)/gamma_i, then probability is abs(coeff)/total_gamma.
        sample_prob = abs(coeff) / total_gamma if total_gamma else 1.0
        estimator_weight = total_gamma * sign
        out.append(ExpandedBranch(
            branch_id=bid,
            term_path=path,
            coefficient=float(coeff),
            abs_coefficient=abs(float(coeff)),
            sign=sign,
            sampling_probability=float(sample_prob),
            estimator_weight=float(estimator_weight),
            gates=bgates,
            ls_routes=routes,
        ))
    return out, t_events


def qpd_reconstruct_distribution(branches: Sequence[ExpandedBranch], n: int) -> Dict[Tuple[int, ...], float]:
    dist: Dict[Tuple[int, ...], float] = {}
    for br in branches:
        bd = z_distribution_from_gates(br.gates, n)
        for key, p in bd.items():
            dist[key] = dist.get(key, 0.0) + br.coefficient * p
    for k in list(dist):
        if abs(dist[k]) < 1e-13:
            dist[k] = 0.0
    return {k: dist[k] for k in sorted(dist)}


def sample_qpd_measurements(branches: Sequence[ExpandedBranch], n: int, shots: int, seed: int) -> Tuple[Dict[str, int], Dict[Tuple[int, ...], float]]:
    rng = np.random.default_rng(seed)
    probs = np.array([br.sampling_probability for br in branches], dtype=float)
    probs = probs / probs.sum()
    branch_dists = [z_distribution_from_gates(br.gates, n) for br in branches]
    branch_keys = [list(d.keys()) for d in branch_dists]
    branch_probs = [np.array([d[k] for k in keys], dtype=float) for d, keys in zip(branch_dists, branch_keys)]

    branch_counts: Dict[str, int] = {}
    signed: Dict[Tuple[int, ...], float] = {}
    choices = rng.choice(len(branches), size=shots, p=probs)
    for i in choices:
        br = branches[int(i)]
        label = "/".join(br.term_path) if br.term_path else "<no T>"
        branch_counts[label] = branch_counts.get(label, 0) + 1
        keys = branch_keys[int(i)]
        kp = branch_probs[int(i)]
        kp = kp / kp.sum()
        outcome = keys[int(rng.choice(len(keys), p=kp))]
        signed[outcome] = signed.get(outcome, 0.0) + br.estimator_weight / shots
    return dict(sorted(branch_counts.items())), {k: signed[k] for k in sorted(signed)}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def fmt_gate(g: Gate) -> str:
    name, a, b = g
    return f"{name}({a},{b})" if b is not None else f"{name}({a})"


def fmt_path(path: Sequence[str]) -> str:
    return " / ".join(path) if path else "<no T>"


def outcome_str(k: Tuple[int, ...]) -> str:
    return "".join(str(x) for x in k)


def clean_float(x: float, digits: int = 6) -> str:
    if abs(x) < 5 * 10 ** (-(digits + 1)):
        x = 0.0
    return f"{x:.{digits}f}"


def route_badge(term: str) -> str:
    cls = "route-ls" if term in ("S", "SDG") else "route-qec"
    text = "LS-S shell" if term in ("S", "SDG") else "Clifford/QEC"
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def build_tree_svg(branches: Sequence[ExpandedBranch], events: Sequence[QPDTEvent], max_branches: int = 243) -> str:
    t_count = len(events)
    if t_count == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="120"><text x="20" y="40">No T/Tdag gates: no QPD branch tree.</text></svg>'
    if len(branches) > max_branches:
        w = 800
        h = 160
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
        <text x="20" y="40" font-family="Arial" font-size="18">Branch tree omitted: {len(branches)} branches exceeds max {max_branches}.</text>
        <text x="20" y="75" font-family="Arial" font-size="14">Use --max-tree-branches to raise the limit, or inspect the JSON/table.</text>
        </svg>'''

    # Collect all prefixes.
    prefixes = {(): {"coeff": 1.0, "prob": 1.0}}
    total_gamma = math.prod(ev.gamma for ev in events) if events else 1.0
    for br in branches:
        coeff = 1.0
        prob = 1.0
        for depth, term in enumerate(br.term_path, start=1):
            ev = events[depth - 1]
            c = dict((t, cf) for t, cf in terms_for_gate(ev.gate))[term]
            coeff *= c
            prob *= abs(c) / ev.gamma
            prefixes[br.term_path[:depth]] = {"coeff": coeff, "prob": prob}

    leaves = sorted([br.term_path for br in branches])
    leaf_x: Dict[Tuple[str, ...], float] = {}
    margin_x = 90
    leaf_gap = 120
    width = max(760, margin_x * 2 + max(1, len(leaves) - 1) * leaf_gap)
    level_gap = 128
    height = 120 + t_count * level_gap
    for i, leaf in enumerate(leaves):
        leaf_x[leaf] = margin_x + i * leaf_gap

    def prefix_x(prefix: Tuple[str, ...]) -> float:
        if len(prefix) == t_count:
            return leaf_x[prefix]
        descendants = [leaf_x[l] for l in leaves if l[:len(prefix)] == prefix]
        return sum(descendants) / len(descendants)

    def node_y(depth: int) -> float:
        return 58 + depth * level_gap

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('''<defs>
      <style>
        .edge{stroke:#8592a3;stroke-width:1.4;fill:none}.node{stroke:#1f2937;stroke-width:1.1}.root{fill:#f8fafc}.pos{fill:#e8f5e9}.neg{fill:#ffebee}.ls{fill:#fff3cd}.qec{fill:#e3f2fd}.txt{font-family:Arial, sans-serif;fill:#111827}.small{font-size:11px}.med{font-size:13px;font-weight:700}.edge-label{font-size:10px;fill:#374151}
      </style>
    </defs>''')
    parts.append(f'<text class="txt med" x="20" y="24">T-gate QPD branch tree: {t_count} T events, {len(branches)} Clifford-only branches, gamma={total_gamma:.6f}</text>')

    for prefix in sorted(prefixes, key=lambda p: (len(p), p)):
        depth = len(prefix)
        if depth == 0:
            continue
        parent = prefix[:-1]
        term = prefix[-1]
        x1, y1 = prefix_x(parent), node_y(depth - 1)
        x2, y2 = prefix_x(prefix), node_y(depth)
        parts.append(f'<path class="edge" d="M{x1:.1f},{y1+18:.1f} C{x1:.1f},{(y1+y2)/2:.1f} {x2:.1f},{(y1+y2)/2:.1f} {x2:.1f},{y2-20:.1f}"/>')
        ev = events[depth - 1]
        coeff = dict((t, c) for t, c in terms_for_gate(ev.gate))[term]
        label = f'{term}  c={coeff:+.3f}'
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2 - 4
        parts.append(f'<text class="txt edge-label" x="{midx:.1f}" y="{midy:.1f}" text-anchor="middle">{html.escape(label)}</text>')

    for prefix in sorted(prefixes, key=lambda p: (len(p), p)):
        depth = len(prefix)
        x, y = prefix_x(prefix), node_y(depth)
        if depth == 0:
            cls = "root"
            label1, label2 = "start", "coeff=+1"
        else:
            term = prefix[-1]
            coeff = prefixes[prefix]["coeff"]
            cls = "ls" if term in ("S", "SDG") else ("neg" if coeff < 0 else "qec")
            label1 = fmt_path(prefix)
            label2 = f'coef={coeff:+.4f}'
        rx, ry = 44, 21
        parts.append(f'<rect class="node {cls}" x="{x-rx:.1f}" y="{y-ry:.1f}" width="{2*rx}" height="{2*ry}" rx="9"/>')
        parts.append(f'<text class="txt med" x="{x:.1f}" y="{y-4:.1f}" text-anchor="middle">{html.escape(label1)}</text>')
        parts.append(f'<text class="txt small" x="{x:.1f}" y="{y+12:.1f}" text-anchor="middle">{html.escape(label2)}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def render_distribution_table(title: str, dist: Dict[Tuple[int, ...], float], reference: Optional[Dict[Tuple[int, ...], float]] = None) -> str:
    keys = sorted(set(dist) | (set(reference) if reference else set()))
    max_abs = max([abs(dist.get(k, 0.0)) for k in keys] + ([abs(reference.get(k, 0.0)) for k in keys] if reference else [1e-9]))
    rows = []
    for k in keys:
        v = dist.get(k, 0.0)
        width = 0 if max_abs == 0 else min(100.0, 100.0 * abs(v) / max_abs)
        sign_cls = "negative" if v < -1e-12 else "positive"
        ref_cell = ""
        if reference is not None:
            rv = reference.get(k, 0.0)
            ref_cell = f"<td>{clean_float(rv)}</td><td>{clean_float(abs(v-rv), 6)}</td>"
        rows.append(f'''
        <tr>
          <td><code>{outcome_str(k)}</code></td>
          <td>{clean_float(v)}</td>
          {ref_cell}
          <td><div class="bar-bg"><div class="bar {sign_cls}" style="width:{width:.1f}%"></div></div></td>
        </tr>''')
    extra_heads = "<th>ideal T</th><th>|diff|</th>" if reference is not None else ""
    return f'''
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <table>
        <thead><tr><th>outcome</th><th>value</th>{extra_heads}<th>bar</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    '''


def render_html(
    gates: Sequence[Gate],
    n: int,
    branches: Sequence[ExpandedBranch],
    events: Sequence[QPDTEvent],
    tree_svg: str,
    qpd_dist: Dict[Tuple[int, ...], float],
    ideal_dist: Dict[Tuple[int, ...], float],
    sampled_counts: Dict[str, int],
    sampled_qpd_dist: Dict[Tuple[int, ...], float],
    shots: int,
    title: str,
) -> str:
    total_gamma = math.prod(ev.gamma for ev in events) if events else 1.0
    variance_overhead = total_gamma ** 2
    t_count = len(events)
    program_s = "  ".join(fmt_gate(g) for g in gates)

    timeline_cells = []
    t_event_by_gate_index = {ev.gate_index: ev for ev in events}
    for gi, g in enumerate(gates):
        name, a, b = g
        if gi in t_event_by_gate_index:
            ev = t_event_by_gate_index[gi]
            term_cards = []
            for t in ev.terms:
                term = str(t["term"])
                term_cards.append(f'''
                    <div class="term-card {'ls-term' if term in ('S','SDG') else 'qec-term'}">
                      <b>{html.escape(term)}</b><br/>
                      coeff={float(t['coefficient']):+.6f}<br/>
                      prob={float(t['sampling_probability']):.4f}<br/>
                      {route_badge(term)}
                    </div>
                ''')
            timeline_cells.append(f'''
                <div class="gate tgate">
                  <div class="gate-title">g{gi}: {html.escape(name)} on q{a}</div>
                  <div class="split">{''.join(term_cards)}</div>
                </div>
            ''')
        else:
            timeline_cells.append(f'<div class="gate normal"><div class="gate-title">g{gi}: {html.escape(fmt_gate(g))}</div></div>')

    event_rows = []
    for ev in events:
        term_rows = "".join(
            f'''<tr><td>{ev.event_id}</td><td>g{ev.gate_index}</td><td>{html.escape(ev.gate)} q{ev.target}</td><td>{html.escape(str(t['term']))}</td><td>{float(t['coefficient']):+.9f}</td><td>{float(t['sampling_probability']):.6f}</td><td>{route_badge(str(t['term']))}</td></tr>'''
            for t in ev.terms
        )
        event_rows.append(term_rows)

    branch_rows = []
    for br in branches:
        route_text = "<br/>".join(html.escape(r.shell_label) for r in br.ls_routes) or '<span class="muted">none</span>'
        gates_text = " ".join(fmt_gate(g) for g in br.gates) or "I"
        branch_rows.append(f'''
        <tr>
          <td>{br.branch_id}</td>
          <td><code>{html.escape(fmt_path(br.term_path))}</code></td>
          <td>{br.coefficient:+.9f}</td>
          <td>{br.sign:+d}</td>
          <td>{br.sampling_probability:.6f}</td>
          <td>{br.estimator_weight:+.6f}</td>
          <td>{route_text}</td>
          <td><code>{html.escape(gates_text)}</code></td>
        </tr>
        ''')

    sample_rows = []
    for path, cnt in sorted(sampled_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        sample_rows.append(f'<tr><td><code>{html.escape(path)}</code></td><td>{cnt}</td><td>{cnt/shots:.4f}</td></tr>')

    return f'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{ --ink:#111827; --muted:#6b7280; --line:#d1d5db; --bg:#f3f4f6; --card:#ffffff; --qec:#e3f2fd; --ls:#fff3cd; --neg:#ffebee; --pos:#e8f5e9; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
header {{ padding:28px 34px; background:#0f172a; color:white; }}
header h1 {{ margin:0 0 8px; font-size:26px; }}
header p {{ margin:4px 0; color:#dbeafe; }}
main {{ padding:24px; max-width:1420px; margin:0 auto; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; margin:0 0 18px; box-shadow:0 1px 2px rgba(15,23,42,.05); }}
h2 {{ font-size:18px; margin:0 0 12px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.stat {{ border:1px solid var(--line); border-radius:12px; padding:12px; background:#fafafa; }}
.stat b {{ display:block; font-size:22px; margin-bottom:3px; }}
.muted {{ color:var(--muted); }}
.timeline {{ display:flex; gap:10px; align-items:stretch; overflow-x:auto; padding-bottom:6px; }}
.gate {{ min-width:170px; border:1px solid var(--line); border-radius:14px; padding:10px; background:#fff; }}
.gate.normal {{ background:#f8fafc; }}
.gate.tgate {{ background:#fff7ed; border-color:#fed7aa; min-width:360px; }}
.gate-title {{ font-weight:700; margin-bottom:8px; }}
.split {{ display:flex; gap:8px; }}
.term-card {{ flex:1; border-radius:12px; padding:10px; font-size:12px; line-height:1.5; border:1px solid var(--line); }}
.qec-term {{ background:var(--qec); }}
.ls-term {{ background:var(--ls); border-color:#facc15; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; margin-top:4px; }}
.route-ls {{ background:#f59e0b; color:#111827; }}
.route-qec {{ background:#bfdbfe; color:#1e3a8a; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ border-bottom:1px solid #e5e7eb; padding:8px 10px; vertical-align:top; text-align:left; }}
th {{ background:#f9fafb; position:sticky; top:0; z-index:1; }}
code {{ background:#f3f4f6; border:1px solid #e5e7eb; padding:1px 5px; border-radius:6px; }}
.scroll {{ max-height:480px; overflow:auto; border:1px solid #e5e7eb; border-radius:12px; }}
.svgbox {{ overflow:auto; border:1px solid #e5e7eb; border-radius:12px; background:white; }}
.formula {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:#f8fafc; border:1px solid #e5e7eb; padding:12px; border-radius:12px; }}
.bar-bg {{ width:180px; height:14px; background:#f3f4f6; border-radius:999px; overflow:hidden; border:1px solid #e5e7eb; }}
.bar {{ height:100%; }} .bar.positive {{ background:#93c5fd; }} .bar.negative {{ background:#fca5a5; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width: 900px) {{ .two {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>QPD backend view: T/T† is replaced by a signed Clifford branch ensemble. The S/SDG term is routed to the LS-S shell.</p>
  <p class="muted">program: {html.escape(program_s)}</p>
</header>
<main>
  <section class="card">
    <h2>概要</h2>
    <div class="stats">
      <div class="stat"><b>{n}</b><span>logical qubits</span></div>
      <div class="stat"><b>{len(gates)}</b><span>logical gates</span></div>
      <div class="stat"><b>{t_count}</b><span>T/T† gates</span></div>
      <div class="stat"><b>{len(branches)}</b><span>Clifford branches = 3^T</span></div>
      <div class="stat"><b>{total_gamma:.6f}</b><span>total gamma = (sqrt(2))^T</span></div>
      <div class="stat"><b>{variance_overhead:.6f}</b><span>variance overhead gamma^2</span></div>
    </div>
  </section>

  <section class="card">
    <h2>T QPD decomposition</h2>
    <div class="formula">T = 0.5 I + 1/sqrt(2) S - ((sqrt(2)-1)/2) Z<br/>T† = 0.5 I + 1/sqrt(2) S† - ((sqrt(2)-1)/2) Z</div>
    <p class="muted">Monte Carlo branch sampling uses p_i = |c_i| / gamma. The signed estimator multiplies each sampled T term by gamma × sign(c_i).</p>
    <div class="scroll"><table><thead><tr><th>T event</th><th>gate idx</th><th>gate</th><th>term</th><th>coefficient</th><th>sample prob</th><th>route</th></tr></thead><tbody>{''.join(event_rows)}</tbody></table></div>
  </section>

  <section class="card">
    <h2>Logical gate timeline with T split points</h2>
    <div class="timeline">{''.join(timeline_cells)}</div>
  </section>

  <section class="card">
    <h2>Branch tree</h2>
    <div class="svgbox">{tree_svg}</div>
  </section>

  <section class="card">
    <h2>Expanded Clifford-only branches</h2>
    <div class="scroll"><table><thead><tr><th>#</th><th>term path</th><th>coefficient</th><th>sign</th><th>sample prob</th><th>estimator weight</th><th>LS-S routes</th><th>branch gates</th></tr></thead><tbody>{''.join(branch_rows)}</tbody></table></div>
  </section>

  <section class="card">
    <h2>Sampled branch frequencies ({shots} shots)</h2>
    <div class="scroll"><table><thead><tr><th>term path</th><th>count</th><th>frequency</th></tr></thead><tbody>{''.join(sample_rows)}</tbody></table></div>
  </section>

  <div class="two">
    {render_distribution_table('Exact signed QPD reconstruction vs ideal T statevector', qpd_dist, ideal_dist)}
    {render_distribution_table('Finite-shot signed measurement estimator', sampled_qpd_dist, ideal_dist)}
  </div>

  <section class="card">
    <h2>読み方</h2>
    <p><b>I/Z branch</b> は Clifford-only なので通常の Stim/Clifford QEC chunk に流せます。<b>S/SDG branch</b> は phase gate branch なので <code>LS-S shell</code> にルーティングされます。</p>
    <p>負係数を持つ Z branch は物理確率ではなく、後処理の signed weight で復元します。そのため有限ショット estimator には小さな負値が出ることがあります。</p>
  </section>
</main>
</body>
</html>'''


def to_jsonable_distribution(dist: Dict[Tuple[int, ...], float]) -> Dict[str, float]:
    return {outcome_str(k): float(v) for k, v in sorted(dist.items())}


def build_manifest(
    gates: Sequence[Gate],
    n: int,
    events: Sequence[QPDTEvent],
    branches: Sequence[ExpandedBranch],
    qpd_dist: Dict[Tuple[int, ...], float],
    ideal_dist: Dict[Tuple[int, ...], float],
    sampled_counts: Dict[str, int],
    sampled_qpd_dist: Dict[Tuple[int, ...], float],
    shots: int,
    seed: int,
) -> Dict[str, object]:
    total_gamma = math.prod(ev.gamma for ev in events) if events else 1.0
    return {
        "schema": "t_branch_visualizer.v1",
        "n_logical": n,
        "gates": [{"name": name, "a": a, "b": b} for name, a, b in gates],
        "t_count": len(events),
        "num_branches": len(branches),
        "gamma_total": total_gamma,
        "variance_overhead": total_gamma ** 2,
        "qpd_terms": {
            "T": [{"term": t, "coefficient": c} for t, c in T_QPD_TERMS],
            "T_DAG": [{"term": t, "coefficient": c} for t, c in TDG_QPD_TERMS],
        },
        "t_events": [asdict(ev) for ev in events],
        "branches": [
            {
                "branch_id": br.branch_id,
                "term_path": list(br.term_path),
                "coefficient": br.coefficient,
                "abs_coefficient": br.abs_coefficient,
                "sign": br.sign,
                "sampling_probability": br.sampling_probability,
                "estimator_weight": br.estimator_weight,
                "gates": [{"name": name, "a": a, "b": b} for name, a, b in br.gates],
                "ls_routes": [asdict(r) for r in br.ls_routes],
            }
            for br in branches
        ],
        "distributions": {
            "exact_qpd_signed": to_jsonable_distribution(qpd_dist),
            "ideal_statevector": to_jsonable_distribution(ideal_dist),
            "sampled_qpd_signed": to_jsonable_distribution(sampled_qpd_dist),
        },
        "sampled_branch_counts": sampled_counts,
        "sampling": {"shots": shots, "seed": seed},
    }


def default_gates() -> List[Gate]:
    return [
        ("H", 0, None),
        ("T", 0, None),
        ("H", 1, None),
        ("CNOT", 0, 1),
        ("T", 1, None),
        ("S", 0, None),
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Visualize QPD branches at T-gate execution points.")
    ap.add_argument("--gates-json", default=None, help="JSON list of gates, e.g. [[\"H\",0,null],[\"T\",0,null]].")
    ap.add_argument("--gates", default=None, help="Lightweight gate text, e.g. 'H0 T0 H1 CNOT0,1 T1 S0'.")
    ap.add_argument("--n-logical", type=int, default=None)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="T-gate branch visualization")
    ap.add_argument("--max-tree-branches", type=int, default=243)
    ap.add_argument("--out-html", default="/mnt/data/t_branch_visualization.html")
    ap.add_argument("--out-json", default="/mnt/data/t_branch_visualization.json")
    ap.add_argument("--out-svg", default="/mnt/data/t_branch_visualization.svg")
    #ap.add_argument("--zip", default="/mnt/data/t_branch_visualizer_bundle.zip")
    ap.add_argument("--zip", default="")
    args = ap.parse_args(argv)

    if args.gates_json:
        gates = parse_gate_json(args.gates_json)
    elif args.gates:
        gates = parse_gate_text(args.gates)
    else:
        gates = default_gates()
    n = infer_n(gates, args.n_logical)
    branches, events = expand_branches(gates)
    qpd_dist = qpd_reconstruct_distribution(branches, n)
    ideal_dist = z_distribution_from_gates(gates, n)
    sampled_counts, sampled_qpd_dist = sample_qpd_measurements(branches, n, args.shots, args.seed)
    tree_svg = build_tree_svg(branches, events, max_branches=args.max_tree_branches)
    html_doc = render_html(gates, n, branches, events, tree_svg, qpd_dist, ideal_dist, sampled_counts, sampled_qpd_dist, args.shots, args.title)
    manifest = build_manifest(gates, n, events, branches, qpd_dist, ideal_dist, sampled_counts, sampled_qpd_dist, args.shots, args.seed)

    for path, content, mode in [
        (args.out_html, html_doc, "w"),
        (args.out_svg, tree_svg, "w"),
        (args.out_json, json.dumps(manifest, indent=2, ensure_ascii=False), "w"),
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in [args.out_html, args.out_svg, args.out_json, __file__]:
                zf.write(p, arcname=os.path.basename(p))

    print("created:")
    print("  html:", args.out_html)
    print("  svg :", args.out_svg)
    print("  json:", args.out_json)
    if args.zip:
        print("  zip :", args.zip)


if __name__ == "__main__":
    main()
