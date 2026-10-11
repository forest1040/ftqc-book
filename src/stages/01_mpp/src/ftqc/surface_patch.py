"""
surface_patch.py — 回路レベル単一パッチメモリ (geom 規約整合)
=============================================================
ハイブリッド FTQC の MPP -> 回路レベル移行の土台。Stim の
`surface_code:rotated_memory_z`(ancilla ベース syndrome extraction, 回路レベル
ノイズ)を1パッチのメモリとしてラップし、geom_lattice_surgery.py のローカル規約
(data at (2c+1,2r+1), local id = r*d+c, XL=左列, ZL=下行)と整合させる。
"""
import numpy as np
import stim

# geom 規約の基準 (d=3 検証用)
GEOM_STABS_D3 = [("X", [0, 1]), ("Z", [0, 1, 3, 4]), ("X", [1, 2, 4, 5]), ("Z", [2, 5]),
                 ("Z", [3, 6]), ("X", [3, 4, 6, 7]), ("Z", [4, 5, 7, 8]), ("X", [7, 8])]


def remap_circuit(circ, qmap, shift=(0, 0)):
    """qubit id を qmap で、QUBIT_COORDS/DETECTOR 座標を shift で平行移動した新回路。
    FT を保つ CX スケジュール順はそのまま保存される(=相対関係は不変)。"""
    dx, dy = shift
    out = stim.Circuit()
    for it in circ:
        if isinstance(it, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(it.repeat_count, remap_circuit(it.body_copy(), qmap, shift)))
            continue
        nm = it.name
        args = it.gate_args_copy()
        if nm == "QUBIT_COORDS":
            tv = it.targets_copy()[0].value
            out.append("QUBIT_COORDS", [qmap[tv]], [args[0] + dx, args[1] + dy])
            continue
        if nm == "SHIFT_COORDS":
            out.append(nm, [], args)
            continue
        newt = []
        for t in it.targets_copy():
            if t.is_qubit_target:
                newt.append(qmap[t.value])
            elif t.is_measurement_record_target:
                newt.append(stim.target_rec(t.value))
            elif t.is_combiner:
                newt.append(stim.target_combiner())
            elif t.is_x_target:
                newt.append(stim.target_x(qmap[t.value]))
            elif t.is_y_target:
                newt.append(stim.target_y(qmap[t.value]))
            elif t.is_z_target:
                newt.append(stim.target_z(qmap[t.value]))
            else:
                newt.append(t)
        if nm == "DETECTOR" and len(args) >= 2:
            args = [args[0] + dx, args[1] + dy] + list(args[2:])
        out.append(nm, newt, args)
    return out


def _extract_stabilizers(circ, data, anc):
    """生成回路の CX schedule から各 ancilla が測る (型, data support) を復元。"""
    flat = []
    for it in circ:
        if isinstance(it, stim.CircuitRepeatBlock):
            flat.extend(list(it.body_copy()))
        else:
            flat.append(it)
    h_anc, cx = set(), []
    for it in flat:
        if it.name == "H":
            for t in it.targets_copy():
                h_anc.add(t.value)
        if it.name in ("CX", "ZCX"):
            ts = it.targets_copy()
            cx += [(ts[i].value, ts[i + 1].value) for i in range(0, len(ts), 2)]
    stabs = {}
    for a, b in cx:
        for x, y in ((a, b), (b, a)):
            if x in anc and y in data:
                stabs.setdefault(x, ["X" if x in h_anc else "Z", set()])
                stabs[x][1].add(y)
    return stabs


class SurfacePatch:
    """回路レベル rotated 表面符号の単一パッチ。geom ローカル規約を公開する。"""

    def __init__(self, distance=3):
        self.d = distance
        ref = stim.Circuit.generated("surface_code:rotated_memory_z",
                                     distance=distance, rounds=distance)
        coords = {q: (int(x), int(y)) for q, (x, y) in ref.get_final_qubit_coordinates().items()}
        self.data = {q: c for q, c in coords.items() if c[0] % 2 == 1 and c[1] % 2 == 1}
        self.anc = {q: c for q, c in coords.items() if not (c[0] % 2 == 1 and c[1] % 2 == 1)}

        # geom ローカル規約: local id = r*d + c, 座標 (2c+1, 2r+1)
        self.coord2stim = {c: q for q, c in self.data.items()}
        self.local2stim = {}
        for r in range(distance):
            for c in range(distance):
                self.local2stim[r * distance + c] = self.coord2stim[(2 * c + 1, 2 * r + 1)]
        self.stim2local = {v: k for k, v in self.local2stim.items()}

        # スタビライザ (Stim id -> ローカル)
        raw = _extract_stabilizers(ref, set(self.data), set(self.anc))
        self.stabilizers = []   # geom ローカル規約 (型, [local ids])
        for aid in sorted(raw):
            t, sup = raw[aid]
            self.stabilizers.append((t, sorted(self.stim2local[s] for s in sup)))

        # 論理演算子: XL=左列(x=1), ZL=下行(y=1)
        self.XL = sorted(self.stim2local[q] for q, c in self.data.items() if c[0] == 1)
        self.ZL = sorted(self.stim2local[q] for q, c in self.data.items() if c[1] == 1)

    def memory_circuit(self, rounds=None, p=0.0, basis="Z", slot=0, coord_shift=None):
        rounds = rounds or self.d
        task = f"surface_code:rotated_memory_{basis.lower()}"
        c = stim.Circuit.generated(task, distance=self.d, rounds=rounds,
                                   after_clifford_depolarization=p,
                                   before_measure_flip_probability=p,
                                   after_reset_flip_probability=p,
                                   before_round_data_depolarization=p)
        if slot == 0 and coord_shift is None:
            return c
        stride = 100
        ids = sorted(self.data) + sorted(self.anc)
        qmap = {q: q + stride * slot for q in set(q for q in
                stim.Circuit.generated(task, distance=self.d, rounds=rounds).get_final_qubit_coordinates())}
        shift = coord_shift if coord_shift is not None else (2 * (self.d + 1) * slot, 0)
        return remap_circuit(c, qmap, shift)

    def decode(self, circ, shots=40000):
        import pymatching
        det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
        try:
            dem = circ.detector_error_model(decompose_errors=True)
            m = pymatching.Matching.from_detector_error_model(dem)
            pred = m.decode_batch(det)
            which = "MWPM"
        except ValueError:
            from stimbposd import BPOSD
            pred = BPOSD(circ.detector_error_model(decompose_errors=False), max_bp_iters=40).decode_batch(det)
            which = "BP+OSD"
        return (obs ^ pred).mean(), which


def mpp_memory_logical_error(p, rounds=3, shots=40000):
    """旧来の phenomenological MPP メモリ (ancilla なし, 各スタビライザを MPP 直接測定)。"""
    import pymatching
    sp = SurfacePatch(3)
    c = stim.Circuit()
    data = list(range(9))
    c.append("R", data)
    mc = 0
    ridx = {}
    prev = {}
    dets = []
    for rnd in range(rounds + 1):
        c.append("DEPOLARIZE1", data, p)
        for si, (t, sup) in enumerate(sp.stabilizers):
            tg, first = [], True
            fn = {"X": stim.target_x, "Z": stim.target_z}
            for q in sup:
                if not first:
                    tg.append(stim.target_combiner())
                tg.append(fn[t](q))
                first = False
            c.append("MPP", tg, p)
            if si in prev:
                dets.append([mc, prev[si]])
            elif t == "Z":
                dets.append([mc])
            prev[si] = mc
            mc += 1
    zl = sp.ZL
    tg, first = [], True
    for q in zl:
        if not first:
            tg.append(stim.target_combiner())
        tg.append(stim.target_z(q))
        first = False
    c.append("MPP", tg)
    obs_rec = mc
    mc += 1
    rr = lambda a: stim.target_rec(a - mc)
    for d in dets:
        c.append("DETECTOR", [rr(i) for i in d])
    c.append("OBSERVABLE_INCLUDE", [rr(obs_rec)], 0)
    det, obs = c.compile_detector_sampler().sample(shots, separate_observables=True)
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    return (obs ^ m.decode_batch(det)).mean()
