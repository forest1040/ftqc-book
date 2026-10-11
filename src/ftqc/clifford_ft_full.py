"""
完全 Clifford の FT 化: frame 追跡 + 双対格子検出器
=====================================================
{Pauli, H, CNOT} の任意 Clifford 回路を物理回路へコンパイルし、H を含めて
フォールトトレラントにデコードする。

中心となる考え方 — 生成子追跡 (generator tracking)
--------------------------------------------------
パッチ固有の検出器を「張り替える」のではなく、安定化子生成子そのものを Pauli 文字列
として保持し、各 transversal ゲートで共役変換する:
    H      : 対象データ qubit で X<->Z を入れ替え (双対格子化)
    CNOT   : X を制御->標的へ、Z を標的->制御へ伝播 (生成子は2パッチに広がりうる)
    Pauli  : 符号のみ変化 (Stim の基準が処理するので文字列は不変)
各ラウンドでは「現在の」生成子を MPP で測り、検出器は
    「同インデックス生成子の連続ラウンド比較」
という一様規則にする。round r の生成子 = G(round r-1 の生成子)G† なので、
G を挟んでも測定値は一致し、ノイズなしで検出器は決定論的に 0 になる。
エラー伝播は Stim の detector_error_model が取り込み、PyMatching が
(decompose_errors により) graphlike に分解してデコードする。

到達範囲: {Pauli, H, CNOT} = 完全 Clifford 群 (S は H と CNOT からは作れないが、
S 自体の transversal 実装は別途必要。本実装は H を含む Clifford のデコードを担う)。
"""
import numpy as np
import stim
import pymatching

STABS = [("X", [0, 1]), ("Z", [0, 1, 3, 4]), ("X", [1, 2, 4, 5]), ("Z", [2, 5]),
         ("Z", [3, 6]), ("X", [3, 4, 6, 7]), ("Z", [4, 5, 7, 8]), ("X", [7, 8])]
XL = [0, 3, 6]
ZL = [0, 1, 2]
NDATA = 9
SUPPORTED = {"X", "Y", "Z", "H", "CNOT", "CX"}


# ----------------------------------------------------------------- ゲート解析
def parse_gates(qulacs_circuit):
    out = []
    for g in range(qulacs_circuit.get_gate_count()):
        gate = qulacs_circuit.get_gate(g)
        name = gate.get_name()
        if name not in SUPPORTED:
            raise ValueError(f"未対応のゲート '{name}' (S/T は別機構)")
        c, t = gate.get_control_index_list(), gate.get_target_index_list()
        out.append(("CNOT", c[0], t[0]) if name in ("CNOT", "CX") else (name, t[0], None))
    return out


def layer_gates(gates, n_logical):
    """依存関係(時間順序)を保つ ASAP スケジューリング。"""
    last = {q: -1 for q in range(n_logical)}
    layers = []
    for name, a, b in gates:
        qs = [a] if b is None else [a, b]
        L = 1 + max(last[q] for q in qs)
        while len(layers) <= L:
            layers.append([])
        layers[L].append((name, a, b))
        for q in qs:
            last[q] = L
    return layers


# ----------------------------------------------------- 生成子 (Pauli 文字列) 追跡
class PauliStr:
    __slots__ = ("x", "z")

    def __init__(self, M):
        self.x = np.zeros(M, np.int8)
        self.z = np.zeros(M, np.int8)

    def conj_H(self, qubits):
        for t in qubits:
            self.x[t], self.z[t] = self.z[t], self.x[t]

    def conj_CX(self, a, b):           # 制御 a, 標的 b
        self.x[b] ^= self.x[a]
        self.z[a] ^= self.z[b]

    def mpp_targets(self):
        tg, first = [], True
        for q in range(len(self.x)):
            xx, zz = self.x[q], self.z[q]
            if not xx and not zz:
                continue
            t = stim.target_y(q) if (xx and zz) else (stim.target_x(q) if xx else stim.target_z(q))
            if not first:
                tg.append(stim.target_combiner())
            tg.append(t)
            first = False
        return tg


class CliffordFTCompiler:
    def __init__(self, n_logical, p=0.005, tail_rounds=1):
        self.N = n_logical
        self.p = p
        self.tail_rounds = tail_rounds

    def _init_generators(self):
        M = NDATA * self.N
        gens = []  # [patch, stab_idx, PauliStr, init_type]
        for q in range(self.N):
            base = q * NDATA
            for si, (typ, sup) in enumerate(STABS):
                g = PauliStr(M)
                for s in sup:
                    (g.x if typ == "X" else g.z)[base + s] = 1
                gens.append([q, si, g, typ])
        logz = []
        for q in range(self.N):
            base = q * NDATA
            g = PauliStr(M)
            for s in ZL:
                g.z[base + s] = 1
            logz.append(g)
        return gens, logz

    def compile(self, layers, noise=True):
        pp = self.p if noise else 0.0
        c = stim.Circuit()
        all_data = list(range(NDATA * self.N))
        c.append("R", all_data)
        gens, logz = self._init_generators()

        mc = 0
        ridx = {}
        round_id = 0

        def syndrome_round():
            nonlocal mc, round_id
            c.append("DEPOLARIZE1", all_data, pp)
            for gi, (_, _, g, _) in enumerate(gens):
                c.append("MPP", g.mpp_targets(), pp)
                ridx[(round_id, gi)] = mc
                mc += 1
            rid = round_id
            round_id += 1
            return rid

        def apply_gate(name, a, b):
            ba = a * NDATA
            bb = b * NDATA if b is not None else None
            if name == "H":
                qs = [ba + i for i in range(NDATA)]
                c.append("H", qs)
                for (_, _, g, _) in gens:
                    g.conj_H(qs)
                for g in logz:
                    g.conj_H(qs)
            elif name == "CNOT":
                for i in range(NDATA):
                    c.append("CX", [ba + i, bb + i])
                if noise:
                    for i in range(NDATA):
                        c.append("DEPOLARIZE2", [ba + i, bb + i], pp)
                for (_, _, g, _) in gens:
                    for i in range(NDATA):
                        g.conj_CX(ba + i, bb + i)
                for g in logz:
                    for i in range(NDATA):
                        g.conj_CX(ba + i, bb + i)
            elif name == "X":
                c.append("X", [ba + s for s in XL])
            elif name == "Z":
                c.append("Z", [ba + s for s in ZL])
            elif name == "Y":
                c.append("X", [ba + s for s in XL])
                c.append("Z", [ba + s for s in ZL])

        syndrome_round()                       # round 0 (encode)
        for layer in layers:
            for (name, a, b) in layer:
                apply_gate(name, a, b)
            syndrome_round()
        for _ in range(self.tail_rounds):
            syndrome_round()
        R = round_id

        # 論理 Z 測定 (現在の共役済み logz を MPP)
        obs_rec = {}
        for q in range(self.N):
            c.append("MPP", logz[q].mpp_targets())
            obs_rec[q] = mc
            mc += 1
        rr = lambda a: stim.target_rec(a - mc)

        # 検出器
        for gi, (_, _, _, typ) in enumerate(gens):
            if typ == "Z":                      # round 0: Z 型は決定論的
                c.append("DETECTOR", [rr(ridx[(0, gi)])])
        for r in range(1, R):                   # 一様規則: 同インデックス連続比較
            for gi in range(len(gens)):
                c.append("DETECTOR", [rr(ridx[(r, gi)]), rr(ridx[(r - 1, gi)])])
        for q in range(self.N):
            c.append("OBSERVABLE_INCLUDE", [rr(obs_rec[q])], q)
        return c, obs_rec

    def decode(self, circ, shots=40000):
        """MWPM を試し、非 graphlike なら BP+OSD (ハイパーグラフ) にフォールバック。"""
        det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
        try:
            dem = circ.detector_error_model(decompose_errors=True, allow_gauge_detectors=False)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            pred = matcher.decode_batch(det)
            decoder = "MWPM"
        except ValueError:
            from stimbposd import BPOSD
            dem = circ.detector_error_model(decompose_errors=False, allow_gauge_detectors=False)
            pred = BPOSD(dem, max_bp_iters=40).decode_batch(det)
            decoder = "BP+OSD"
        return (obs ^ pred).mean(axis=0), obs.mean(axis=0), decoder

    # ---- 論理正しさ検証 (ノイズなしデータ Z 読み出し vs 参照論理シミュレーション) ----
    def _logical_only_circuit(self, layers):
        """検出器なし: 符号化 -> ゲート -> 全データ Z 測定 (論理値読み出し用)。"""
        c = stim.Circuit()
        all_data = list(range(NDATA * self.N))
        c.append("R", all_data)

        def proj():
            for q in range(self.N):
                base = q * NDATA
                for typ, sup in STABS:
                    g = PauliStr(NDATA * self.N)
                    for s in sup:
                        (g.x if typ == "X" else g.z)[base + s] = 1
                    c.append("MPP", g.mpp_targets())

        def apply_gate(name, a, b):
            ba = a * NDATA
            bb = b * NDATA if b is not None else None
            if name == "H":
                c.append("H", [ba + i for i in range(NDATA)])
            elif name == "CNOT":
                for i in range(NDATA):
                    c.append("CX", [ba + i, bb + i])
            elif name == "X":
                c.append("X", [ba + s for s in XL])
            elif name == "Z":
                c.append("Z", [ba + s for s in ZL])
            elif name == "Y":
                c.append("X", [ba + s for s in XL])
                c.append("Z", [ba + s for s in ZL])

        proj()
        for layer in layers:
            for (name, a, b) in layer:
                apply_gate(name, a, b)
            proj()
        c.append("M", all_data)
        return c

    def validate(self, qulacs_circuit, shots=8000):
        gates = parse_gates(qulacs_circuit)
        layers = layer_gates(gates, self.N)
        circ = self._logical_only_circuit(layers)
        nm = circ.num_measurements
        samp = circ.compile_sampler().sample(shots)
        data = samp[:, nm - NDATA * self.N:nm].astype(int)
        phys_keys = [tuple(int(row[q * NDATA:(q + 1) * NDATA][ZL].sum() % 2) for q in range(self.N))
                     for row in data]

        ref = stim.Circuit()
        for (name, a, b) in gates:
            ref.append("CX", [a, b]) if name == "CNOT" else ref.append(name, [a])
        ref.append("M", list(range(self.N)))
        ref_keys = [tuple(row) for row in ref.compile_sampler().sample(shots).astype(int)]

        from collections import Counter
        rc, pc = Counter(ref_keys), Counter(phys_keys)
        outs = sorted(set(rc) | set(pc))
        tvd = 0.5 * sum(abs(rc.get(o, 0) - pc.get(o, 0)) for o in outs) / shots
        return tvd, rc, pc, outs


# ===========================================================================
def demo():
    from qulacs import QuantumCircuit

    print("#" * 70)
    print("#  完全 Clifford の FT 化: 生成子追跡 + 双対格子検出器")
    print("#  {Pauli, H, CNOT} を H 込みでコンパイル & デコード")
    print("#" * 70)

    def bell():
        c = QuantumCircuit(2); c.add_H_gate(0); c.add_CNOT_gate(0, 1); return c, 2

    def ghz():
        c = QuantumCircuit(3); c.add_H_gate(0); c.add_CNOT_gate(0, 1); c.add_CNOT_gate(1, 2); return c, 3

    def mixed():
        c = QuantumCircuit(3)
        c.add_H_gate(0); c.add_CNOT_gate(0, 1); c.add_X_gate(2)
        c.add_CNOT_gate(1, 2); c.add_H_gate(1); c.add_Z_gate(0)
        return c, 3

    cases = [("Bell  H(0);CX(0,1)", bell()),
             ("GHZ   H(0);CX(0,1);CX(1,2)", ghz()),
             ("Mixed H,CX,X,CX,H,Z", mixed())]

    print("\n[A] 論理正しさ (参照論理シミュレーションと比較)")
    for label, (qc, N) in cases:
        comp = CliffordFTCompiler(N)
        tvd, rc, pc, outs = comp.validate(qc, shots=8000)
        print(f"    {label:30s} TVD={tvd:.3f} {'OK' if tvd < 0.03 else 'NG'}")

    print("\n[B] H 込みの決定論性 (ノイズなしで検出器が全0)")
    for label, (qc, N) in cases:
        comp = CliffordFTCompiler(N)
        layers = layer_gates(parse_gates(qc), N)
        circ, _ = comp.compile(layers, noise=False)
        det, _ = circ.compile_detector_sampler().sample(200, separate_observables=True)
        print(f"    {label:30s} 検出器全0 = {not det.any()}")

    print("\n[C] H 込みの FT デコード (MWPM、非graphlikeは BP+OSD に自動切替)")
    for label, (qc, N) in cases:
        layers = layer_gates(parse_gates(qc), N)
        comp = CliffordFTCompiler(N, p=0.005)
        circ, _ = comp.compile(layers, noise=True)
        dec, _, which = comp.decode(circ, shots=30000)
        print(f"    {label:30s} [{which:6s}] p=0.005 論理エラー率={np.round(dec, 4).tolist()}")

    print("\n[D] デコードが本物である確認 (Bell, デコード後 << デコードなし, p で単調)")
    qc, N = bell()
    layers = layer_gates(parse_gates(qc), N)
    for p in (0.001, 0.003, 0.01):
        comp = CliffordFTCompiler(N, p=p)
        circ, _ = comp.compile(layers, noise=True)
        dec, undec, which = comp.decode(circ, shots=40000)
        print(f"    p={p}: [{which}] デコード後={np.round(dec,5).tolist()}  デコードなし={np.round(undec,4).tolist()}")

    print("\n[サマリ]")
    print("    生成子追跡 + 一様検出器規則により、H を含む完全 Clifford 回路を")
    print("    ノイズなし決定論性を保ったままコンパイルし、graphlike は MWPM、")
    print("    非 graphlike は BP+OSD で自動デコードまで実現。")
    print("    (残: S の transversal 実装 / 非Clifford T は magic state 注入)")


if __name__ == "__main__":
    demo()
