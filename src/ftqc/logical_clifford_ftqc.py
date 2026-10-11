"""
論理 Clifford 回路 -> 物理 (Stim) コンパイラ  [MPP バックボーン統合版]
======================================================================
入力 : qulacs の論理 Clifford 回路 (X, Y, Z, H, CNOT) / N 論理量子ビット
方式 : 各論理量子ビット = rotated 表面符号パッチ(d=3)。安定化子は MPP で測定。
       回路を層(moment)に分解し、各層で物理 transversal ゲートを適用 -> シンドローム
       ラウンドを挿入。論理ゲートの物理実装:
         Pauli X/Y/Z : 固定の論理演算子台への transversal Pauli (frame 非依存)
         H           : 全データへの transversal H (X<->Z 安定化子が双対化)
         CNOT        : パッチ間 transversal CX (+ 境界検出器補正)

到達範囲 (rotated 表面符号で純 transversal で正しく届く範囲)
  * 論理「値」の正しさ (ノイズなし) : {Pauli, H, CNOT} を参照論理シミュレーションで検証
  * フォールトトレラントなデコード   : {Pauli, CNOT} 部分集合 (常に primal frame で
      検出器が clean)。H を含むデコードは双対格子検出器 + frame 依存の CNOT 補正が
      必要で本実装の対象外、S は transversal 不可 (lattice surgery / magic state)。
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


def _mpp(typ, locs, base):
    tg = []
    for k, q in enumerate(locs):
        tg.append(stim.target_x(base + q) if typ == "X" else stim.target_z(base + q))
        if k < len(locs) - 1:
            tg.append(stim.target_combiner())
    return tg


# ---------------------------------------------------------------------------
# qulacs 論理回路 -> ゲート列 -> 層(moment)分解
# ---------------------------------------------------------------------------
def parse_gates(qulacs_circuit):
    gates = []
    for g in range(qulacs_circuit.get_gate_count()):
        gate = qulacs_circuit.get_gate(g)
        name = gate.get_name()
        if name not in SUPPORTED:
            raise ValueError(f"未対応のゲート '{name}'. 対応: {sorted(SUPPORTED)} (S/T は別機構)")
        ctrl = gate.get_control_index_list()
        tgt = gate.get_target_index_list()
        if name in ("CNOT", "CX"):
            gates.append(("CNOT", ctrl[0], tgt[0]))
        else:
            gates.append((name, tgt[0], None))
    return gates


def layer_gates(gates, n_logical):
    """依存関係(時間順序)を保つ ASAP スケジューリングで層に分割。
    各ゲートは、共有する論理量子ビットに触れた直近の層より後ろに置く。"""
    last = {q: -1 for q in range(n_logical)}  # 各論理量子ビットを最後に使った層
    layers = []
    for name, a, b in gates:
        qubits = [a] if b is None else [a, b]
        L = 1 + max(last[q] for q in qubits)
        while len(layers) <= L:
            layers.append([])
        layers[L].append((name, a, b))
        for q in qubits:
            last[q] = L
    return layers


# ---------------------------------------------------------------------------
# コンパイラ本体
# ---------------------------------------------------------------------------
class LogicalCliffordFTQC:
    def __init__(self, n_logical, p=0.005, stride=NDATA):
        self.N = n_logical
        self.p = p
        self.stride = stride

    def base(self, q):
        return q * self.stride

    def data(self, q):
        b = self.base(q)
        return [b + i for i in range(NDATA)]

    # ---- 物理ゲートの適用 (frame 非依存な配置) ----
    def _apply_layer_gates(self, c, layer, noise):
        pp = self.p if noise else 0.0
        for (name, a, b) in layer:
            if name == "X":
                c.append("X", [self.base(a) + q for q in XL])
            elif name == "Z":
                c.append("Z", [self.base(a) + q for q in ZL])
            elif name == "Y":
                c.append("X", [self.base(a) + q for q in XL])
                c.append("Z", [self.base(a) + q for q in ZL])
            elif name == "H":
                c.append("H", self.data(a))
            elif name == "CNOT":
                for i in range(NDATA):
                    c.append("CX", [self.base(a) + i, self.base(b) + i])
                if noise:
                    for i in range(NDATA):
                        c.append("DEPOLARIZE2", [self.base(a) + i, self.base(b) + i], pp)

    # =====================================================================
    # (1) 論理「値」の正しさ検証用: ノイズなし物理回路 (検出器なし)
    # =====================================================================
    def compile_logical_only(self, layers):
        c = stim.Circuit()
        all_data = [d for q in range(self.N) for d in self.data(q)]
        c.append("R", all_data)
        # encode: 全パッチ全安定化子を一度測って符号空間へ射影
        for q in range(self.N):
            for typ, sup in STABS:
                c.append("MPP", _mpp(typ, sup, self.base(q)))
        # ゲート層を適用 (層間に射影ラウンドを挟み符号空間を保つ)
        for layer in layers:
            self._apply_layer_gates(c, layer, noise=False)
            for q in range(self.N):
                for typ, sup in STABS:
                    c.append("MPP", _mpp(typ, sup, self.base(q)))
        # 全データを Z 測定
        c.append("M", all_data)
        return c

    def validate(self, qulacs_circuit, shots=8000):
        """物理回路の論理 Z 読み出し分布を、参照論理シミュレーションと比較。"""
        gates = parse_gates(qulacs_circuit)
        layers = layer_gates(gates, self.N)

        # 参照: 同じ Clifford 回路を N 論理量子ビットで直接サンプル
        ref = stim.Circuit()
        for (name, a, b) in gates:
            if name == "CNOT":
                ref.append("CX", [a, b])
            else:
                ref.append(name, [a])
        ref.append("M", list(range(self.N)))
        ref_samp = ref.compile_sampler().sample(shots)
        ref_keys = [tuple(row) for row in ref_samp.astype(int)]

        # 物理: Z_L 台のパリティを各パッチで読む
        phys = self.compile_logical_only(layers)
        nm = phys.num_measurements
        psamp = phys.compile_sampler().sample(shots)
        data = psamp[:, nm - NDATA * self.N:nm]
        phys_keys = []
        for row in data.astype(int):
            key = tuple(int(row[q * NDATA:(q + 1) * NDATA][ZL].sum() % 2) for q in range(self.N))
            phys_keys.append(key)

        from collections import Counter
        rc, pc = Counter(ref_keys), Counter(phys_keys)
        outcomes = sorted(set(rc) | set(pc))
        # 分布の総変動距離
        tvd = 0.5 * sum(abs(rc.get(o, 0) - pc.get(o, 0)) for o in outcomes) / shots
        return tvd, rc, pc, outcomes

    # =====================================================================
    # (2) フォールトトレラント版 (検出器あり, {Pauli, CNOT} のみ対応)
    # =====================================================================
    def compile_ft(self, layers, rounds_tail=2, noise=True):
        if any(name == "H" for layer in layers for (name, _, _) in layer):
            raise ValueError("compile_ft は {Pauli, CNOT} 専用です (H のデコードは双対格子検出器が必要)。")
        pp = self.p if noise else 0.0
        c = stim.Circuit()
        all_data = [d for q in range(self.N) for d in self.data(q)]
        c.append("R", all_data)

        mc = 0
        ridx = {}      # (round, q, stab) -> abs idx
        round_id = 0

        def syndrome_round(cnot_pairs):
            nonlocal mc, round_id
            c.append("DEPOLARIZE1", all_data, pp)
            for q in range(self.N):
                for si, (typ, sup) in enumerate(STABS):
                    c.append("MPP", _mpp(typ, sup, self.base(q)), pp)
                    ridx[(round_id, q, si)] = mc
                    mc += 1
            round_id_local = round_id
            round_id += 1
            return round_id_local

        # round 0 (encode)
        r0 = syndrome_round([])
        # 各層: ゲート適用 -> シンドロームラウンド (CNOT 層は境界補正)
        cnot_at_round = {}  # round -> list of (control,target)
        for layer in layers:
            self._apply_layer_gates(c, layer, noise=noise)
            pairs = [(a, b) for (name, a, b) in layer if name == "CNOT"]
            r = syndrome_round(pairs)
            cnot_at_round[r] = pairs
        # 末尾ラウンド (距離確保)
        for _ in range(rounds_tail):
            syndrome_round([])
        R = round_id

        c.append("X_ERROR", all_data, pp)
        c.append("M", all_data)
        drec = {}
        for q in range(self.N):
            for i in range(NDATA):
                drec[(q, i)] = mc
                mc += 1
        rr = lambda a: stim.target_rec(a - mc)

        # ----- 検出器 -----
        # round 0: Z 安定化子は決定論的
        for q in range(self.N):
            for si, (typ, sup) in enumerate(STABS):
                if typ == "Z":
                    c.append("DETECTOR", [rr(ridx[(0, q, si)])])
        # round 1..R-1: 各安定化子を前ラウンドと比較。CNOT 層直後は境界補正。
        for r in range(1, R):
            pairs = cnot_at_round.get(r, [])
            cnot_ctrl = {a for (a, b) in pairs}
            cnot_tgt = {b for (a, b) in pairs}
            partner = {}
            for (a, b) in pairs:
                partner[a] = b   # control -> target
                partner[b] = a   # target -> control
            for q in range(self.N):
                for si, (typ, sup) in enumerate(STABS):
                    tg = [rr(ridx[(r, q, si)]), rr(ridx[(r - 1, q, si)])]
                    if q in cnot_ctrl and typ == "X":     # S_X^C -> S_X^C·S_X^T
                        tg.append(rr(ridx[(r - 1, partner[q], si)]))
                    if q in cnot_tgt and typ == "Z":      # S_Z^T -> S_Z^C·S_Z^T
                        tg.append(rr(ridx[(r - 1, partner[q], si)]))
                    c.append("DETECTOR", tg)
        # 最終: データから Z 安定化子を再構成し最終ラウンドと比較
        for q in range(self.N):
            for si, (typ, sup) in enumerate(STABS):
                if typ == "Z":
                    tg = [rr(ridx[(R - 1, q, si)])] + [rr(drec[(q, s)]) for s in sup]
                    c.append("DETECTOR", tg)
        # 観測量: 各論理量子ビットの Z_L
        for q in range(self.N):
            c.append("OBSERVABLE_INCLUDE", [rr(drec[(q, s)]) for s in ZL], q)
        return c

    def decode(self, circ, shots=40000):
        det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
        dem = circ.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        pred = matcher.decode_batch(det)
        res = obs ^ pred
        return res.mean(axis=0)


# ===========================================================================
def demo():
    from qulacs import QuantumCircuit

    print("#" * 70)
    print("#  論理 Clifford 回路 -> 物理 (Stim) コンパイラ  [MPP + transversal CX]")
    print("#" * 70)

    # ---- (A) {Pauli, H, CNOT} の論理正しさ検証 (ノイズなし, 参照比較) ----
    print("\n[A] 任意 Clifford 回路の論理正しさ (参照論理シミュレーションと比較)")

    def bell():
        c = QuantumCircuit(2); c.add_H_gate(0); c.add_CNOT_gate(0, 1); return c, 2

    def ghz():
        c = QuantumCircuit(3); c.add_H_gate(0); c.add_CNOT_gate(0, 1); c.add_CNOT_gate(1, 2); return c, 3

    def mixed():
        c = QuantumCircuit(3)
        c.add_H_gate(0); c.add_CNOT_gate(0, 1); c.add_X_gate(2)
        c.add_CNOT_gate(1, 2); c.add_H_gate(1); c.add_Z_gate(0)
        return c, 3

    for label, (qc, N) in [("Bell  H(0);CX(0,1)", bell()),
                           ("GHZ   H(0);CX(0,1);CX(1,2)", ghz()),
                           ("Mixed H,CX,X,CX,H,Z", mixed())]:
        comp = LogicalCliffordFTQC(N)
        tvd, rc, pc, outs = comp.validate(qc, shots=8000)
        top = sorted(outs, key=lambda o: -(rc.get(o, 0) + pc.get(o, 0)))[:4]
        dist = "  ".join(f"{''.join(map(str,o))}:参照{rc.get(o,0)/8000:.2f}/物理{pc.get(o,0)/8000:.2f}" for o in top)
        print(f"    {label}")
        print(f"        TVD(参照,物理)={tvd:.3f}  {'一致 OK' if tvd < 0.03 else 'NG'}   {dist}")

    # ---- (B) {Pauli, CNOT} のフォールトトレラント実行 (検出器 + デコード) ----
    print("\n[B] フォールトトレラント実行 ({Pauli, CNOT}, 検出器 + MWPM デコード)")
    # X(0); CNOT(0,1); CNOT(1,2)  (論理エラー伝搬する 3 量子ビット回路)
    qc = QuantumCircuit(3)
    qc.add_X_gate(0); qc.add_CNOT_gate(0, 1); qc.add_CNOT_gate(1, 2)
    N = 3
    comp = LogicalCliffordFTQC(N, p=0.005)
    gates = parse_gates(qc)
    layers = layer_gates(gates, N)
    print(f"    回路: X(0); CNOT(0,1); CNOT(1,2)  -> 層数={len(layers)}")

    # 境界補正の検証 (ノイズなしで検出器が全0か)
    cn = comp.compile_ft(layers, noise=False)
    det, _ = cn.compile_detector_sampler().sample(100, separate_observables=True)
    print(f"    ノイズなし検出器が全0 (境界補正の正しさ): {not det.any()}")

    # 論理値 (ノイズなし): X(0)=1 が伝搬し 1->1->1
    vals_tvd, rc, pc, outs = comp.validate(qc, shots=4000)
    best = max(pc, key=pc.get)
    print(f"    ノイズなし論理値(最頻) = {''.join(map(str,best))}  (期待 111: X が CNOT 連鎖で伝搬)")

    for p in (0.002, 0.005, 0.01):
        comp_p = LogicalCliffordFTQC(N, p=p)
        circ = comp_p.compile_ft(layers, noise=True)
        rates = comp_p.decode(circ, shots=40000)
        print(f"    p={p}: 論理エラー率 " + ", ".join(f"q{i}={rates[i]:.5f}" for i in range(N)))

    print("\n[サマリ]")
    print("    * {Pauli, H, CNOT} の任意 Clifford 回路を物理回路へ変換し、論理正しさを検証 (A)")
    print("    * {Pauli, CNOT} は検出器込みでコンパイルし、MWPM デコードまで実行 (B)")
    print("    * H のデコード(双対格子検出器)・S(lattice surgery/magic state)が残課題")


if __name__ == "__main__":
    demo()
