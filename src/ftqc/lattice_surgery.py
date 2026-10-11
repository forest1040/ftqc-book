"""
論理レベル lattice surgery: joint 測定プリミティブ + LS CNOT + LS S
====================================================================
各論理量子ビット = rotated 表面符号パッチ(d=3)。安定化子は MPP。
joint 測定 M_PP(A,B) は論理演算子の積 P_L(A)·P_L(B) を MPP で測り、
d ラウンド繰り返して検出器で守る (反復による FT)。

ゲジェット (補正則は stim で実機導出)
  LS CNOT : アンシラ|+_L>; M_ZZ(c,a); M_XX(a,t); アンシラ Z 測定
            補正  Z_c if m_xx,   X_t if (m_zz xor m_anc)
  LS S    : アンシラ|Y_L>; M_ZZ(t,a); M_X(a)
            補正  Z_t if (m_zz xor m_x)
  ※ |Y_L> は Clifford(スタビライザ)状態なので蒸留不要。本実装では Y_L 射影で準備。

Pauli フレームで補正を追跡し、最終 Z 読み出しに反映する。論理正しさは
参照論理シミュレーション(stim, S を含む)との分布一致で検証する。
"""
import numpy as np
import stim

STABS = [("X", [0, 1]), ("Z", [0, 1, 3, 4]), ("X", [1, 2, 4, 5]), ("Z", [2, 5]),
         ("Z", [3, 6]), ("X", [3, 4, 6, 7]), ("Z", [4, 5, 7, 8]), ("X", [7, 8])]
XL = [0, 3, 6]
ZL = [0, 1, 2]
NDATA = 9


def pauli_targets(ops, base):
    """ops: list of (local_qubit, 'X'/'Y'/'Z'). base: patch offset."""
    tg, first = [], True
    fn = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    for q, p in ops:
        if not first:
            tg.append(stim.target_combiner())
        tg.append(fn[p](base + q))
        first = False
    return tg


def logical_ops(kind):
    if kind == "X":
        return [(q, "X") for q in XL]
    if kind == "Z":
        return [(q, "Z") for q in ZL]
    if kind == "Y":   # Y_L = i X_L Z_L : qubit0 -> Y, {3,6} -> X, {1,2} -> Z
        return [(0, "Y"), (3, "X"), (6, "X"), (1, "Z"), (2, "Z")]


class LatticeSurgery:
    """論理レベル lattice surgery のビルダ (測定レコードを追跡)。

    p/noise は hybrid runtime から LS-S shell を detector sampler に渡すための
    phenomenological MPP ノイズ設定。既存デモとの互換性のため、既定では
    noise=False として従来どおりノイズ無しの論理検証を行う。
    """

    def __init__(self, n_patches, merge_rounds=3, p=0.0, noise=False):
        self.n = n_patches
        self.mr = merge_rounds
        self.p = float(p)
        self.noise = bool(noise)
        self.pp = self.p if self.noise else 0.0
        self.c = stim.Circuit()
        self.mc = 0
        self.detectors = []   # list of lists of abs indices (XOR=0 expected, noiseless)
        self._detectors_appended = False
        all_data = [q * NDATA + i for q in range(n_patches) for i in range(NDATA)]
        self.all_data = all_data
        self.c.append("R", all_data)
        self.prev_stab = {}   # (patch, stab) -> last abs idx

    def base(self, q):
        return q * NDATA

    def _mpp(self, ops_global, p=None):
        """ops_global: list of (global_qubit,'X/Y/Z'). returns record index."""
        tg, first = [], True
        fn = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
        for q, pa in ops_global:
            if not first:
                tg.append(stim.target_combiner())
            tg.append(fn[pa](q))
            first = False
        pp = self.pp if p is None else float(p)
        self.c.append("MPP", tg, pp)
        idx = self.mc
        self.mc += 1
        return idx

    def stab_round(self, patches, first_round=False):
        if self.noise:
            qs = [self.base(q) + i for q in patches for i in range(NDATA)]
            self.c.append("DEPOLARIZE1", qs, self.pp)
        for q in patches:
            base = self.base(q)
            for si, (typ, sup) in enumerate(STABS):
                ops = [(base + s, typ) for s in sup]
                idx = self._mpp(ops)
                if (q, si) in self.prev_stab:
                    self.detectors.append([idx, self.prev_stab[(q, si)]])
                elif typ == "Z":           # round0: Z 型は決定論的
                    self.detectors.append([idx])
                self.prev_stab[(q, si)] = idx

    # --- アンシラ準備 (Y は射影結果を返す) ---
    def prep(self, q, state):
        base = self.base(q)
        if state == "+":
            self.c.append("H", [base + i for i in range(NDATA)])
        elif state == "0":
            pass
        elif state == "Y":
            self.c.append("H", [base + i for i in range(NDATA)])
        self.stab_round([q], first_round=True)
        if state == "Y":
            # |+_L> を Y_L で射影 -> |Y_L=±1>。outcome (m_prep) を返し補正で使う。
            ops = [(base + l, p) for l, p in logical_ops("Y")]
            return self._mpp(ops)
        return None

    # --- joint 測定 M_PP(A,B): d ラウンド反復, 検出器で FT ---
    def joint_measure(self, P, A, B):
        ops = [(self.base(A) + l, p) for l, p in logical_ops(P)] + \
              [(self.base(B) + l, p) for l, p in logical_ops(P)]
        last = None
        idxs = []
        for r in range(self.mr):
            idx = self._mpp(ops)
            idxs.append(idx)
            if last is not None:
                self.detectors.append([idx, last])
            last = idx
            self.stab_round([A, B])
        return idxs[-1]

    def measure_logical(self, q, basis):
        ops = [(self.base(q) + l, p) for l, p in logical_ops(basis)]
        return self._mpp(ops)

    def append_detectors(self):
        """Materialize accumulated absolute-index detector constraints in Stim.

        The builder keeps detector constraints as absolute measurement indices while
        gadgets are being assembled.  This method converts them to Stim rec[-k]
        references after all relevant measurements/observables have been appended.
        It is idempotent so shell builders can safely call it once at the end.
        """
        if self._detectors_appended:
            return self
        rr = lambda a: stim.target_rec(a - self.mc)
        for d in self.detectors:
            self.c.append("DETECTOR", [rr(int(i)) for i in d])
        self._detectors_appended = True
        return self

    # --- ゲジェット ---
    def ls_cnot(self, ctrl, tgt, anc):
        self.prep(anc, "+")
        m_zz = self.joint_measure("Z", ctrl, anc)
        m_xx = self.joint_measure("X", anc, tgt)
        m_anc = self.measure_logical(anc, "Z")
        return {"Z": [(ctrl, [m_xx])], "X": [(tgt, [m_zz, m_anc])]}

    def ls_s(self, tgt, anc):
        m_prep = self.prep(anc, "Y")          # |Y_L=±1> の射影結果
        m_zz = self.joint_measure("Z", tgt, anc)
        m_x = self.measure_logical(anc, "X")
        # 補正: Z_tgt if (m_zz xor m_x xor m_prep)  (m_prep は S vs S† を吸収)
        return {"Z": [(tgt, [m_zz, m_x, m_prep])], "X": []}


# =====================================================================
# 検証: 論理正しさを参照論理シミュレーション(stim)と比較
# =====================================================================
# =====================================================================
# 検証: 基底を考慮した補正適用 + 参照論理シミュレーション比較
# =====================================================================
# 補正 P が 測定基底 B の読み出しを反転するか (反可換なら反転)
_ANTI = {("X", "Z"): True, ("X", "Y"): True, ("X", "X"): False,
         ("Z", "X"): True, ("Z", "Y"): True, ("Z", "Z"): False}


def run_validation(build, ref_gates, n_logical, meas_basis, shots=8000):
    """build() -> (ls, rec{q:idx}, basis{q}, corrections[list of spec])"""
    ls, rec, basis, corrections = build()
    samp = ls.c.compile_sampler().sample(shots).astype(int)
    keys = []
    for row in samp:
        bits = {q: int(row[r]) for q, r in rec.items()}
        for spec in corrections:
            for ptype in ("X", "Z"):
                for (tg, srcs) in spec.get(ptype, []):
                    if tg not in bits:
                        continue
                    if _ANTI[(ptype, basis[tg])]:
                        f = 0
                        for s in srcs:
                            f ^= int(row[s])
                        bits[tg] ^= f
        keys.append(tuple(bits[q] for q in sorted(bits)))

    ref = stim.Circuit()
    for (name, a, b) in ref_gates:
        ref.append("CX", [a, b]) if name == "CNOT" else ref.append(name, [a])
    for q in range(n_logical):
        ref.append({"Z": "M", "X": "MX", "Y": "MY"}[meas_basis[q]], [q])
    ref_keys = [tuple(x) for x in ref.compile_sampler().sample(shots).astype(int)]

    from collections import Counter
    RC, PC = Counter(ref_keys), Counter(keys)
    outs = sorted(set(RC) | set(PC))
    tvd = 0.5 * sum(abs(RC.get(o, 0) - PC.get(o, 0)) for o in outs) / shots
    top = sorted(outs, key=lambda o: -(RC.get(o, 0) + PC.get(o, 0)))[:4]
    ds = "  ".join(f"{''.join(map(str,o))}:参照{RC.get(o,0)/shots:.2f}/物理{PC.get(o,0)/shots:.2f}" for o in top)
    return tvd, ds


def demo():
    print("#" * 68)
    print("#  論理レベル lattice surgery: joint測定 + LS CNOT + LS S")
    print("#" * 68)

    print("\n[A] LS CNOT の論理正しさ (M_ZZ, M_XX, アンシラ Z 測定)")

    def build_bell():
        ls = LatticeSurgery(3, merge_rounds=3)
        ls.prep(0, "+"); ls.prep(1, "0")
        corr = ls.ls_cnot(0, 1, anc=2)
        rec = {0: ls.measure_logical(0, "Z"), 1: ls.measure_logical(1, "Z")}
        return ls, rec, {0: "Z", 1: "Z"}, [corr]
    tvd, ds = run_validation(build_bell, [("H", 0, None), ("CNOT", 0, 1)], 2, {0: "Z", 1: "Z"})
    print(f"    Bell H(0);CX(0,1)  TVD={tvd:.3f} {'OK' if tvd<0.04 else 'NG'}  {ds}")

    def build_cnot_11():
        ls = LatticeSurgery(3, merge_rounds=3)
        ls.prep(0, "0"); ls.c.append("X", [ls.base(0) + q for q in XL])
        ls.prep(1, "0")
        corr = ls.ls_cnot(0, 1, anc=2)
        rec = {0: ls.measure_logical(0, "Z"), 1: ls.measure_logical(1, "Z")}
        return ls, rec, {0: "Z", 1: "Z"}, [corr]
    tvd, ds = run_validation(build_cnot_11, [("X", 0, None), ("CNOT", 0, 1)], 2, {0: "Z", 1: "Z"})
    print(f"    CNOT c=|1>,t=|0>   TVD={tvd:.3f} {'OK' if tvd<0.04 else 'NG'}  {ds}")

    print("\n[B] LS S の論理正しさ (Y 状態経由, Y 基底で直接検証)")

    def build_s_on_plus():
        # |+_L> に S -> |Y_L>。Y 基底測定で +1 (=0) を確認。
        ls = LatticeSurgery(2, merge_rounds=3)
        ls.prep(0, "+")
        corr = ls.ls_s(0, anc=1)
        rec = {0: ls.measure_logical(0, "Y")}
        return ls, rec, {0: "Y"}, [corr]
    tvd, ds = run_validation(build_s_on_plus, [("H", 0, None), ("S", 0, None)], 1, {0: "Y"})
    print(f"    S|+>=|Y> (Y測定)   TVD={tvd:.3f} {'OK' if tvd<0.04 else 'NG'}  {ds}")

    def build_s_dag_effect():
        # |+_L> に S を2回 = Z -> |-_L>。X 基底測定で -1 (=1) を確認。
        ls = LatticeSurgery(3, merge_rounds=3)
        ls.prep(0, "+")
        c1 = ls.ls_s(0, anc=1)
        c2 = ls.ls_s(0, anc=2)
        rec = {0: ls.measure_logical(0, "X")}
        return ls, rec, {0: "X"}, [c1, c2]
    tvd, ds = run_validation(build_s_dag_effect, [("H", 0, None), ("S", 0, None), ("S", 0, None)], 1, {0: "X"})
    print(f"    S;S|+>=|-> (X測定) TVD={tvd:.3f} {'OK' if tvd<0.04 else 'NG'}  {ds}")

    print("\n[C] joint 測定の FT 構造 (d ラウンド反復 + 検出器)")
    ls = LatticeSurgery(2, merge_rounds=3)
    ls.prep(0, "0"); ls.prep(1, "0")
    ls.joint_measure("Z", 0, 1)
    print(f"    merge_rounds=3 -> joint 反復検出器 = {ls.mr-1}/merge, 総検出器 = {len(ls.detectors)}")

    print("\n[サマリ]")
    print("    joint 測定(M_ZZ/M_XX/M_YY, d反復+検出器)から LS CNOT と LS S(Y状態)")
    print("    を構成し、論理正しさを参照と照合。Clifford {Pauli,H,CNOT,S} が揃った。")
    print("    補正は Pauli フレームで追跡 (X/Z が測定基底と反可換なら読み出し反転)。")


if __name__ == "__main__":
    demo()
