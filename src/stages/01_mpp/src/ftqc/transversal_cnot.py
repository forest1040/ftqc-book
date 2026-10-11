"""
論理 CNOT を transversal CNOT で Stim 内に実装
================================================
2 枚の rotated 表面符号パッチ(d=3) の対応データ量子ビット間に物理 CX を当て、
論理 CNOT を実現する。安定化子は MPP(多体 Pauli 測定) で測り、検出器は自前で管理する。

要点
----
1. CSS 符号間の transversal CX は論理 CNOT そのもの:
     X_L^C -> X_L^C · X_L^T,   Z_L^T -> Z_L^C · Z_L^T  (他は不変)
2. CX は安定化子も広げる (S_X^C -> S_X^C·S_X^T, S_Z^T -> S_Z^C·S_Z^T)。
   そのため CX 直後ラウンドの検出器に「相手パッチの前ラウンド値」を XOR で混ぜる
   ★クロス境界補正★ が必須 (これが無いとノイズなしでも検出器が反応してしまう)。
3. エラー伝播 (X:C->T, Z:T->C) は Stim の detector_error_model が自動で取り込み、
   PyMatching がパッチ間相関を含めてデコードする。

検証 (main 参照)
  [A] 論理作用    : Tableau で X_L^C->X_L^C X_L^T 等を代数確認
  [B] 真理値表    : ノイズなしで CNOT(c,t)->(c, c xor t) を確認
  [C] 境界補正    : 補正の有無で検出器の決定論性が変わることを確認
  [D] ノイズ実行  : MWPM デコードで両論理量子ビットの論理エラー率
"""
import numpy as np
import stim
import pymatching

# rotated surface code d=3 (生成回路から抽出, local data index 0..8)
STABS = [("X", [0, 1]), ("Z", [0, 1, 3, 4]), ("X", [1, 2, 4, 5]), ("Z", [2, 5]),
         ("Z", [3, 6]), ("X", [3, 4, 6, 7]), ("Z", [4, 5, 7, 8]), ("X", [7, 8])]
XL = [0, 3, 6]   # 論理 X 演算子の台 (data local index)
ZL = [0, 1, 2]   # 論理 Z 演算子の台
NDATA = 9
BC, BT = 0, 9    # control / target パッチの qubit index base


def _mpp(typ, locs, base):
    tg = []
    for k, q in enumerate(locs):
        tg.append(stim.target_x(base + q) if typ == "X" else stim.target_z(base + q))
        if k < len(locs) - 1:
            tg.append(stim.target_combiner())
    return tg


def build_cnot_circuit(rounds_before=2, rounds_after=2, p=0.005, noise=True,
                       control_one=False, target_one=False, remap=True):
    """
    制御・標的とも Z 基底メモリ。両者を rounds_before 回安定化 -> transversal CX
    -> rounds_after 回安定化 -> Z 測定。control_one/target_one で初期 |1_L>。
    返り値: (circuit, n_meas)  ※データ測定は末尾 18 レコード(制御9, 標的9)。
    """
    pp = p if noise else 0.0
    c = stim.Circuit()
    dc = [BC + i for i in range(NDATA)]
    dt = [BT + i for i in range(NDATA)]
    c.append("R", dc + dt)
    if control_one:
        c.append("X", [BC + q for q in XL])   # |1_L> on control
    if target_one:
        c.append("X", [BT + q for q in XL])   # |1_L> on target

    mc = 0
    ridx = {}  # (round, patch, stab) -> 絶対測定 index ; patch in {'c','t'}

    def syndrome_round(r):
        nonlocal mc
        c.append("DEPOLARIZE1", dc + dt, pp)
        for si, (typ, sup) in enumerate(STABS):
            c.append("MPP", _mpp(typ, sup, BC), pp); ridx[(r, 'c', si)] = mc; mc += 1
            c.append("MPP", _mpp(typ, sup, BT), pp); ridx[(r, 't', si)] = mc; mc += 1

    R = rounds_before + rounds_after
    for r in range(rounds_before):
        syndrome_round(r)

    # --- transversal CNOT (control -> target) ---
    for i in range(NDATA):
        c.append("CX", [BC + i, BT + i])
    if noise:
        for i in range(NDATA):
            c.append("DEPOLARIZE2", [BC + i, BT + i], pp)

    for r in range(rounds_before, R):
        syndrome_round(r)

    c.append("X_ERROR", dc + dt, pp)
    c.append("M", dc + dt)
    drec = {}
    for i in range(NDATA):
        drec[('c', i)] = mc; mc += 1
    for i in range(NDATA):
        drec[('t', i)] = mc; mc += 1
    rr = lambda a: stim.target_rec(a - mc)

    # ---------- 検出器 ----------
    # round 0: Z 安定化子は決定論的
    for si, (typ, sup) in enumerate(STABS):
        if typ == "Z":
            c.append("DETECTOR", [rr(ridx[(0, 'c', si)])])
            c.append("DETECTOR", [rr(ridx[(0, 't', si)])])
    # round 1..R-1: 各安定化子を前ラウンドと比較。CX 直後ラウンドはクロス境界補正。
    for r in range(1, R):
        boundary = (r == rounds_before)
        for si, (typ, sup) in enumerate(STABS):
            for patch in ('c', 't'):
                tg = [rr(ridx[(r, patch, si)]), rr(ridx[(r - 1, patch, si)])]
                if remap and boundary:
                    if typ == "X" and patch == 'c':       # S_X^C -> S_X^C·S_X^T
                        tg.append(rr(ridx[(r - 1, 't', si)]))
                    if typ == "Z" and patch == 't':       # S_Z^T -> S_Z^C·S_Z^T
                        tg.append(rr(ridx[(r - 1, 'c', si)]))
                c.append("DETECTOR", tg)
    # 最終: データから Z 安定化子を再構成し最後の MPP と比較
    for si, (typ, sup) in enumerate(STABS):
        if typ == "Z":
            for patch in ('c', 't'):
                tg = [rr(ridx[(R - 1, patch, si)])] + [rr(drec[(patch, q)]) for q in sup]
                c.append("DETECTOR", tg)
    # 観測量: obs0 = 制御 Z_L, obs1 = 標的 Z_L
    c.append("OBSERVABLE_INCLUDE", [rr(drec[('c', q)]) for q in ZL], 0)
    c.append("OBSERVABLE_INCLUDE", [rr(drec[('t', q)]) for q in ZL], 1)
    return c, mc


# ---------------------------------------------------------------- 検証関数
def verify_logical_action():
    """[A] transversal CX が論理 CNOT であることを Tableau で代数確認。"""
    circ = stim.Circuit()
    for i in range(NDATA):
        circ.append("CX", [i, NDATA + i])
    tab = circ.to_tableau()

    def ps(xs=(), zs=()):
        s = ["_"] * (2 * NDATA)
        for q in xs: s[q] = "X"
        for q in zs: s[q] = "Z"
        return stim.PauliString("".join(s))

    checks = {
        "X_L^C -> X_L^C·X_L^T": (tab(ps(xs=XL)), ps(xs=XL + [NDATA + q for q in XL])),
        "Z_L^T -> Z_L^C·Z_L^T": (tab(ps(zs=[NDATA + q for q in ZL])), ps(zs=ZL + [NDATA + q for q in ZL])),
        "X_L^T -> X_L^T (不変)": (tab(ps(xs=[NDATA + q for q in XL])), ps(xs=[NDATA + q for q in XL])),
        "Z_L^C -> Z_L^C (不変)": (tab(ps(zs=ZL)), ps(zs=ZL)),
    }
    ok = True
    for label, (got, exp) in checks.items():
        match = (got == exp)
        ok &= match
        print(f"    {label}: {'OK' if match else 'NG'}")
    return ok


def _logical_z(ref, n_meas, patch):
    base = n_meas - 18 + (0 if patch == 'c' else 9)
    bits = ref[base:base + NDATA]
    return int(bits[ZL].sum() % 2)


def verify_truth_table():
    """[B] ノイズなしで CNOT 真理値表 (control 不変, target <- c xor t)。"""
    ok = True
    for c1 in (False, True):
        for t1 in (False, True):
            circ, nm = build_cnot_circuit(noise=False, control_one=c1, target_one=t1)
            ref = np.array(circ.without_noise().reference_sample(), int)
            cz = _logical_z(ref, nm, 'c')
            tz = _logical_z(ref, nm, 't')
            exp_t = int(c1) ^ int(t1)
            good = (cz == int(c1)) and (tz == exp_t)
            ok &= good
            print(f"    in(c={int(c1)},t={int(t1)}) -> out(c={cz},t={tz})  "
                  f"期待(c={int(c1)},t={exp_t})  {'OK' if good else 'NG'}")
    return ok


def verify_boundary_remap():
    """[C] クロス境界補正の有無で検出器の決定論性が変わることを確認。"""
    res = {}
    for rm in (False, True):
        circ, _ = build_cnot_circuit(noise=False, control_one=True, remap=rm)
        det, _ = circ.compile_detector_sampler().sample(100, separate_observables=True)
        res[rm] = not det.any()
        print(f"    remap={rm}: ノイズなし検出器が全0 = {res[rm]}")
    return (res[True] is True) and (res[False] is False)


def decode(circ, shots=40000):
    """[D] サンプリング -> MWPM デコード -> 各論理量子ビットの論理エラー率。"""
    det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
    dem = circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred = matcher.decode_batch(det)
    res = obs ^ pred
    return res[:, 0].mean(), res[:, 1].mean()


# ------------------------------------------- qulacs 論理回路からの入口 (例)
def from_qulacs(qulacs_circuit, **kw):
    """qulacs の 2 論理量子ビット回路から CNOT(0->1) を読み、物理回路を生成。"""
    cnot = None
    for g in range(qulacs_circuit.get_gate_count()):
        gate = qulacs_circuit.get_gate(g)
        if gate.get_name() in ("CNOT", "CX"):
            ctrl = gate.get_control_index_list()[0]
            tgt = gate.get_target_index_list()[0]
            cnot = (ctrl, tgt)
    if cnot is None:
        raise ValueError("qulacs 回路に CNOT が見つかりません。")
    print(f"    qulacs CNOT(control={cnot[0]} -> target={cnot[1]}) を transversal CX に変換")
    return build_cnot_circuit(**kw)


def main():
    print("#" * 66)
    print("#  論理 CNOT を transversal CNOT で Stim 内に実装")
    print("#  rotated surface code d=3 x2 パッチ / MPP 安定化子測定")
    print("#" * 66)

    print("\n[A] 論理作用の代数確認 (Tableau)")
    a = verify_logical_action()

    print("\n[B] 論理 CNOT 真理値表 (ノイズなし)")
    b = verify_truth_table()

    print("\n[C] クロス境界 検出器補正の必要性")
    c = verify_boundary_remap()

    print("\n[D] ノイズあり実行 + MWPM デコード")
    for p in (0.002, 0.005, 0.01):
        circ, _ = build_cnot_circuit(p=p, control_one=True, noise=True)
        ec, et = decode(circ, shots=40000)
        print(f"    p={p}: 論理エラー率  制御={ec:.5f}  標的={et:.5f}")

    print("\n[サマリ]")
    print(f"    [A] 論理作用=CNOT : {'OK' if a else 'NG'}")
    print(f"    [B] 真理値表      : {'OK' if b else 'NG'}")
    print(f"    [C] 境界補正      : {'OK' if c else 'NG'}")
    print("    => transversal CX による論理 CNOT を、検出器補正込みで")
    print("       Stim 内に実装し、ノイズ下のデコードまで確認した。")


if __name__ == "__main__":
    main()
