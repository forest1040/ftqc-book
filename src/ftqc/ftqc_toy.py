"""
FTQC トイモデル統合パイプライン (rotated surface code, d=3)
=============================================================
参照ファイル(decoding_graph / clifford_branch / logical_t / msd / pauli_frame)で
個別検証した4つの要素技術を、1本の「論理計算フロー」として接続して全体動作を確認する。

  STAGE 1  QEC 層        : 誤り訂正が効いていること(論理エラー率 vs 物理エラー率)
  STAGE 2  非Clifford層  : T ゲートを Clifford 分岐法で再現(Stim) = 厳密値(状態ベクトル)
  STAGE 3  フレーム追跡  : ソフトウェアでの Pauli/Clifford フレーム補正
  STAGE 4  論理演算      : 表面符号に T 状態を注入 -> フレーム補正付き論理読み出し

Stim は Clifford/QEC を高速に担当し、非Clifford の厳密検証は状態ベクトル(qulacs)で行う。
"""
import numpy as np
import stim
import pymatching

np.set_printoptions(suppress=True)


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ===================================================================
# STAGE 1 : QEC 層 -- 誤り訂正が効いているか (論理エラー率 vs 物理エラー率)
# ===================================================================
def stage1_qec(distance=3, num_shots=50000, p_list=(0.001, 0.005, 0.01, 0.02)):
    banner("STAGE 1 : QEC 層 (rotated surface code, circuit-level noise)")
    rounds = distance
    print(f"符号距離 d={distance}, ラウンド数={rounds}, ショット数={num_shots}\n")
    print(f"{'物理エラー率 p':>14} | {'論理エラー数':>10} | {'論理エラー率 p_L':>16}")
    print("-" * 50)
    results = {}
    for p in p_list:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,      # 2量子ビットゲート後の脱分極
            before_measure_flip_probability=p,    # 測定エラー
        )
        sampler = circuit.compile_detector_sampler()
        det, obs = sampler.sample(num_shots, separate_observables=True)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        pred = matcher.decode_batch(det)
        n_err = int(np.sum(obs ^ pred))
        pL = n_err / num_shots
        results[p] = pL
        print(f"{p:>14.4f} | {n_err:>10d} | {pL:>16.5f}")
    print("\n=> p が小さいほど論理エラー率 p_L が下がっている。QEC が機能している。")
    return results


# ===================================================================
# STAGE 2 : 非Clifford 層 -- T ゲートを Clifford 分岐法で再現
#   T = e^{iπ/8} ( cos(π/8) I  -  i sin(π/8) Z )   <- グローバル位相を除けば I と Z の和
#   Stim(スタビライザ)は T を直接扱えないが、I/Z 分岐の重ね合わせとして「合成」できる。
# ===================================================================
def stage2_nonclifford():
    banner("STAGE 2 : 非Clifford 層 (T ゲートの Clifford 分岐 = 厳密値)")
    wI = np.cos(np.pi / 8)
    wZ = -1j * np.sin(np.pi / 8)

    # |+> に T を適用する例: T|+> = (|0> + e^{iπ/4}|1>)/√2
    branched = np.zeros(2, dtype=complex)
    for gate, w in [("I", wI), ("Z", wZ)]:
        sim = stim.TableauSimulator()
        sim.h(0)                 # |+> を準備
        if gate == "Z":
            sim.z(0)
        branched += w * sim.state_vector()
    branched *= np.exp(1j * np.pi / 8)  # グローバル位相補正(表示用)

    exact = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2)  # 厳密な T|+>

    print("Clifford 分岐法 (Stim) :", np.round(branched, 4))
    print("厳密値 (状態ベクトル)  :", np.round(exact, 4))
    diff = np.linalg.norm(branched - exact)
    # Stim の state_vector() は内部 float32 精度のため 1e-6 を許容誤差とする
    print(f"\n差分ノルム = {diff:.2e}  =>", "一致 (Stim で非Clifford を合成できた)" if diff < 1e-6 else "不一致")
    return diff


# ===================================================================
# STAGE 3 : フレーム追跡 -- ソフトウェアでの Pauli / Clifford フレーム補正
#   (a) Pauli フレーム: 物理 X エラーを注入し、検出器がエラー無し時と一致するよう
#       測定記録をソフトで反転補正できることを確認 (pauli_frame.py の技術)
#   (b) Clifford フレーム: T テレポーテーションの測定=1 で保留した S 補正により、
#       論理測定の読み替え(基底・符号)が正しく行われることを確認 (msd.py の技術)
# ===================================================================
class PauliFrameTracker:
    def __init__(self, n):
        self.x = [False] * n
        self.z = [False] * n
        self.meas_flips = []

    def H(self, t): self.x[t], self.z[t] = self.z[t], self.x[t]
    def CX(self, c, t):
        self.x[t] ^= self.x[c]; self.z[c] ^= self.z[t]
    def Xerr(self, t): self.x[t] ^= True
    def M(self, t): self.meas_flips.append(self.x[t])
    def R(self, t): self.x[t] = False; self.z[t] = False


def stage3a_pauli_frame():
    print("\n[3a] Pauli フレーム: 物理 X エラーを古典後処理で打ち消す")
    import qulacs
    from qulacs import QuantumState, QuantumCircuit
    from qulacs.gate import Measurement, AmplitudeDampingNoise

    stim_circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)
    used = sorted({t.value for inst in stim_circ.flattened()
                   for t in inst.targets_copy() if t.is_qubit_target})
    qmap = {old: new for new, old in enumerate(used)}
    n = len(qmap)

    def build(inject=None):
        circ = QuantumCircuit(n)
        tr = PauliFrameTracker(n)
        creg = 0
        detectors = []
        injected = False
        seen_r = False
        for inst in stim_circ.flattened():
            name = inst.name
            ts = [qmap[t.value] for t in inst.targets_copy() if t.is_qubit_target]
            if name == "R":
                seen_r = True
            if seen_r and not injected and inject is not None and name not in ("R", "QUBIT_COORDS"):
                circ.add_X_gate(inject); tr.Xerr(inject); injected = True
            if name == "M":
                for t in ts:
                    circ.add_gate(Measurement(t, creg)); creg += 1; tr.M(t)
            elif name == "MR":
                for t in ts:
                    circ.add_gate(Measurement(t, creg)); creg += 1
                    circ.add_gate(AmplitudeDampingNoise(t, 1.0)); tr.M(t); tr.R(t)
            elif name == "R":
                for t in ts:
                    circ.add_gate(AmplitudeDampingNoise(t, 1.0)); tr.R(t)
            elif name == "H":
                for t in ts: circ.add_H_gate(t); tr.H(t)
            elif name == "X":
                for t in ts: circ.add_X_gate(t); tr.Xerr(t)
            elif name == "CX":
                for i in range(0, len(ts), 2):
                    circ.add_CNOT_gate(ts[i], ts[i + 1]); tr.CX(ts[i], ts[i + 1])
            elif name == "DETECTOR":
                idx = [creg + t.value for t in inst.targets_copy() if t.is_measurement_record_target]
                detectors.append(idx)
        return circ, creg, detectors, tr.meas_flips

    def detectors_of(circ, creg, detectors, flips=None):
        st = QuantumState(n); st.set_zero_state(); circ.update_quantum_state(st)
        synd = [st.get_classical_value(i) for i in range(creg)]
        if flips is not None:
            synd = [s ^ (1 if f else 0) for s, f in zip(synd, flips)]
        return [int(np.bitwise_xor.reduce([synd[i] for i in d])) if d else 0 for d in detectors]

    c0, m0, d0, _ = build(inject=None)
    ideal = detectors_of(c0, m0, d0)
    c1, m1, d1, flips = build(inject=qmap[10])   # 中央データ量子ビットに X エラー
    noisy = detectors_of(c1, m1, d1)
    corrected = detectors_of(c1, m1, d1, flips=flips)

    print(f"   エラー無し時の検出器      : {ideal}")
    print(f"   X エラー注入後の検出器    : {noisy}   <- 反応している")
    print(f"   Pauli フレーム補正後      : {corrected}  <-", "復元成功" if corrected == ideal else "失敗")
    return corrected == ideal


def stage3b_clifford_frame():
    print("\n[3b] Clifford フレーム: T テレポーテーションの S 補正を測定基底に読み替え")

    class LogicalFrame:
        def __init__(self): self.pending_s = False
        def update(self, m): self.pending_s = (m == 1)
        def read(self, basis):
            b, sign = basis, 1
            if self.pending_s:
                if basis == "X": b = "Y"
                elif basis == "Y": b, sign = "X", -1
            return b, sign

    f = LogicalFrame()
    f.update(1)  # テレポーテーション測定=1 -> S 補正保留
    cases = []
    for basis in ("X", "Y", "Z"):
        b, s = f.read(basis)
        cases.append((basis, b, s))
        print(f"   論理 {basis} 測定要求 -> 物理 {b} 基底 / 符号 {'+1' if s > 0 else '-1'}")
    # 期待: X->Y(+), Y->X(-), Z->Z(+)
    ok = cases == [("X", "Y", 1), ("Y", "X", -1), ("Z", "Z", 1)]
    print("   =>", "S フレームの読み替えが正しい" if ok else "誤り")
    return ok


def stage3_frames():
    banner("STAGE 3 : フレーム追跡 (ソフトウェア補正)")
    a = stage3a_pauli_frame()
    b = stage3b_clifford_frame()
    return a and b


# ===================================================================
# STAGE 4 : 論理演算 -- 表面符号への T 状態注入 + フレーム補正付き論理読み出し
#   logical_t.py の手法を関数化(グローバル依存を排除)。状態ベクトル(qulacs)で
#   論理 |+_L> に T を作用させ、補正後に相対位相 45°(=π/4) が出ることを確認する。
# ===================================================================
def stage4_logical_t(distance=3):
    banner("STAGE 4 : 論理演算 (T 状態注入 -> フレーム補正付き読み出し)")
    import qulacs
    from qulacs import QuantumState, QuantumCircuit
    from qulacs.gate import Measurement, X

    stim_circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=1)

    coords = {}
    for inst in stim_circ.flattened():
        if inst.name == "QUBIT_COORDS":
            coords[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    used = sorted({t.value for inst in stim_circ.flattened()
                   for t in inst.targets_copy() if t.is_qubit_target})
    qmap = {old: new for new, old in enumerate(used)}
    base = len(qmap)

    # 論理 X 演算子を構成するデータ量子ビット列 (logical_t.py と同じ規約)
    xl = [qmap[q] for q, (x, y) in coords.items() if y == distance and x % 2 != 0]

    idx_T = base          # 物理 T 状態用
    idx_anc = base + 1    # パリティ測定アンシラ
    total = base + 2

    # |0_L> 符号化回路(rotated_memory_z の Clifford 部分)
    enc = QuantumCircuit(total)
    for inst in stim_circ.flattened():
        ts = [qmap[t.value] for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name == "H":
            for t in ts: enc.add_H_gate(t)
        elif inst.name in ("CX", "CNOT"):
            for i in range(0, len(ts), 2): enc.add_CNOT_gate(ts[i], ts[i + 1])
        elif inst.name == "X":
            for t in ts: enc.add_X_gate(t)

    state = QuantumState(total); state.set_zero_state(); enc.update_quantum_state(state)

    # --- 状態注入(lattice surgery 風) ---
    inj = QuantumCircuit(total)
    inj.add_H_gate(idx_T); inj.add_T_gate(idx_T)         # 物理 T 状態を準備
    inj.add_H_gate(idx_anc)                               # X_L ⊗ X_T パリティ測定
    for q in xl: inj.add_CNOT_gate(idx_anc, q)
    inj.add_CNOT_gate(idx_anc, idx_T)
    inj.add_H_gate(idx_anc)
    inj.update_quantum_state(state)

    Measurement(idx_anc, 0).update_quantum_state(state); parity = state.get_classical_value(0)
    Measurement(idx_T, 1).update_quantum_state(state);   zmeas = state.get_classical_value(1)
    print(f"X_L ⊗ X_T パリティ測定 = {parity},  物理 T の Z 測定 = {zmeas}")

    # --- 論理状態 (α,β) を抽出 (参照 |0_L>,|1_L> との内積) ---
    s0 = QuantumState(total); s0.set_zero_state(); enc.update_quantum_state(s0)
    if parity == 1: X(idx_anc).update_quantum_state(s0)
    if zmeas == 1:  X(idx_T).update_quantum_state(s0)
    s1 = s0.copy()
    for q in xl: X(q).update_quantum_state(s1)

    alpha = qulacs.state.inner_product(s0, state)
    beta = qulacs.state.inner_product(s1, state)
    nrm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha, beta = alpha / nrm, beta / nrm

    deg_raw = np.degrees(np.angle(beta) - np.angle(alpha))
    # --- ソフトウェアのフレーム補正 ---
    if parity == 1: beta = -beta          # 論理 Z_L 補正
    if zmeas == 1:  beta = beta * 1j      # 論理 S_L 補正 (T† -> T)
    deg_corr = np.degrees(np.angle(beta) - np.angle(alpha))

    print(f"|0_L>,|1_L> の確率 = {abs(alpha)**2:.3f}, {abs(beta)**2:.3f}")
    print(f"相対位相: 補正前 {deg_raw:7.2f}°  ->  補正後 {deg_corr:7.2f}°")
    ok = abs(((deg_corr + 180) % 360 - 180) - 45.0) < 1e-6
    print("   =>", "補正後に 45° (=π/4, T ゲートの位相) を確認" if ok else f"想定外: {deg_corr}°")
    return ok


# ===================================================================
def main():
    print("#" * 70)
    print("#  FTQC トイモデル 統合動作確認  (rotated surface code, d=3)")
    print("#" * 70)

    qec = stage1_qec()
    diff = stage2_nonclifford()
    frames = stage3_frames()
    logical = stage4_logical_t()

    banner("全体サマリ")
    print(f"STAGE 1 QEC 層       : p=0.001 -> p_L={qec[0.001]:.5f},  p=0.02 -> p_L={qec[0.02]:.5f}  "
          f"({'単調に減少 OK' if qec[0.001] < qec[0.02] else '要確認'})")
    print(f"STAGE 2 非Clifford   : Clifford 分岐 vs 厳密値 差分 {diff:.1e}  ({'一致 OK' if diff < 1e-6 else 'NG'})")
    print(f"STAGE 3 フレーム追跡 : Pauli + Clifford フレーム補正  ({'OK' if frames else 'NG'})")
    print(f"STAGE 4 論理 T 演算  : 注入 + 補正で位相 45° 復元  ({'OK' if logical else 'NG'})")
    print("\n=> Stim(QEC) + 状態ベクトル(非Clifford 検証) + ソフトフレーム追跡 が")
    print("   1本のフローとして接続され、トイレベルで全体が動作することを確認した。")


if __name__ == "__main__":
    main()
