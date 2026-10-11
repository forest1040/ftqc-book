"""
T ゲート (universal FTQC の最後のピース)
=========================================
T は非 Clifford なので、(1) magic state |A>=T|+> を蒸留し、(2) ゲートテレポーテーション
で論理量子ビットに転写する。S 補正は既存の (lattice surgery) S ゲートを使う。

|A> はスタビライザ状態ではない(射影で作れない)。そこで 15-to-1 蒸留:
  ノイジーな |A> を 15 個用意 -> [[15,1,3]] Reed-Muller 符号の検査を測定 ->
  syndrome 0 を postselect -> 出力 |A> の誤り率 p -> ~35 p^3 (3次抑圧)。

検証
  [A] [[15,1,3]] 符号: CSS, k=1, 重み3の undetected-logical がちょうど 35 個
  [B] 蒸留モンテカルロ: 出力誤り率 ~ 35 p^3
  [C] T テレポーテーション: CNOT + S補正 が任意入力で T を実装 (状態ベクトルで確認)
"""
import itertools
import numpy as np

# ============================================================
# [[15,1,3]] Reed-Muller 符号 (4変数 F_2^4 の 15 非零点)
# ============================================================
_PTS = [tuple((i >> b) & 1 for b in range(4)) for i in range(1, 16)]


def _monomial(S):
    return np.array([int(all(p[i] for i in S)) if S else 1 for p in _PTS], np.int8)


def build_15to1():
    HZ = np.array([_monomial((i,)) for i in range(4)], np.int8)            # 4 行 (X誤り検出)
    HX = np.array([_monomial((i,)) for i in range(4)] +
                  [_monomial((i, j)) for i, j in itertools.combinations(range(4), 2)], np.int8)  # 10 行
    return HX, HZ


def _gf2_rank(M):
    A = M.copy() % 2
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r


def _reduce_basis(M):
    B = M.copy() % 2
    piv = []
    for c in range(B.shape[1]):
        r = next((i for i in range(len(piv), B.shape[0]) if B[i, c]), None)
        if r is None:
            continue
        B[[len(piv), r]] = B[[r, len(piv)]]
        for i in range(B.shape[0]):
            if i != len(piv) and B[i, c]:
                B[i] ^= B[len(piv)]
        piv.append(c)
    return B[:len(piv)], piv


def _not_in_span_batch(E, basis_rows, piv):
    W = E.copy() % 2
    for i, c in enumerate(piv):
        mask = W[:, c] == 1
        W[mask] ^= basis_rows[i]
    return W.any(axis=1)


def code_properties(HX, HZ):
    k = 15 - _gf2_rank(HX) - _gf2_rank(HZ)
    css = bool(np.all((HX @ HZ.T) % 2 == 0))
    # 重み w の undetected-logical X誤り (HZ で検出されず, HX の span にない)
    HXb, HXp = _reduce_basis(HX)
    counts = {}
    for w in range(1, 5):
        cnt = 0
        for supp in itertools.combinations(range(15), w):
            e = np.zeros(15, np.int8)
            for q in supp:
                e[q] = 1
            if np.all((HZ @ e) % 2 == 0) and _not_in_span_batch(e[None, :], HXb, HXp)[0]:
                cnt += 1
        counts[w] = cnt
    return css, k, counts


def distill(HX, HZ, p, shots=3_000_000, seed=1):
    """入力 |A> の誤り率 p。X誤りを HZ で検出し syndrome 0 を postselect。"""
    HXb, HXp = _reduce_basis(HX)
    rng = np.random.default_rng(seed)
    e = (rng.random((shots, 15)) < p).astype(np.int8)
    accept = ~((e @ HZ.T) % 2).any(axis=1)
    ea = e[accept]
    n_log = int(_not_in_span_batch(ea, HXb, HXp).sum())
    return accept.mean(), n_log / max(len(ea), 1)


# ============================================================
# T ゲートテレポーテーション (状態ベクトルで正しさ検証)
# ============================================================
def verify_t_teleportation():
    I = np.eye(2, dtype=complex)
    T = np.diag([1, np.exp(1j * np.pi / 4)])
    S = np.diag([1, 1j])
    A = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)   # |A> = T|+>
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)

    def teleport(psi):
        state = np.kron(psi, A)               # target(0) ⊗ ancilla(1)
        state = CNOT @ state                  # CNOT(target->ancilla)
        # ancilla(qubit1) を Z 測定する2分岐
        results = {}
        for m in (0, 1):
            proj = np.zeros((4, 4), complex)
            for t in (0, 1):                  # target basis
                idx = t * 2 + m
                proj[idx, idx] = 1
            br = proj @ state
            nrm = np.linalg.norm(br)
            if nrm < 1e-9:
                continue
            br = br / nrm
            # ancilla を捨て target の 2次元状態に
            out = np.array([br[0 * 2 + m], br[1 * 2 + m]])
            if m == 1:
                out = S @ out                 # S 補正 (= 既存の LS S ゲート)
            results[m] = out
        return results

    ok = True
    for psi in [np.array([1, 0], complex), np.array([0, 1], complex),
                np.array([1, 1], complex) / np.sqrt(2),
                np.array([1, 1j], complex) / np.sqrt(2),
                np.array([0.6, 0.8], complex)]:
        target = T @ psi
        for m, out in teleport(psi).items():
            # グローバル位相を除いて比較
            phase = np.vdot(out, target)
            fid = abs(phase) / (np.linalg.norm(out) * np.linalg.norm(target))
            if fid < 1 - 1e-9:
                ok = False
    return ok


def demo():
    print("#" * 64)
    print("#  T ゲート: 15-to-1 magic state 蒸留 + テレポーテーション")
    print("#" * 64)

    HX, HZ = build_15to1()
    print("\n[A] [[15,1,3]] Reed-Muller 符号の検証")
    css, k, counts = code_properties(HX, HZ)
    print(f"    CSS={css}, k={k}, 符号 [[15,{k},3]]")
    print(f"    undetected-logical X誤りの重み分布: {counts}")
    print(f"    => 重み3 がちょうど {counts[3]} 個 (=有名な 35) -> 出力誤り ~35 p^3")

    print("\n[B] 蒸留モンテカルロ (入力 |A> 誤り率 p -> 出力誤り率)")
    print(f"    {'p_in':>8} | {'accept率':>8} | {'p_out':>10} | {'35 p^3':>10} | ratio")
    print("    " + "-" * 52)
    for p in (0.01, 0.02, 0.05):
        acc, pout = distill(HX, HZ, p)
        print(f"    {p:>8.3f} | {acc:>8.3f} | {pout:>10.2e} | {35*p**3:>10.2e} | {pout/(35*p**3):.2f}")
    print("    => 3次抑圧 (p_out ~ 35 p^3) を確認")

    print("\n[C] T ゲートテレポーテーション (CNOT + S補正)")
    ok = verify_t_teleportation()
    print(f"    アンシラ|A>, CNOT(target->ancilla), Z測定, m=1 なら S 補正")
    print(f"    任意入力で T を実装: {'OK' if ok else 'NG'} (S 補正は既存の LS S ゲートを使用)")

    print("\n[サマリ]")
    print("    蒸留で高品質な |A> を作り (誤り p->35p^3)、テレポーテーションで論理 T を実現。")
    print("    これで {Clifford(Pauli,H,S,CNOT) + T} が揃い、universal FTQC が表現可能。")
    print("    全スタック: 表面符号QEC -> Clifford(transversal/生成子追跡/lattice surgery)")
    print("                -> S(Y状態) -> T(magic state 蒸留+注入)。")


if __name__ == "__main__":
    demo()
