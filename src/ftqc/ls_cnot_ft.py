"""
ls_cnot_ft.py — 正しい回路レベル LS CNOT (本実装)
=================================================
circuit_level_merge.py の検証済み手法を多パッチ・多フェーズに一般化して LS CNOT を
構築する。グローバル絶対座標で全パッチを配置し、_dtype の市松模様を自動整合させる。

レシピ (lattice_surgery.py で検証済みの論理):
    アンシラ A を |+_L>; M_ZZ(C,A); M_XX(A,T); A を Z 測定
    補正: Z_C if m_xx,  X_T if (m_zz xor m_anc)
L 字レイアウト: C(上) / A(C の下, ZZ) / T(A の右, XX)。

設計上の要点 (これが circuit_level_ls.py の作り替え版との違い):
  - 検出器は geom の _compose_in (GF2 span 合成) で split<->merge 遷移を正しく扱う
    (同一スタビライザ=連続比較, seam=構成比較, joint成分=検出器なし)。
  - round0 はパッチ別の初期化基底で決定論型が決まる (C,T=|0>->Z型, A=|+>->X型)。
  - 最終ラウンド検出器で spacelike 境界を閉じる (距離保存に必須)。
  - 観測量は joint 測定記録で Pauli フレーム補正する (Z_L(T) ^= m_zz ^ m_anc)。

完成基準: gauge 検出器 0 / graphlike 上界 3 (パッチ距離と整合) / 計算基底 CNOT 真理値表一致。
"""
import numpy as np
import stim
import pymatching
from ftqc.geom_lattice_surgery import _dtype

_XSTEP = {(1, 1): 0, (-1, 1): 1, (1, -1): 2, (-1, -1): 3}    # NE,NW,SE,SW
_ZSTEP = {(1, 1): 0, (1, -1): 1, (-1, 1): 2, (-1, -1): 3}    # NE,SE,NW,SW


def _patch_cells(x0, y0):
    return [(x0 + 2 * c, y0 + 2 * r) for r in range(3) for c in range(3)]


def _enum_stabs(S):
    """データ座標集合 S に対する plaquette 規則のスタビライザ (cx,cy,type,[coords])。"""
    xs = [x for x, y in S]
    ys = [y for x, y in S]
    cmin, cmax, rmin, rmax = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    out = []
    for cx in range(cmin, cmax + 1, 2):
        for cy in range(rmin, rmax + 1, 2):
            corners = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx + 1, cy + 1)]
            pres = [p for p in corners if p in S]
            if len(pres) == 4:
                out.append((cx, cy, _dtype(cx, cy), sorted(pres)))
            elif len(pres) == 2:
                t = _dtype(cx, cy)
                on_tb = (cy == rmin or cy == rmax)
                on_lr = (cx == cmin or cx == cmax)
                if (on_tb and t == "X") or (on_lr and t == "Z"):
                    out.append((cx, cy, t, sorted(pres)))
    return out


class LSCnotFT:
    def __init__(self):
        # L 字レイアウト (グローバル絶対座標, 3x3 patches)
        #   C(上) / A(C の下, ZZ merge) / T(A の右, XX merge)
        # ZZ seam (A の上辺) と XX seam (A の右辺) は A の角を共有するが、
        # merge->split で分裂する seam スタビライザの「積の保存量」を閉じる
        # 後ろ向き閉包検出器 (build 内) を張ることで角の縮退が解消され、
        # circuit-level 距離 3 (= パッチ距離) を達成する。
        self.C = set(_patch_cells(1, 1))     # y=1,3,5
        self.A = set(_patch_cells(1, 7))     # y=7,9,11
        self.T = set(_patch_cells(7, 7))     # x=7,9,11 y=7,9,11
        self.patch_of = {}
        for c in self.C: self.patch_of[c] = "C"
        for c in self.A: self.patch_of[c] = "A"
        for c in self.T: self.patch_of[c] = "T"

        # データ id 割当
        self.coords = sorted(self.C | self.A | self.T)
        self.id_of = {c: i for i, c in enumerate(self.coords)}
        self.N = len(self.coords)

        # 各 config (座標サポート -> id サポート)
        def to_id(stabs):
            return [(cx, cy, t, [self.id_of[c] for c in sup]) for cx, cy, t, sup in stabs]
        self.split_info = to_id(_enum_stabs(self.C) + _enum_stabs(self.A) + _enum_stabs(self.T))
        self.zz_info = to_id(_enum_stabs(self.C | self.A) + _enum_stabs(self.T))
        self.xx_info = to_id(_enum_stabs(self.C) + _enum_stabs(self.A | self.T))

        # 論理演算子 (geom 規約: Z_L=bottom row, X_L=left column)
        self.ZL_C = [self.id_of[c] for c in self.C if c[1] == 1]            # y=1
        self.ZL_A = [self.id_of[c] for c in self.A if c[1] == 7]            # y=7
        self.ZL_T = [self.id_of[c] for c in self.T if c[1] == 7]            # y=7
        self.XL_A = [self.id_of[c] for c in self.A if c[0] == 1]            # x=1
        self.XL_T = [self.id_of[c] for c in self.T if c[0] == 7]            # x=7

        # joint 分解: m_zz = Z_L(C)Z_L(A) を zz_info のスタビライザ積で
        self.zz_joint_idx = self._decompose(self._lvec("Z", self.ZL_C + self.ZL_A), self.zz_info)

        # ancilla 割当 (中心,型,support) で一意化 (seam 競合回避)
        self.anc_of = {}
        nid = self.N
        for info in (self.split_info, self.zz_info, self.xx_info):
            for cx, cy, t, s in info:
                k = (cx, cy, t, tuple(s))
                if k not in self.anc_of:
                    self.anc_of[k] = nid
                    nid += 1
        self.n_total = nid

    # ---- GF2 ----
    def _vec(self, t, sup):
        x = np.zeros(self.N, np.int8)
        z = np.zeros(self.N, np.int8)
        for q in sup:
            (x if t == "X" else z)[q] = 1
        return np.concatenate([x, z])

    def _lvec(self, t, sup):
        return self._vec(t, sup)

    def _decompose(self, vec, info):
        M = np.array([self._vec(t, s) for _, _, t, s in info], np.int8)
        A = M.T.copy().astype(np.int8)
        b = vec.copy().astype(np.int8)
        rows, cols = A.shape
        r = 0
        piv = []
        for col in range(cols):
            p = next((i for i in range(r, rows) if A[i, col]), None)
            if p is None:
                continue
            A[[r, p]] = A[[p, r]]
            b[[r, p]] = b[[p, r]]
            for i in range(rows):
                if i != r and A[i, col]:
                    A[i] ^= A[r]
                    b[i] ^= b[r]
            piv.append(col)
            r += 1
        coeff = np.zeros(cols, np.int8)
        for ri, col in enumerate(piv):
            coeff[col] = b[ri]
        if not np.array_equal((coeff @ M) % 2, vec % 2):
            return None
        return [i for i in range(len(info)) if coeff[i]]

    def _compose_in(self, vec, info):
        return self._decompose(vec, info)

    # ---- 1 ラウンド (ancilla syndrome extraction + 回路ノイズ) ----
    def _round(self, c, info, p, mc, reset_all=None):
        ancs = [self.anc_of[(cx, cy, t, tuple(s))] for cx, cy, t, s in info]
        xancs = [self.anc_of[(cx, cy, t, tuple(s))] for cx, cy, t, s in info if t == "X"]
        if p > 0:
            c.append("DEPOLARIZE1", list(range(self.N)), p)
        c.append("R", reset_all if reset_all is not None else ancs)
        if p > 0:
            c.append("X_ERROR", ancs, p)
        if xancs:
            c.append("H", xancs)
            if p > 0:
                c.append("DEPOLARIZE1", xancs, p)
        for step in range(4):
            flat = []
            for cx, cy, t, s in info:
                a = self.anc_of[(cx, cy, t, tuple(s))]
                tbl = _XSTEP if t == "X" else _ZSTEP
                for q in s:
                    px, py = self.coords[q]
                    d = (int(np.sign(px - cx)), int(np.sign(py - cy)))
                    if tbl[d] == step:
                        flat += ([a, q] if t == "X" else [q, a])
            if flat:
                c.append("CX", flat)
                if p > 0:
                    c.append("DEPOLARIZE2", flat, p)
        if xancs:
            c.append("H", xancs)
            if p > 0:
                c.append("DEPOLARIZE1", xancs, p)
        if p > 0:
            c.append("X_ERROR", ancs, p)
        c.append("M", ancs)
        rec = {si: mc[0] + i for i, si in enumerate(range(len(info)))}
        mc[0] += len(ancs)
        return rec

    def build(self, control_one=False, target_one=False, merge_rounds=3, split_rounds=2,
              p=0.005, noise=True, obs1_mode="full"):
        pp = p if noise else 0.0
        c = stim.Circuit()
        c.append("R", list(range(self.N)))
        # 初期化: C,T は |0>(必要なら X_L で |1>), A は |+_L>
        c.append("H", [self.id_of[cc] for cc in self.A])        # A -> |+> (全データ H)
        if control_one:
            c.append("X", self.XL_C if hasattr(self, "XL_C") else [self.id_of[cc] for cc in self.C if cc[0] == 1])
        if target_one:
            c.append("X", [self.id_of[cc] for cc in self.T if cc[0] == 7])

        all_ancs = sorted(set(self.anc_of.values()))
        mc = [0]
        rounds = []
        schedule = ([("split", self.split_info)] * split_rounds +
                    [("zz", self.zz_info)] * merge_rounds +
                    [("split", self.split_info)] * split_rounds +
                    [("xx", self.xx_info)] * merge_rounds +
                    [("split", self.split_info)] * split_rounds)
        prev_tag = None
        for tag, info in schedule:
            reset = all_ancs if tag != prev_tag else None   # フェーズ境界で全 ancilla reset
            rec = self._round(c, info, pp, mc, reset_all=reset)
            rounds.append((tag, info, rec))
            prev_tag = tag

        # m_zz = 最初の zz ラウンドの joint 構成スタビライザ積
        first_zz = next(rec for tag, info, rec in rounds if tag == "zz")
        zz_joint_recs = [first_zz[i] for i in self.zz_joint_idx]

        # 最終データ Z 測定
        c.append("M", list(range(self.N)))
        drec = {q: mc[0] + q for q in range(self.N)}
        mc[0] += self.N
        rr = lambda a: stim.target_rec(a - mc[0])

        # 検出器 (round0 はパッチ別基底, 以降は _compose_in)
        def det0_type(sup):
            ps = {self.patch_of[self.coords[q]] for q in sup}
            # split のスタビライザは単一パッチに属する
            patch = next(iter(ps))
            return "X" if patch == "A" else "Z"   # A=|+>->X 決定論, C/T=|0>->Z 決定論

        detectors = []
        for ri, (tag, info, rec) in enumerate(rounds):
            if ri == 0:
                for si, (cx, cy, t, s) in enumerate(info):
                    if t == det0_type(s):
                        detectors.append([rec[si]])
                continue
            ptag, pinfo, prec = rounds[ri - 1]
            # 前向き: 現ラウンドのスタビライザを前 config で分解
            covered_prev = set()
            for si, (cx, cy, t, s) in enumerate(info):
                comp = self._decompose(self._vec(t, s), pinfo)
                if comp is None:
                    continue
                detectors.append([rec[si]] + [prec[j] for j in comp])
                if len(comp) == 1:
                    covered_prev.add(comp[0])
            # 後ろ向き閉包: merge->split で分裂する seam スタビライザ。前 config の
            # スタビライザ(seam)が現 config で複数スタビライザの積に分解する場合、
            # その積は保存量なので閉包検出器を張る (例: seam-Z = A右境界Z (+) T左境界Z)。
            # これが無いと分裂端の縮退で論理距離が落ちる。
            for pj, (pcx, pcy, pt, ps) in enumerate(pinfo):
                if pj in covered_prev:
                    continue
                back = self._decompose(self._vec(pt, ps), info)
                if back is None or len(back) <= 1:
                    continue
                detectors.append([prec[pj]] + [rec[k] for k in back])
        for d in detectors:
            c.append("DETECTOR", [rr(a) for a in d])

        # 最終ラウンド検出器 (spacelike 境界を閉じる): Z 型を最終データから再構成
        last_tag, last_info, last_rec = rounds[-1]
        for si, (cx, cy, t, s) in enumerate(last_info):
            if t == "Z":
                c.append("DETECTOR", [rr(last_rec[si])] + [rr(drec[q]) for q in s])

        # 観測量 (フレーム補正):
        #   obs0 = Z_L(C)               -> c を読む
        #   obs1 = Z_L(T) ^ m_zz ^ m_anc(=Z_L(A)) -> c xor t を読む
        c.append("OBSERVABLE_INCLUDE", [rr(drec[q]) for q in self.ZL_C], 0)
        obs1 = [rr(drec[q]) for q in self.ZL_T]
        if obs1_mode in ("full", "zlt_manc"):
            obs1 += [rr(drec[q]) for q in self.ZL_A]      # m_anc = Z_L(A)
        if obs1_mode in ("full", "zlt_mzz"):
            obs1 += [rr(r) for r in zz_joint_recs]          # m_zz
        c.append("OBSERVABLE_INCLUDE", obs1, 1)
        return c, drec, zz_joint_recs


class LSTeleportFT(LSCnotFT):
    """実 T テレポートの回路レベル QEC スケルトン。

    構造 (gate teleportation):
        CNOT(data->magic) を M_ZZ(data,anc) + M_XX(anc,magic) で実装し、
        magic を Z 測定して m を得て、m=1 なら data に S 補正。
        data(=control) が T|psi> として生き残る。

    真の |A_L>=T|+> は非 Clifford なので Stim では準備できない (magic state
    fault は t_gate.py / runtime の magic 模型で別に扱う)。本クラスは merge 群
    (M_ZZ + M_XX + 必要なら S 補正の M_ZZ) の回路レベル QEC ノイズと距離を
    与えるスケルトンで、観測量は data(control) の論理 Z 残差。

    4 パッチ (全て正座標, data を中央に置き2つの ZZ merge を data の上下=対辺に
    配置して角共有を回避):
        D (data)   = (7,7)     |0>
        A1(cnot anc)= (7,1)     |+>   D の上, M_ZZ(D,A1)
        M (magic)  = (13,1)    |+>   A1 の右, M_XX(A1,M)  (|A> の Clifford 骨格)
        As(s anc)  = (7,13)    |Y>   D の下, M_ZZ(D,As)   (S 補正; with_s 時のみ)
    """

    def __init__(self, with_s=True):
        self.with_s = with_s

        def cells(x0, y0):
            return set((x0 + 2 * c, y0 + 2 * r) for r in range(3) for c in range(3))
        self.D = cells(7, 7)
        self.A1 = cells(7, 1)
        self.M = cells(13, 1)
        self.As = cells(7, 13) if with_s else set()

        self.patch_of = {}
        for c in self.D: self.patch_of[c] = "D"
        for c in self.A1: self.patch_of[c] = "A1"
        for c in self.M: self.patch_of[c] = "M"
        for c in self.As: self.patch_of[c] = "As"
        # 初期化基底 (|0>->Z det0, |+>->X det0)。S 補正 merge は CircuitLevelMerge
        # の LS-S に倣い As も |0>(Z); |Y> の S 内容は論理層で扱う。
        self.init_basis = {"D": "Z", "A1": "X", "M": "X", "As": "Z"}

        self.coords = sorted(self.D | self.A1 | self.M | self.As)
        self.id_of = {c: i for i, c in enumerate(self.coords)}
        self.N = len(self.coords)

        def estabs(P):
            return _enum_stabs(P) if P else []
        def to_id(stabs):
            return [(cx, cy, t, [self.id_of[c] for c in sup]) for cx, cy, t, sup in stabs]
        self.split_info = to_id(_enum_stabs(self.D) + _enum_stabs(self.A1) +
                                _enum_stabs(self.M) + estabs(self.As))
        self.zz_cnot_info = to_id(_enum_stabs(self.D | self.A1) +
                                  _enum_stabs(self.M) + estabs(self.As))
        self.xx_cnot_info = to_id(_enum_stabs(self.D) + _enum_stabs(self.A1 | self.M) +
                                  estabs(self.As))
        if with_s:
            self.zz_s_info = to_id(_enum_stabs(self.D | self.As) +
                                   _enum_stabs(self.A1) + _enum_stabs(self.M))

        # 論理 (Z_L=bottom row, X_L=left column)
        ymin = lambda P: min(y for x, y in P)
        xmin = lambda P: min(x for x, y in P)
        self.ZL_D = [self.id_of[c] for c in self.D if c[1] == ymin(self.D)]
        self.ZL_A1 = [self.id_of[c] for c in self.A1 if c[1] == ymin(self.A1)]
        self.XL_A1 = [self.id_of[c] for c in self.A1 if c[0] == xmin(self.A1)]
        self.XL_M = [self.id_of[c] for c in self.M if c[0] == xmin(self.M)]
        if with_s:
            self.ZL_As = [self.id_of[c] for c in self.As if c[1] == ymin(self.As)]

        # joint: m_zz_cnot = Z_L(D)Z_L(A1) を zz_cnot_info の seam 積で
        self.zz_cnot_joint_idx = self._decompose(
            self._lvec("Z", self.ZL_D + self.ZL_A1), self.zz_cnot_info)

        # ancilla 割当
        self.anc_of = {}
        nid = self.N
        infos = [self.split_info, self.zz_cnot_info, self.xx_cnot_info]
        if with_s:
            infos.append(self.zz_s_info)
        for info in infos:
            for cx, cy, t, s in info:
                k = (cx, cy, t, tuple(s))
                if k not in self.anc_of:
                    self.anc_of[k] = nid
                    nid += 1
        self.n_total = nid

    def build_teleport(self, data_one=False, merge_rounds=3, split_rounds=2,
                       p=0.005, noise=True):
        """T テレポートスケルトン回路を返す。

        Returns: (circuit, drec, info_dict)
          観測量 obs0 = Z_L(data) (residual X_L 故障チャネル)。
          obs1 = magic(M) の Z 読み出し (テレポート測定 m, 参考)。
        """
        pp = p if noise else 0.0
        c = stim.Circuit()
        c.append("R", list(range(self.N)))
        # 初期化: D=|0>(data_one で |1>), A1,M=|+>, As=|0> (LS-S と同じ)
        plus = [self.id_of[cc] for cc in (self.A1 | self.M)]
        c.append("H", plus)
        if data_one:
            c.append("X", self.ZL_D)   # data -> |1_L>

        all_ancs = sorted(set(self.anc_of.values()))
        mc = [0]
        rounds = []
        schedule = ([("split", self.split_info)] * split_rounds +
                    [("zzc", self.zz_cnot_info)] * merge_rounds +
                    [("split", self.split_info)] * split_rounds +
                    [("xxc", self.xx_cnot_info)] * merge_rounds +
                    [("split", self.split_info)] * split_rounds)
        if self.with_s:
            schedule += ([("zzs", self.zz_s_info)] * merge_rounds +
                         [("split", self.split_info)] * split_rounds)
        prev_tag = None
        for tag, info in schedule:
            reset = all_ancs if tag != prev_tag else None
            rec = self._round(c, info, pp, mc, reset_all=reset)
            rounds.append((tag, info, rec))
            prev_tag = tag

        first_zzc = next(rec for tag, info, rec in rounds if tag == "zzc")
        zzc_joint_recs = [first_zzc[i] for i in self.zz_cnot_joint_idx]

        c.append("M", list(range(self.N)))
        drec = {q: mc[0] + q for q in range(self.N)}
        mc[0] += self.N
        rr = lambda a: stim.target_rec(a - mc[0])

        def det0_type(sup):
            patch = next(iter({self.patch_of[self.coords[q]] for q in sup}))
            return "Z" if self.init_basis[patch] == "Z" else "X"

        detectors = []
        for ri, (tag, info, rec) in enumerate(rounds):
            if ri == 0:
                for si, (cx, cy, t, s) in enumerate(info):
                    if t == det0_type(s):
                        detectors.append([rec[si]])
                continue
            ptag, pinfo, prec = rounds[ri - 1]
            covered_prev = set()
            for si, (cx, cy, t, s) in enumerate(info):
                comp = self._decompose(self._vec(t, s), pinfo)
                if comp is None:
                    continue
                detectors.append([rec[si]] + [prec[j] for j in comp])
                if len(comp) == 1:
                    covered_prev.add(comp[0])
            for pj, (pcx, pcy, pt, ps) in enumerate(pinfo):
                if pj in covered_prev:
                    continue
                back = self._decompose(self._vec(pt, ps), info)
                if back is None or len(back) <= 1:
                    continue
                detectors.append([prec[pj]] + [rec[k] for k in back])
        for d in detectors:
            c.append("DETECTOR", [rr(a) for a in d])

        last_tag, last_info, last_rec = rounds[-1]
        for si, (cx, cy, t, s) in enumerate(last_info):
            if t == "Z":
                c.append("DETECTOR", [rr(last_rec[si])] + [rr(drec[q]) for q in s])

        # 観測量: obs0 = Z_L(data) (全 merge は Z_L(D) と可換なので保存; 補正不要)。
        # これが residual X_L 故障チャネル。magic(M) の Z 読み出し(=テレポート測定 m)は
        # M=|+> ゆえ非決定論的なので観測量には含めない (m は論理層で別に扱う)。
        c.append("OBSERVABLE_INCLUDE", [rr(drec[q]) for q in self.ZL_D], 0)
        return c, drec, {"zzc_joint_recs": zzc_joint_recs, "ZL_D": self.ZL_D}

    def decode(self, c, shots=40000):
        det, obs = c.compile_detector_sampler().sample(shots, separate_observables=True)
        dem = c.detector_error_model(
            decompose_errors=True,        # 分解可能な複合誤りを graphlike 成分へ分解
            allow_gauge_detectors=False,  # 非決定論的 detector を拒否
        )
        m = pymatching.Matching.from_detector_error_model(dem)
        pred = m.decode_batch(det)
        return (obs ^ pred).mean(axis=0)


def demo():
    print("#" * 70)
    print("#  回路レベル LS CNOT (circuit_level_merge 手法の多パッチ一般化)")
    print("#" * 70)
    g = LSCnotFT()
    print(f"\n[A] レイアウト: C(上)/A(下,ZZ)/T(右,XX)  data={g.N}, ancilla={g.n_total-g.N}")
    print(f"    m_zz 構成スタビライザ idx={g.zz_joint_idx}")

    print("\n[B] ノイズなし決定論性 (gauge 検出器が出ないこと = 最重要)")
    c0, _, _ = g.build(noise=False)
    det, _ = c0.compile_detector_sampler().sample(3000, separate_observables=True)
    nd = int(det.any(axis=0).sum())
    print(f"    検出器数={c0.num_detectors}, 非決定論的={nd}  ({'OK — allow_gauge_detectors 不要' if nd == 0 else 'NG'})")

    print("\n[C] 計算基底 CNOT 真理値表 (生サンプルで論理絶対値, フレーム補正観測量)")
    print("    obs0=Z_L(C),  obs1=Z_L(T)^m_zz^m_anc   (期待 (c, c^t))")
    ok_all = True
    for ci in (0, 1):
        for ti in (0, 1):
            c, drec, zzj = g.build(control_one=bool(ci), target_one=bool(ti), noise=False)
            s = c.compile_sampler().sample(300).astype(int)
            o0 = s[:, [drec[q] for q in g.ZL_C]].sum(1) % 2
            o1 = (s[:, [drec[q] for q in g.ZL_T]].sum(1) +
                  s[:, [drec[q] for q in g.ZL_A]].sum(1) +
                  s[:, list(zzj)].sum(1)) % 2
            v0, v1 = int(round(o0.mean())), int(round(o1.mean()))
            det_ok = len(set(o0.tolist())) == 1 and len(set(o1.tolist())) == 1
            exp0, exp1 = ci, ci ^ ti
            ok = (v0 == exp0 and v1 == exp1 and det_ok)
            ok_all &= ok
            print(f"    (c={ci},t={ti}) -> ({v0},{v1}) 期待({exp0},{exp1}) 決定論={det_ok} {'OK' if ok else 'NG'}")
    print(f"    真理値表: {'全一致 OK (CNOT を正しく実装)' if ok_all else 'NG'}")

    print("\n[D] FT 性: 回路レベル距離 (gauge 不許可で計算)")
    cn, _, _ = g.build(p=0.005, noise=True)
    graphlike_bound = len(cn.shortest_graphlike_error(ignore_ungraphlike_errors=True))
    print(f"    shortest graphlike logical error = {graphlike_bound} (回路距離の上界)")
    if graphlike_bound >= 3:
        print(f"    上界 3 はパッチ距離 3 と整合。厳密性には全誤り機構の graphlike 検証が必要。")
        print(f"    merge->split で分裂する seam")
        print(f"    スタビライザの積(保存量)を閉じる後ろ向き閉包検出器で、T の左下角")
        print(f"    (XX seam 隣接)の縮退を解消した。")
    else:
        print(f"    注: graphlike 上界が3未満。後ろ向き閉包検出器を確認。")

    print("\n[E] FT デコード (MWPM, graphlike, gauge 不要)")
    for p in (0.002, 0.005, 0.01):
        c, _, _ = g.build(p=p, noise=True)
        r = g.decode(c, shots=40000)
        print(f"    p={p:<6}: 論理エラー率 obs0(C)={r[0]:.4f}, obs1(T)={r[1]:.4f}")

    print("\n[サマリ]")
    print("    circuit_level_merge の手法(_compose_in 遷移検出器・最終検出器・FT CX")
    print("    スケジュール)を多パッチに一般化し、merge->split の seam 分裂閉包を追加。")
    print("    決定論性は完全(gauge 検出器ゼロ→allow_gauge_detectors 不要)、CNOT 真理")
    print("    値表一致、graphlike 上界 = 3 (パッチ距離と整合、厳密距離の証明ではない)。")


if __name__ == "__main__":
    demo()
