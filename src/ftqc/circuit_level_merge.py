"""
circuit_level_merge.py — 回路レベル幾何的 lattice surgery (seam merge)
======================================================================
geom_lattice_surgery.py は局所 seam スタビライザを MPP で直接測っていた。本実装は
それを **ancilla ベース syndrome extraction (回路レベル)** に置き換える:
各スタビライザに plaquette 中心の ancilla を割り当て、surface_patch で Stim から
抽出した「hook 誤りを避ける CX スケジュール順」を merged 符号全体へ一様適用する。

  X 型 ancilla(control): CX step 順 = NE, NW, SE, SW
  Z 型 ancilla(target) : CX step 順 = NE, SE, NW, SW   (X の転置)

幾何(データ配置・スタビライザ集合・joint 分解・論理演算子)は GeomMerge を再利用。
検出器則も geom の _compose_in をそのまま流用する(記録が MPP→ancilla 測定に変わる
だけで、span 合成の論理は不変)。split -> merge(d回) -> split のスケジュールで、
joint Z_L(A)Z_L(B) は最初の merge ラウンドの seam スタビライザ積として読む。
"""
import numpy as np
import stim
import pymatching
from ftqc.geom_lattice_surgery import GeomMerge, _dtype

# CX 方向 -> 4 ステップ割当 (surface_patch で Stim から抽出した FT 順)
_XSTEP = {(1, 1): 0, (-1, 1): 1, (1, -1): 2, (-1, -1): 3}    # NE,NW,SE,SW
_ZSTEP = {(1, 1): 0, (1, -1): 1, (-1, 1): 2, (-1, -1): 3}    # NE,SE,NW,SW


class CircuitLevelMerge:
    def __init__(self, kind="ZZ"):
        self.kind = kind
        self.g = GeomMerge(kind)
        self.n = self.g.n
        self.dcoord = {idx: coord for coord, idx in self.g.data.items()}

        all_cells = [(c, r) for r in range(self.g.H) for c in range(self.g.W)]
        self.merged_info = self._stabs_with_center(all_cells)
        self.split_info = (self._stabs_with_center(self.g.A_cells) +
                           self._stabs_with_center(self.g.B_cells))

        # geom の stabs 並びに整列 (検出器則が index を使うため)
        self.split_info = self._align(self.split_info, self.g.split_stabs)
        self.merged_info = self._align(self.merged_info, self.g.merged_stabs)

        # ancilla を (中心, 型, support) で一意割当。
        #   bulk: split/merged で support 一致 -> 同一 ancilla を共有 (連続比較が成立)
        #   seam: A上境界/B下境界/merged seam は support が異なる -> 別 ancilla
        # これにより seam 中心 (例 (4,6)) での ancilla 奪い合いを解消する。
        self.anc_of = {}
        nid = self.n
        for info in (self.merged_info, self.split_info):
            for cx, cy, t, s in info:
                key = (cx, cy, t, tuple(s))
                if key not in self.anc_of:
                    self.anc_of[key] = nid
                    nid += 1
        self.n_total = nid

    def _stabs_with_center(self, cells):
        ds = set((2 * c + 1, 2 * r + 1) for (c, r) in cells)
        xs = [2 * c + 1 for c, r in cells]
        ys = [2 * r + 1 for c, r in cells]
        cmin, cmax, rmin, rmax = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
        out = []
        for cx in range(cmin, cmax + 1, 2):
            for cy in range(rmin, rmax + 1, 2):
                corners = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx + 1, cy + 1)]
                pres = [p for p in corners if p in ds]
                if len(pres) == 4:
                    out.append((cx, cy, _dtype(cx, cy), sorted(self.g.data[p] for p in pres)))
                elif len(pres) == 2:
                    t = _dtype(cx, cy)
                    on_tb = (cy == rmin or cy == rmax)
                    on_lr = (cx == cmin or cx == cmax)
                    if (on_tb and t == "X") or (on_lr and t == "Z"):
                        out.append((cx, cy, t, sorted(self.g.data[p] for p in pres)))
        return out

    def _align(self, info, geom_stabs):
        """info を geom_stabs の (type, support) 並び順に整列。"""
        key2info = {(t, tuple(s)): (cx, cy, t, s) for cx, cy, t, s in info}
        return [key2info[(t, tuple(s))] for t, s in geom_stabs]

    # ---- 回路レベル 1 ラウンド (ancilla syndrome extraction + 回路ノイズ) ----
    def _round(self, c, info, p, mc_ref):
        ancillas = [self.anc_of[(cx, cy, t, tuple(s))] for cx, cy, t, s in info]
        xanc = [self.anc_of[(cx, cy, t, tuple(s))] for cx, cy, t, s in info if t == "X"]
        # before-round data depolarization
        if p > 0:
            c.append("DEPOLARIZE1", list(range(self.n)), p)
        # reset ancillas
        c.append("R", ancillas)
        if p > 0:
            c.append("X_ERROR", ancillas, p)
        if xanc:
            c.append("H", xanc)
            if p > 0:
                c.append("DEPOLARIZE1", xanc, p)
        # 4 CX ステップ
        for step in range(4):
            pairs = []
            for cx, cy, t, sup in info:
                a = self.anc_of[(cx, cy, t, tuple(sup))]
                table = _XSTEP if t == "X" else _ZSTEP
                for d in sup:
                    px, py = self.dcoord[d]
                    dirn = (int(np.sign(px - cx)), int(np.sign(py - cy)))
                    if table[dirn] == step:
                        pairs.append((a, d) if t == "X" else (d, a))  # X:anc->data, Z:data->anc
            flat = []
            for u, v in pairs:
                flat += [u, v]
            if flat:
                c.append("CX", flat)
                if p > 0:
                    c.append("DEPOLARIZE2", flat, p)
        if xanc:
            c.append("H", xanc)
            if p > 0:
                c.append("DEPOLARIZE1", xanc, p)
        # measure ancillas
        if p > 0:
            c.append("X_ERROR", ancillas, p)
        c.append("M", ancillas)
        rec = {si: mc_ref[0] + i for i, si in enumerate(range(len(info)))}
        mc_ref[0] += len(ancillas)
        return rec

    def build(self, aone=False, bone=False, merge_rounds=3, split_rounds=2, p=0.005, noise=True):
        pp = p if noise else 0.0
        # 測定基底は merge 型に合わせる:
        #   ZZ: |0>初期化, flip=X_L, 最終Z測定, 観測Z_L, round0決定論=Z型
        #   XX: |+>初期化, flip=Z_L, 最終X測定, 観測X_L, round0決定論=X型
        zbasis = (self.kind == "ZZ")
        det0_type = "Z" if zbasis else "X"
        flip_kind = "X" if zbasis else "Z"   # |0>->|1> は X_L, |+>->|-> は Z_L
        obs_kind = "Z" if zbasis else "X"

        c = stim.Circuit()
        c.append("R", list(range(self.n)))
        if not zbasis:
            c.append("H", list(range(self.n)))   # |+> 初期化
        if aone:
            c.append(flip_kind, [int(q) for q in np.where(self.g._logical(flip_kind, self.g.A_cells)[:self.n] if flip_kind == "X" else self.g._logical(flip_kind, self.g.A_cells)[self.n:])[0]])
        if bone:
            c.append(flip_kind, [int(q) for q in np.where(self.g._logical(flip_kind, self.g.B_cells)[:self.n] if flip_kind == "X" else self.g._logical(flip_kind, self.g.B_cells)[self.n:])[0]])

        mc = [0]
        rounds = []
        schedule = ([("split", self.split_info)] * split_rounds +
                    [("merge", self.merged_info)] * merge_rounds +
                    [("split", self.split_info)] * split_rounds)
        for tag, info in schedule:
            rec = self._round(c, info, pp, mc)
            rounds.append((tag, info, rec))

        # joint: 最初の merge ラウンドの seam スタビライザ積
        first_merge = next(rec for tag, info, rec in rounds if tag == "merge")
        joint_recs = [first_merge[i] for i in self.g.joint_idx]

        # 最終データ測定 (基底に応じ M / MX)
        c.append("M" if zbasis else "MX", list(range(self.n)))
        drec = {q: mc[0] + q for q in range(self.n)}
        mc[0] += self.n
        rr = lambda a: stim.target_rec(a - mc[0])

        # 検出器: geom と同じ _compose_in 則 (round0 決定論型のみ基底依存)
        detectors = []
        for ri, (tag, info, rec) in enumerate(rounds):
            stabs = [(t, s) for _, _, t, s in info]
            if ri == 0:
                for si, (t, s) in enumerate(stabs):
                    if t == det0_type:
                        detectors.append([rec[si]])
                continue
            ptag, pinfo, prec = rounds[ri - 1]
            prev_stabs = [(t, s) for _, _, t, s in pinfo]
            for si, (t, s) in enumerate(stabs):
                comp = self.g._compose_in(self.g._vec(t, s), prev_stabs)
                if comp is None:
                    continue
                detectors.append([rec[si]] + [prec[j] for j in comp])
        for d in detectors:
            c.append("DETECTOR", [rr(a) for a in d])

        # 最終ラウンド検出器: 最終データ測定から測定基底型のスタビライザを再構成し、
        # 直前 syndrome ラウンドの対応 ancilla 記録と比較する(=spacelike 境界を閉じる)。
        # これが無いと読み出し直前の単一データ誤りが捕捉されず距離が1に落ちる。
        last_tag, last_info, last_rec = rounds[-1]
        for si, (cx, cy, t, s) in enumerate(last_info):
            if t == obs_kind:
                c.append("DETECTOR", [rr(last_rec[si])] + [rr(drec[d]) for d in s])

        # 観測量: 各論理 (基底に応じ Z_L / X_L のデータ測定パリティ)
        la = self.g._logical(obs_kind, self.g.A_cells)
        lb = self.g._logical(obs_kind, self.g.B_cells)
        la_sup = list(np.where(la[:self.n] if obs_kind == "X" else la[self.n:])[0])
        lb_sup = list(np.where(lb[:self.n] if obs_kind == "X" else lb[self.n:])[0])
        c.append("OBSERVABLE_INCLUDE", [rr(drec[int(q)]) for q in la_sup], 0)
        c.append("OBSERVABLE_INCLUDE", [rr(drec[int(q)]) for q in lb_sup], 1)
        return c, joint_recs, drec

    def decode(self, c, shots=40000):
        det, obs = c.compile_detector_sampler().sample(shots, separate_observables=True)
        try:
            dem = c.detector_error_model(decompose_errors=True, allow_gauge_detectors=False)
            m = pymatching.Matching.from_detector_error_model(dem)
            pred = m.decode_batch(det)
            which = "MWPM"
        except ValueError:
            from stimbposd import BPOSD
            pred = BPOSD(c.detector_error_model(decompose_errors=False, allow_gauge_detectors=False), max_bp_iters=40).decode_batch(det)
            which = "BP+OSD"
        return (obs ^ pred).mean(axis=0), which


def demo():
    print("#" * 70)
    print("#  回路レベル幾何的 lattice surgery (seam merge, ancilla syndrome extraction)")
    print("#" * 70)

    clm = CircuitLevelMerge("ZZ")
    print(f"\n[A] 構成 (M_ZZ 縦積み 3x6, 回路レベル)")
    print(f"    data={clm.n}, ancilla={clm.n_total - clm.n}, "
          f"split stabs={len(clm.split_info)}, merged stabs={len(clm.merged_info)}")
    print(f"    joint Z_L(A)Z_L(B) = seam スタビライザ {len(clm.g.joint_idx)} 個の積 idx={clm.g.joint_idx}")

    print(f"\n[B] joint 測定の正しさ (ノイズなし, 計算基底)")
    for aone in (False, True):
        for bone in (False, True):
            c, jr, _ = clm.build(aone, bone, noise=False)
            samp = c.compile_sampler().sample(200)
            jout = int(np.bitwise_xor.reduce(samp[0, jr].astype(int)))
            exp = int(aone) ^ int(bone)
            print(f"    A={'|1>' if aone else '|0>'}, B={'|1>' if bone else '|0>'}: "
                  f"joint={jout} 期待={exp} {'OK' if jout == exp else 'NG'}")

    print(f"\n[C] 検出器の決定論性 (ノイズなしで全0)")
    c, _, _ = clm.build(aone=True, noise=False)
    det, _ = c.compile_detector_sampler().sample(300, separate_observables=True)
    print(f"    検出器数={c.num_detectors}, 全0={not det.any()}")

    print(f"\n[D] FT 性: 回路レベル距離 (shortest_graphlike_error)")
    cdist, _, _ = clm.build(aone=True, noise=True, p=0.005)
    graphlike_bound = len(cdist.shortest_graphlike_error(ignore_ungraphlike_errors=True))
    print(f"    DEM 上の最短 graphlike 論理誤り = {graphlike_bound} "
          f"(回路距離の上界; 3 なら d=3 と整合)")

    print(f"\n[E] FT デコード (ancilla 局所スタビライザ -> MWPM, graphlike, p^2 スケール)")
    for p in (0.002, 0.005, 0.01):
        c, _, _ = clm.build(aone=True, noise=True, p=p)
        rates, which = clm.decode(c, shots=40000)
        print(f"    p={p:<6}: 論理エラー率 A={rates[0]:.4f}, B={rates[1]:.4f}  [{which}]")

    print(f"\n[F] M_XX (横並び 6x3) も同型で回路レベル化")
    clx = CircuitLevelMerge("XX")
    cx, jrx, _ = clx.build(noise=False)
    detx, _ = cx.compile_detector_sampler().sample(200, separate_observables=True)
    print(f"    X_L(A)X_L(B) = seam スタビライザ {len(clx.g.joint_idx)} 個の積, "
          f"検出器全0={not detx.any()}")
    cxn, _, _ = clx.build(noise=True, p=0.005)
    rx, wx = clx.decode(cxn, shots=40000)
    print(f"    p=0.005: 論理エラー率 A={rx[0]:.4f}, B={rx[1]:.4f}  [{wx}]")

    print("\n[サマリ]")
    print("    geom の局所 seam スタビライザを ancilla syndrome extraction に置換し、")
    print("    surface_patch 由来の FT CX スケジュールを merged 符号全体へ適用。")
    print("    joint 測定・検出器決定論性・graphlike MWPM デコードを回路レベルで検証。")
    print("    MPP 抽象から脱却した、現実的な回路レベル lattice surgery を実現した。")


if __name__ == "__main__":
    demo()
