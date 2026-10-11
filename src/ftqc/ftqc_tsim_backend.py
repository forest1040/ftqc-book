#!/usr/bin/env python3
"""
ftqc_tsim_backend — `nonclifford_mode="tsim"` バックエンドのプロトタイプ.

HybridFTQCRuntime の run() は、ショットごとに `make_backend(n, backend_name)` で
論理状態バックエンドを作り、`apply_gate` でゲートと QEC 残差(X/Z)を適用し、
最後に `sample_z(rng)` で Z 基底のビット列を引く——という単純な継ぎ目を持ちます。
状態ベクトル(qulacs/numpy)はここで非クリフォード T を「直接適用」しますが、Stim は
クリフォード限定なので、これまでは statevector か QPD で T を処理していました。

本モジュールは、この継ぎ目に **tsim**(ZX スタビライザランク分解, 非クリフォード対応)を
差し込む `TsimStateBackend` を提供します。apply_gate は論理回路へゲートを蓄積し、
sample_z でその Clifford+T 回路を tsim にネイティブ・サンプリングさせます。これにより
T を「単一の直接サンプリング」で扱え(QPD の γ=√2 オーバーヘッドも、状態ベクトルの
指数メモリも不要、コストは T 数に対する ~2^{0.4T} のスタビライザランク)、QEC ハンドオフ
(Stim シェル)・残差注入・magic fault 模型は run() のものを *一切変更せず* 再利用します。

これは「nonclifford_mode='tsim'」の挙動を非侵襲に実証するプロトタイプです。本体に
ネイティブ実装するなら、hybrid_ftqc_runtime.make_backend に 'tsim' 分岐を 1 つ足し、
nonclifford_mode の別名として 'tsim'(=statevector 経路 + backend 'tsim') を通すだけです。

例:
  python ftqc_tsim_backend.py --gates "H0 T0 H0" --n-logical 1 --src-path ./src
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# tsim 命令名へのマッピング（runtime の正準名 -> Stim/tsim テキスト）
_TSIM_NAME = {"SDG": "S_DAG", "T_DAG": "T_DAG", "CNOT": "CNOT"}

# 同一ゲート列のサンプルを束ねて使い回すキャッシュ（特にノイズなしでは全軌跡が同一回路）
_BATCH = 256
_SAMPLE_CACHE: Dict[Tuple[Tuple, int], list] = {}


def _clear_cache() -> None:
    _SAMPLE_CACHE.clear()


def validate_distance_d3(distance: int) -> None:
    """Stage 03 explicit distance boundary check."""
    if distance != 3:
        raise ValueError(f"Stage 03 only supports distance d=3 fixed (got d={distance}).")


class TsimStateBackend:
    """tsim をランタイムの論理状態バックエンドとして使うアダプタ。

    状態ベクトルは保持せず、適用されたゲート列を記録し、sample_z で
    Clifford+T 回路を tsim にサンプリングさせる。
    """

    name = "tsim"

    def __init__(self, n: int, distance: int = 3):
        validate_distance_d3(distance)
        self.n = int(n)
        self.d = int(distance)
        self.gates: List[Tuple[str, int, Optional[int]]] = []

    def copy(self) -> "TsimStateBackend":
        other = TsimStateBackend(self.n)
        other.gates = list(self.gates)
        return other

    def apply_gate(self, name: str, a: int, b: Optional[int] = None) -> None:
        from ftqc.hybrid_ftqc_runtime import normalize_gate_name
        nm = normalize_gate_name(name)
        if nm == "I":
            return
        tnm = _TSIM_NAME.get(nm, nm)
        self.gates.append((tnm, int(a), None if b is None else int(b)))

    def _circuit_text(self) -> str:
        lines = []
        for nm, a, b in self.gates:
            lines.append(f"{nm} {a}" if b is None else f"{nm} {a} {b}")
        lines.append("M " + " ".join(str(q) for q in range(self.n)))
        return "\n".join(lines)

    def sample_z(self, rng: np.random.Generator) -> Tuple[int, ...]:
        key = (tuple(self.gates), self.n)
        entry = _SAMPLE_CACHE.get(key)
        if entry is None or entry[1] >= len(entry[0]):
            import tsim
            tc = tsim.Circuit(self._circuit_text())
            seed = int(rng.integers(0, 2 ** 31 - 1))
            try:
                samp = np.asarray(tc.compile_sampler().sample(shots=_BATCH, seed=seed))
            except TypeError:
                samp = np.asarray(tc.compile_sampler().sample(shots=_BATCH))
            entry = [samp, 0]
            _SAMPLE_CACHE[key] = entry
        samp, cur = entry
        row = samp[cur]
        entry[1] = cur + 1
        return tuple(int(x) for x in row[:self.n])

    def vector(self) -> Optional[np.ndarray]:
        # tsim はサンプラなので明示的な状態ベクトルは返さない。
        return None


def run_tsim(runtime: Any, gates: Any, shots: int = 200, seed: Optional[int] = None):
    """既存の runtime.run() を、論理バックンドだけ tsim に差し替えて実行する。

    QEC 残差サンプリング・残差注入・magic fault 模型は runtime のものを再利用。
    nonclifford_mode は内部的に 'statevector'(=T を直接適用)にし、backend を 'tsim' に。
    """
    import ftqc.hybrid_ftqc_runtime as H
    orig_make = H.make_backend

    def shim(n: int, backend: str = "auto"):
        if str(backend).lower() == "tsim":
            return TsimStateBackend(n)
        return orig_make(n, backend)

    saved = (runtime.backend_name, runtime.nonclifford_mode)
    runtime.backend_name = "tsim"
    runtime.nonclifford_mode = "statevector"
    H.make_backend = shim
    _clear_cache()
    try:
        res = runtime.run(gates, shots=shots, seed=seed)
    finally:
        H.make_backend = orig_make
        runtime.backend_name, runtime.nonclifford_mode = saved
        _clear_cache()
    return res


def tsim_logical_distribution(gates: List[Tuple[str, int, Optional[int]]],
                              n_logical: int, shots: int = 20000,
                              seed: int = 0) -> Dict[Tuple[int, ...], float]:
    """QEC を介さない、tsim による論理出力分布（検証用の理想相当）。"""
    import tsim
    b = TsimStateBackend(n_logical)
    for nm, a, bb in gates:
        b.apply_gate(nm, a, bb)
    tc = tsim.Circuit(b._circuit_text())
    try:
        samp = np.asarray(tc.compile_sampler().sample(shots=shots, seed=seed))
    except TypeError:
        samp = np.asarray(tc.compile_sampler().sample(shots=shots))
    c = Counter(tuple(int(x) for x in row[:n_logical]) for row in samp)
    return {k: v / shots for k, v in c.items()}


def _tvd(p: Dict, q: Dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _parse_gates(text: str):
    from ftqc.viz import t_branch_visualizer as tb
    return list(tb.parse_gate_text(text))


# --------------------------------------------------------------------------- #
# (b) 高速バッチ版: 残差込みの per-shot 回路を記録し、同一回路をまとめてサンプル
# --------------------------------------------------------------------------- #

# 記録用モジュール変数（run_tsim_fast の中だけで使う）
_RECORDED: List[Tuple] = []


class RecordingBackend:
    """ゲート＋残差(X/Z)＋magic 補正を *記録* するだけのバックエンド。

    runtime.run() の per-shot リプレイをそのまま使い（QEC ハンドオフ・残差注入・
    magic fault は無改変）、各ショットの「論理回路（残差を正しい位置に interleave 済）」
    を集める。tsim は一切呼ばないので記録は安価。サンプリングは後で同一回路ごとに
    まとめて 1 回だけ tsim に投げる。
    """

    name = "tsim_record"

    def __init__(self, n: int):
        self.n = int(n)
        self.gates: List[Tuple[str, int, Optional[int]]] = []

    def copy(self) -> "RecordingBackend":
        other = RecordingBackend(self.n)
        other.gates = list(self.gates)
        return other

    def apply_gate(self, name: str, a: int, b: Optional[int] = None) -> None:
        from ftqc.hybrid_ftqc_runtime import normalize_gate_name
        nm = normalize_gate_name(name)
        if nm == "I":
            return
        self.gates.append((_TSIM_NAME.get(nm, nm), int(a), None if b is None else int(b)))

    def vector(self) -> Optional[np.ndarray]:
        return None

    def sample_z(self, rng: np.random.Generator) -> Tuple[int, ...]:
        _RECORDED.append(tuple(self.gates))   # このショットの完成回路を記録
        return tuple(0 for _ in range(self.n))  # ダミー（実測値は後で tsim で）


def _circuit_text_from_seq(seq: Tuple, n: int) -> str:
    lines = [(f"{nm} {a}" if b is None else f"{nm} {a} {b}") for nm, a, b in seq]
    lines.append("M " + " ".join(str(q) for q in range(n)))
    return "\n".join(lines)


def run_tsim_fast(runtime: Any, gates: Any, shots: int = 2000,
                  seed: Optional[int] = None):
    """高速版 nonclifford='tsim' 実行。

    手順: (1) runtime.run() を RecordingBackend で回し、各ショットの残差込み論理回路を
    収集（tsim は呼ばない=安価, QEC ハンドオフは無改変）。(2) 同一回路ごとにまとめ、
    各ユニーク回路を 1 回の tsim バッチサンプリングで処理。低 p ほどユニーク数が少なく
    (残差ゼロ軌跡が支配的)、tsim 呼び出し回数が激減する。

    返り値: (measurement_counts: dict, n_unique_circuits: int, shots: int)
    """
    import ftqc.hybrid_ftqc_runtime as H
    import tsim
    from collections import Counter

    orig_make = H.make_backend

    def shim(n: int, backend: str = "auto"):
        if str(backend).lower() == "tsim_record":
            return RecordingBackend(n)
        return orig_make(n, backend)

    saved = (runtime.backend_name, runtime.nonclifford_mode)
    runtime.backend_name = "tsim_record"
    runtime.nonclifford_mode = "statevector"
    _RECORDED.clear()
    H.make_backend = shim
    try:
        runtime.run(gates, shots=shots, seed=seed)   # counts は捨て、_RECORDED を埋める
    finally:
        H.make_backend = orig_make
        runtime.backend_name, runtime.nonclifford_mode = saved

    seqs = list(_RECORDED)
    _RECORDED.clear()
    n = runtime.n
    groups = Counter(seqs)
    rng = np.random.default_rng(0 if seed is None else seed + 1)
    counts: Dict[Tuple[int, ...], int] = {}
    for seq, cnt in groups.items():
        tc = tsim.Circuit(_circuit_text_from_seq(seq, n))
        sv = int(rng.integers(0, 2 ** 31 - 1))
        try:
            samp = np.asarray(tc.compile_sampler().sample(shots=cnt, seed=sv))
        except TypeError:
            samp = np.asarray(tc.compile_sampler().sample(shots=cnt))
        for row in samp:
            key = tuple(int(x) for x in row[:n])
            counts[key] = counts.get(key, 0) + 1
    return counts, len(groups), shots


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="nonclifford_mode='tsim' バックエンドのプロトタイプ検証")
    ap.add_argument("--gates", default="H0 T0 H0")
    ap.add_argument("--n-logical", type=int, default=None)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--p-phys", type=float, default=0.0, help="0 なら理想(論理動作)検証, >0 で QEC と合成")
    ap.add_argument("--src-path", default=None)
    ap.add_argument("--vis-path", default=None, help="t_branch_visualizer の場所(ゲート解析に使用)")
    args = ap.parse_args(argv)

    sys.path.insert(0, args.vis_path or os.path.dirname(os.path.abspath(__file__)))
    if args.src_path:
        sys.path.insert(0, str(Path(args.src_path).resolve()))

    gates = _parse_gates(args.gates)
    n = args.n_logical or (max((max(a, b if b is not None else 0) for _n, a, b in gates), default=0) + 1)

    from ftqc.viz import t_branch_visualizer as tb
    from ftqc.hybrid_ftqc_runtime import HybridFTQCRuntime

    ideal = tb.z_distribution_from_gates(gates, n)
    tsim_log = tsim_logical_distribution(gates, n, shots=20000)
    print("gates=%s  n=%d  p_phys=%g" % (args.gates, n, args.p_phys))
    print("理想(statevector)      :", {k: round(v, 4) for k, v in sorted(ideal.items())})
    print("tsim 論理(QECなし)      :", {k: round(v, 4) for k, v in sorted(tsim_log.items())},
          " TVD=%.4f" % _tvd(ideal, tsim_log))

    rt = HybridFTQCRuntime(n, p_phys=args.p_phys)
    res = run_tsim(rt, gates, shots=args.shots)
    dist = res.logical_distribution() if hasattr(res, "logical_distribution") else \
        {k: v / sum(res.measurement_counts.values()) for k, v in res.measurement_counts.items()}
    print("nonclifford='tsim' run :", {k: round(v, 4) for k, v in sorted(dist.items())},
          " backend=%s shots=%d TVD(vs理想)=%.4f" % (res.backend, args.shots, _tvd(ideal, dist)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
