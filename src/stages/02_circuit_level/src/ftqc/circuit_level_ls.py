"""circuit_level_ls.py — DEPRECATED / 退役

このモジュールは回路レベル lattice surgery (LS CNOT / LS S) の初期実装でしたが、
検出器則が split<->merge 遷移の _compose_in (GF2 span) 合成を取りこぼすため、
ノイズなしでも非決定論的検出器が残り、決定論性が崩れていました。当時はそれを
detector_error_model(..., allow_gauge_detectors=True) で隠して動かしていました
が、これは距離計算を壊し論理エラー率を不正に押し上げる「隠蔽」であり、修正では
ありません (allow_gauge_detectors は非決定論検出器を情報なしとして捨てるため、
seam 角で距離が崩壊する)。

このファイルは現在どこからも import されておらず、検証済みの後継に置き換え済みです:

  * LS CNOT / T テレポート  -> ls_cnot_ft.py
        - LSCnotFT.build()        : M_ZZ + M_XX, gauge 検出器 0,
                                    graphlike 上界 = 3 (パッチ距離 3 と整合),
                                    CNOT 真理値表一致。
        - LSTeleportFT.build_teleport() : 実 T テレポート (M_ZZ + M_XX + S 補正の M_ZZ),
                                    gauge 0, 距離 3。
        いずれも allow_gauge_detectors を一切使わない。決定論崩れを起こさない鍵は
        merge->split で分裂する seam スタビライザの「積の保存量」を閉じる後ろ向き
        閉包検出器 (ls_cnot_ft.build 内)。

  * LS S / 一般の seam merge -> circuit_level_merge.py
        - CircuitLevelMerge("ZZ"/"XX") : gauge 0, 距離 3。

  * シェル API                -> ls_handoff_shells.py
        - build_ls_cnot_shell / build_ls_s_shell / build_ls_t_injection_shell
        - decode_shell は allow_gauge_detectors を使わない (非決定論検出器が出たら
          例外を素通しさせる正しさのガード)。

したがって circuit_level_ls の build_ls_cnot_circuit / build_ls_s_circuit は使わず、
上記後継を使うこと。互換のため名前だけ残し、呼ぶと後継への移行を促す。
"""

from __future__ import annotations

_REPLACEMENTS = (
    "build_ls_cnot_circuit -> ls_cnot_ft.LSCnotFT().build()  (gauge0, 距離3)\n"
    "build_ls_s_circuit    -> circuit_level_merge.CircuitLevelMerge('ZZ').build()  (gauge0, 距離3)\n"
    "T テレポート          -> ls_cnot_ft.LSTeleportFT().build_teleport()  (gauge0, 距離3)\n"
    "シェル API            -> ls_handoff_shells.build_ls_{cnot,s,t_injection}_shell"
)


def _deprecated(name: str):
    raise NotImplementedError(
        f"circuit_level_ls.{name} は退役しました。この実装は split<->merge 遷移の "
        f"検出器を取りこぼし、allow_gauge_detectors=True で非決定論を隠していました "
        f"(距離崩壊)。検証済みの後継を使ってください:\n{_REPLACEMENTS}"
    )


def build_ls_cnot_circuit(*args, **kwargs):
    _deprecated("build_ls_cnot_circuit")


def build_ls_s_circuit(*args, **kwargs):
    _deprecated("build_ls_s_circuit")


def demo():
    print(__doc__)


if __name__ == "__main__":
    demo()
