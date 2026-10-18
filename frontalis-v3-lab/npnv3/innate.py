"""生得の層（固定・全人格共通）: 欲求、効果表 M、効用。

ここの値は学習で変わらない。人格の差は経験史からだけ生じる（SPEC 0）。
設計者が与える最上位の軸であり、「欲求そのものが経験から生まれる」ことは主張しない（docs/04 第8節 1）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- 欲求 ---------------------------------------------------------------
NEEDS: Tuple[str, ...] = ("R", "C", "A", "S", "Sa")
NEED_LABELS: Dict[str, str] = {
    "R": "関係性", "C": "有能感", "A": "自律", "S": "刺激", "Sa": "安全",
}
N_NEEDS = len(NEEDS)

H0 = 1.0          # 欲求の状態の初期値・基準値
H_MAX = 6.0       # 状態の上限
LAMBDA_H = 0.1    # 基準へ戻る速さ
S_GROW = 3.0      # 成長欲求の飽和の尺度 s
H_STAR = 1.0      # 安全の必要水準 h*

# --- 結果の型と効果表 M -------------------------------------------------
# 各行は、型 1 単位（magnitude = 1）あたりの欲求の変化 Δh（R, C, A, S, Sa の順）。
# 基本の大きさは 0.30（docs/04 玩具の gM = 0.3 と同じ）。
# 安全の効用は不足側だけの二乗罰で、成長欲求より傾きが小さいので、安全の行は大きめ（0.60）にしてある。
_M_TABLE: Dict[str, Tuple[float, float, float, float, float]] = {
    #                    R      C      A      S      Sa
    "connection":      (+0.30,  0.00,  0.00,  0.00,  0.00),
    "rejection":       (-0.30, -0.05,  0.00,  0.00, -0.10),
    "support":         (+0.15, +0.15,  0.00,  0.00, +0.15),
    "betrayal":        (-0.30,  0.00,  0.00,  0.00, -0.30),
    "harm":            (-0.05, -0.05,  0.00,  0.00, -0.60),
    # v3.1: praise は生得の効果を持たない。社会的承認の価値は、随伴関係から条件づけで形成される（SPEC-v3.1 1.2, U2）
    "praise":          ( 0.00,  0.00,  0.00,  0.00,  0.00),
    "criticism":       (-0.05, -0.20,  0.00,  0.00, -0.05),
    "credit_taken":    ( 0.00,  0.00, -0.30,  0.00,  0.00),
    "credit_given":    (+0.05, +0.10, +0.20,  0.00,  0.00),
    # v3.1: 手柄を共有された（曖昧な出来事「手柄を取られた／共有された」の候補、SPEC-v3.1 1.4）
    "credit_shared":   (+0.10, +0.10, +0.10,  0.00,  0.00),
    "taught":          ( 0.00,  0.00,  0.00, +0.30,  0.00),
    "achievement":     ( 0.00, +0.30,  0.00,  0.00,  0.00),
    "failure":         ( 0.00, -0.30,  0.00,  0.00, -0.10),
    "autonomy_gain":   ( 0.00,  0.00, +0.30,  0.00,  0.00),
    "autonomy_loss":   ( 0.00,  0.00, -0.30,  0.00,  0.00),
    "novelty":         ( 0.00,  0.00,  0.00, +0.30,  0.00),
    "monotony":        ( 0.00,  0.00,  0.00, -0.30,  0.00),
    "safety_threat":   ( 0.00,  0.00,  0.00,  0.00, -0.60),
    "comfort":         (+0.05,  0.00,  0.00,  0.00, +0.30),
}

OUTCOME_TYPES: Tuple[str, ...] = tuple(_M_TABLE.keys())
TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(OUTCOME_TYPES)}
N_TYPES = len(OUTCOME_TYPES)

M = np.array([_M_TABLE[t] for t in OUTCOME_TYPES], dtype=np.float64)
M.setflags(write=False)


def m_table_markdown() -> str:
    """README に載せる効果表 M（コードから生成して食い違いを防ぐ）。"""
    head = "| 結果の型 | " + " | ".join(f"{n}（{NEED_LABELS[n]}）" for n in NEEDS) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in NEEDS) + " |"
    rows = []
    for t in OUTCOME_TYPES:
        cells = [f"{v:+.2f}" if v != 0 else "0" for v in _M_TABLE[t]]
        rows.append(f"| `{t}` | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


# --- 効用 ---------------------------------------------------------------
def utility(h: np.ndarray) -> np.ndarray:
    """欲求ごとの効用 u_k(h_k)。成長欲求は s(1 - exp(-h/s))、安全は -1/2 max(h* - h, 0)^2。"""
    h = np.asarray(h, dtype=np.float64)
    out = np.empty_like(h)
    out[..., :4] = S_GROW * (1.0 - np.exp(-h[..., :4] / S_GROW))
    out[..., 4] = -0.5 * np.maximum(H_STAR - h[..., 4], 0.0) ** 2
    return out


def total_utility(h: np.ndarray) -> np.ndarray:
    return utility(h).sum(axis=-1)


def clip_h(h: np.ndarray) -> np.ndarray:
    return np.clip(h, 0.0, H_MAX)


def delta_u(h: np.ndarray, dh: np.ndarray) -> np.ndarray:
    """事象の効用変化（ベクトル） Δu_k = u_k(h_k + Δh_k) - u_k(h_k)。h の上限・下限をまたぐときは丸めた後の値で測る。"""
    return utility(clip_h(h + dh)) - utility(h)


def next_h(h: np.ndarray, dh: np.ndarray) -> np.ndarray:
    """h ← clip(h + Δh, 0, 6)、続いて h ← h + λ_h (h0 - h)。"""
    h2 = clip_h(h + dh)
    return h2 + LAMBDA_H * (H0 - h2)


def outcome_rows(types: Sequence[str], magnitudes: Sequence[float], table: Optional[np.ndarray] = None) -> np.ndarray:
    """各結果の欲求変化 magnitude_o * M[type_o]。形は (n_outcomes, 5)。table は効果表（既定は固定の M）。"""
    if len(types) == 0:
        return np.zeros((0, N_NEEDS))
    tab = M if table is None else table
    idx = np.array([TYPE_INDEX[t] for t in types])
    mag = np.asarray(magnitudes, dtype=np.float64)
    return mag[:, None] * tab[idx]


def mixed_rows(weights: np.ndarray, magnitudes: Sequence[float], table: Optional[np.ndarray] = None) -> np.ndarray:
    """曖昧な結果を含む欲求変化。weights は (n_outcomes, 型数) で、行ごとに解釈の重み（明確な結果は one-hot）。
    row_o = magnitude_o * Σ_I w_{o,I} M[I]。"""
    tab = M if table is None else table
    if len(weights) == 0:
        return np.zeros((0, N_NEEDS))
    return np.asarray(magnitudes, dtype=np.float64)[:, None] * (np.asarray(weights) @ tab)


def perturb_table(table: np.ndarray, rng: np.random.Generator, rel: float = 0.3) -> np.ndarray:
    """効果表の各要素を ±rel の一様乱数で揺らす（0 の要素は 0 のまま）。U3（M への頑健性）用。"""
    return table * (1.0 + rng.uniform(-rel, rel, size=table.shape))


def contributions(h: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """寄与の分解（SPEC 3）。

    戻り値 (dh, du, contrib):
      dh      : Δh = Σ_o rows_o                      （5,）
      du      : Δu_k                                  （5,）
      contrib : 結果 o の欲求 k への寄与 Δu_k * rows[o,k] / Δh_k   （n_o, 5）。Δh_k = 0 なら 0。
    分離可能な効用なので、寄与の合計は Δu_k に等しい（Δh_k ≠ 0 の欲求について）。
    """
    dh = rows.sum(axis=0) if len(rows) else np.zeros(N_NEEDS)
    du = delta_u(h, dh)
    contrib = np.zeros_like(rows)
    nz = np.abs(dh) > 1e-12
    if len(rows) and nz.any():
        ratio = np.zeros(N_NEEDS)
        ratio[nz] = du[nz] / dh[nz]
        contrib = rows * ratio[None, :]
    return dh, du, contrib


def vec_to_dict(v: np.ndarray) -> Dict[str, float]:
    return {n: float(x) for n, x in zip(NEEDS, v)}
