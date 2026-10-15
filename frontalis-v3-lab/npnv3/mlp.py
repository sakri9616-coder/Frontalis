"""numpy だけの小さな MLP（tanh 隠れ層 1 層、ロジット出力、二値交差エントロピー、SGD）。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # 数値的に安定なシグモイド
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MLP:
    """入力 → 隠れ層（tanh）→ ロジット。学習は SGD（Adam は使わない）。"""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 rng: Optional[np.random.Generator] = None, out_bias: Optional[np.ndarray] = None):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        if rng is None:
            # 復元用（重みは set_arrays で上書きする）
            self.W1 = np.zeros((n_in, n_hidden)); self.b1 = np.zeros(n_hidden)
            self.W2 = np.zeros((n_hidden, n_out)); self.b2 = np.zeros(n_out)
            return
        l1 = np.sqrt(6.0 / (n_in + n_hidden))
        l2 = np.sqrt(6.0 / (n_hidden + n_out))
        self.W1 = rng.uniform(-l1, l1, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.uniform(-l2, l2, size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out) if out_bias is None else np.array(out_bias, dtype=np.float64)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        H = np.tanh(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2, H

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return sigmoid(logits)

    def sgd_step(self, X: np.ndarray, Y: np.ndarray, lr: float, mask: Optional[np.ndarray] = None) -> float:
        """1 ミニバッチの SGD。損失 = 出力型について足し、バッチで平均した二値交差エントロピー。
        mask があれば、観測されたセル（mask = 1）だけを損失に入れる（曖昧な結果の型は観測なし）。"""
        X = np.atleast_2d(X)
        B = X.shape[0]
        logits, H = self.forward(X)
        P = sigmoid(logits)
        eps = 1e-12
        m = 1.0 if mask is None else mask
        loss = float(-(m * (Y * np.log(P + eps) + (1.0 - Y) * np.log(1.0 - P + eps))).sum() / B)
        dZ = (P - Y) * m / B                   # (B, n_out)
        dW2 = H.T @ dZ
        db2 = dZ.sum(axis=0)
        dH = dZ @ self.W2.T
        dA = dH * (1.0 - H ** 2)
        dW1 = X.T @ dA
        db1 = dA.sum(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return loss

    # --- 保存・復元 ---
    def get_arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {f"{prefix}W1": self.W1, f"{prefix}b1": self.b1, f"{prefix}W2": self.W2, f"{prefix}b2": self.b2}

    def set_arrays(self, arrays: Dict[str, np.ndarray], prefix: str) -> None:
        self.W1 = np.array(arrays[f"{prefix}W1"], dtype=np.float64)
        self.b1 = np.array(arrays[f"{prefix}b1"], dtype=np.float64)
        self.W2 = np.array(arrays[f"{prefix}W2"], dtype=np.float64)
        self.b2 = np.array(arrays[f"{prefix}b2"], dtype=np.float64)


class LogReg:
    """線形（ロジスティック回帰）の基線。MLP と同じ入出力・同じ SGD（隠れ層なし）。SPEC-v3.1 1.5 の head_kind = "linear"。"""

    def __init__(self, n_in: int, n_out: int, out_bias: Optional[np.ndarray] = None):
        self.n_in, self.n_out = n_in, n_out
        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros(n_out) if out_bias is None else np.array(out_bias, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.atleast_2d(X) @ self.W + self.b)

    def sgd_step(self, X: np.ndarray, Y: np.ndarray, lr: float, mask: Optional[np.ndarray] = None) -> float:
        X = np.atleast_2d(X)
        B = X.shape[0]
        P = self.predict_proba(X)
        eps = 1e-12
        m = 1.0 if mask is None else mask
        loss = float(-(m * (Y * np.log(P + eps) + (1.0 - Y) * np.log(1.0 - P + eps))).sum() / B)
        dZ = (P - Y) * m / B
        self.W -= lr * (X.T @ dZ)
        self.b -= lr * dZ.sum(axis=0)
        return loss

    def get_arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {f"{prefix}W": self.W, f"{prefix}b": self.b}

    def set_arrays(self, arrays: Dict[str, np.ndarray], prefix: str) -> None:
        self.W = np.array(arrays[f"{prefix}W"], dtype=np.float64)
        self.b = np.array(arrays[f"{prefix}b"], dtype=np.float64)
