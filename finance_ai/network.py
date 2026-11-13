"""Minimal NumPy neural network with explicit backpropagation and AdamW."""
from __future__ import annotations

import numpy as np

LAYERS = (485, 128, 64, 32, 1)


class MLP:
    def __init__(self, sizes=LAYERS, seed=42):
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, b)) for b in sizes[1:]]
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.step = 0

    def forward(self, x, cache=False):
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1] @ w + b, 0))
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        # Clipping keeps exp finite; logits are retained for stable BCE.
        p = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
        if cache:
            return z, p, activations
        return p.reshape(-1)

    def gradients(self, x, y):
        z, p, acts = self.forward(x, cache=True)
        y = y.reshape(-1, 1)
        # Stable BCE-with-logits: logaddexp(0,z)-y*z.
        loss = np.mean(np.logaddexp(0, z) - y * z)
        delta = (p - y) / len(x)
        gw, gb = [None] * len(self.weights), [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            gw[i] = acts[i].T @ delta
            gb[i] = np.sum(delta, axis=0, keepdims=True)
            if i:
                delta = (delta @ self.weights[i].T) * (acts[i] > 0)
        return float(loss), gw, gb

    def train_batch(self, x, y, lr=0.001, weight_decay=0.0001):
        loss, gw, gb = self.gradients(x, y)
        self.step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * gw[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * gw[i] ** 2
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            mw_hat = self.mw[i] / (1 - b1 ** self.step)
            vw_hat = self.vw[i] / (1 - b2 ** self.step)
            mb_hat = self.mb[i] / (1 - b1 ** self.step)
            vb_hat = self.vb[i] / (1 - b2 ** self.step)
            self.weights[i] -= lr * (mw_hat / (np.sqrt(vw_hat) + eps) + weight_decay * self.weights[i])
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        return loss

    def state_dict(self):
        return {**{f"w{i}": w.copy() for i, w in enumerate(self.weights)},
                **{f"b{i}": b.copy() for i, b in enumerate(self.biases)}}

    def load_state_dict(self, state):
        for i in range(len(self.weights)):
            w, b = np.asarray(state[f"w{i}"]), np.asarray(state[f"b{i}"])
            if w.shape != self.weights[i].shape or b.shape != self.biases[i].shape:
                raise ValueError("Checkpoint architecture mismatch")
            self.weights[i] = w.copy()
            self.biases[i] = b.copy()
