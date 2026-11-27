"""ANEW: Activation Norm and Entropy Weighting for post-hoc OOD detection.

Implementation of
    J. Gu, Z. Qiao, Z. Li. "Activation Shape Matters: OOD Detection with Norm-Entropy Fusion."
    ACM MM 2025. https://doi.org/10.1145/3746027.3754972

Given the penultimate activation a = f(x) in R^d of a trained classifier, ANEW fuses the
strength of a (its L2 norm, Eq. 6) with its shape (the Shannon entropy of the rectified
activation read as a distribution over channels, Eqs. 7-8):

    S(x)      = ||a||_2 * (1 + H(a))            Eq. (9), the default
    S_beta(x) = ||a||_2 * exp(beta * H(a))      Eq. (10), optional

Larger scores mean "more in-distribution": an input is accepted as ID when S(x) > tau
(App. D.2.3, Fig. 2), with tau calibrated on held-out ID data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = [
    "activation_norm",
    "channel_distribution",
    "activation_entropy",
    "anew_score",
    "penultimate_features",
    "ANEWDetector",
]


def activation_norm(a: Tensor) -> Tensor:
    """Overall activation strength ||a||_2, Eq. (6).

    Args:
        a: floating-point activations of shape (..., d); the last dimension holds the channels.

    Returns:
        Tensor of shape (...).
    """
    return torch.linalg.vector_norm(a, ord=2, dim=-1)


def channel_distribution(a: Tensor) -> Tensor:
    """Rectified activation normalised over channels, Eq. (7):

        p_i = max(a_i, 0) / sum_j max(a_j, 0).

    Rows whose activations are all non-positive get p = 0, so their entropy is 0
    (the all-negative case of Algorithm 1).

    Args:
        a: floating-point activations of shape (..., d).

    Returns:
        Tensor of shape (..., d) whose rows sum to 1 (or are all zero).
    """
    a_pos = a.clamp_min(0)
    total = a_pos.sum(dim=-1, keepdim=True)
    return a_pos / total.clamp_min(torch.finfo(a_pos.dtype).tiny)


def activation_entropy(a: Tensor, eps: float = 1e-9, normalize: bool = False) -> Tensor:
    """Activation shape: Shannon entropy (natural log) of the channel distribution, Eq. (8):

        H(a) = -sum_i p_i log(p_i + eps).

    H is large when many channels share the activation and small when a few channels
    dominate. eps keeps the log finite at p_i = 0 and sits inside the log as in
    Algorithm 1. With normalize=True, H is divided by log d to lie in [0, 1]
    (optional in Sec. 2.3.3; Table 19 reports the same accuracy).

    Args:
        a: floating-point activations of shape (..., d), d >= 2.
        eps: numerical constant (the paper uses 1e-9).
        normalize: divide by log d.

    Returns:
        Tensor of shape (...).
    """
    p = channel_distribution(a)
    h = -(p * torch.log(p + eps)).sum(dim=-1)
    if normalize:
        h = h / math.log(a.shape[-1])
    return h


def anew_score(
    a: Tensor,
    fusion: str = "mul",
    beta: float = 1.0,
    eps: float = 1e-9,
    normalize: bool = False,
) -> Tensor:
    """ANEW score of penultimate activations; larger means more in-distribution.

        fusion="mul":  S = ||a||_2 * (1 + H(a))           Eq. (9), the paper's default
        fusion="exp":  S = ||a||_2 * exp(beta * H(a))     Eq. (10), beta > 0

    The norm uses a as given; the entropy uses the rectified activation (Algorithm 1).

    Args:
        a: floating-point activations of shape (..., d).
        fusion: "mul" or "exp".
        beta: strength of the entropy term for fusion="exp" (default 1.0, Table 8).
        eps: numerical constant of the entropy.
        normalize: use the entropy divided by log d.

    Returns:
        Tensor of shape (...).
    """
    norm = activation_norm(a)
    h = activation_entropy(a, eps=eps, normalize=normalize)
    if fusion == "mul":
        return norm * (1.0 + h)
    if fusion == "exp":
        return norm * torch.exp(beta * h)
    raise ValueError(f"fusion must be 'mul' or 'exp', got {fusion!r}")


@torch.no_grad()
def penultimate_features(model: nn.Module, head: nn.Module, x: Tensor) -> Tensor:
    """Penultimate activation a = f(x): the input of the classifier's final layer (Sec. 2.3.1).

    Registers a forward pre-hook on `head`, runs one forward pass of `model` and returns the
    hooked input flattened to (N, d). Typical heads: model.fc (ResNet), model.classifier
    (DenseNet), model.classifier[-1] (MobileNetV2), model.heads.head (torchvision ViT),
    model.head (torchvision Swin). Put the model in eval() mode first.

    Args:
        model: trained classifier.
        head: its final linear layer.
        x: a batch of inputs for `model`.

    Returns:
        Tensor of shape (N, d).
    """
    captured: list[Tensor] = []
    handle = head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].flatten(1)))
    try:
        model(x)
    finally:
        handle.remove()
    return captured[0]


@dataclass
class ANEWDetector:
    """ANEW with a decision threshold: an input is in-distribution when S(x) > tau (App. D.2.3).

    fit() sets tau on held-out ID activations so that a fraction `tpr` of them is accepted;
    tpr = 0.95 is the operating point of the FPR95 metric.
    """

    fusion: str = "mul"
    beta: float = 1.0
    normalize: bool = False
    eps: float = 1e-9
    tau: Optional[float] = None

    def score(self, a: Tensor) -> Tensor:
        """ANEW scores of activations of shape (..., d); larger means more in-distribution."""
        return anew_score(a, fusion=self.fusion, beta=self.beta, eps=self.eps, normalize=self.normalize)

    def fit(self, id_activations: Tensor, tpr: float = 0.95) -> ANEWDetector:
        """Set tau to the (1 - tpr) quantile of the scores of held-out ID activations."""
        scores = self.score(id_activations).flatten().double()
        self.tau = torch.quantile(scores, 1.0 - tpr).item()
        return self

    def predict(self, a: Tensor) -> Tensor:
        """Boolean tensor of shape (...): True = in-distribution (S > tau)."""
        if self.tau is None:
            raise RuntimeError("call fit() on held-out in-distribution activations first")
        return self.score(a) > self.tau
