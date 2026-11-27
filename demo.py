"""Toy demo of ANEW on synthetic penultimate activations (CPU only, no downloads).

ID activations are spread over many channels. OOD activations are either "spiky"
(a few channels carry a large norm, the case that fools a norm-only score) or "weak"
(spread out but small, the case a norm catches). The script prints AUROC and FPR95,
with ID as the positive class, for the norm alone, the entropy alone and ANEW, then the
acceptance rates of an ANEWDetector whose threshold is fitted on held-out ID data.

Run:  python demo.py
"""
from __future__ import annotations

import numpy as np
import torch

from anew import ANEWDetector, activation_entropy, activation_norm, anew_score

D = 256             # channels of the penultimate layer
N = 2000            # ID samples per split; the OOD set has the same size
GEN = torch.Generator().manual_seed(0)
# ten class prototypes, each activating about 60 % of the channels
PROTOTYPES = (torch.rand(10, D, generator=GEN) < 0.6).float() * (0.3 + 0.7 * torch.rand(10, D, generator=GEN))


def sample_id(n: int) -> torch.Tensor:
    """ID: a class prototype at a random overall strength (some ID inputs have a moderate norm)."""
    cls = torch.randint(0, len(PROTOTYPES), (n,), generator=GEN)
    strength = 0.6 + 0.8 * torch.rand(n, 1, generator=GEN)
    return (strength * PROTOTYPES[cls] + 0.1 * torch.randn(n, D, generator=GEN)).clamp_min(0)


def sample_spiky_ood(n: int, spikes: int = 4) -> torch.Tensor:
    """OOD that fakes a large norm: a few channels carry large values, the rest is near zero."""
    base = 0.02 * torch.rand(n, D, generator=GEN)
    idx = torch.rand(n, D, generator=GEN).argsort(dim=1)[:, :spikes]
    return base.scatter(1, idx, 2.0 + 4.0 * torch.rand(n, spikes, generator=GEN))


def sample_weak_ood(n: int) -> torch.Tensor:
    """OOD with a spread but weak response: an ID-like shape at a low overall strength."""
    return 0.35 * sample_id(n)


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Area under the ROC curve, ID = positive class (Mann-Whitney U with averaged ties)."""
    scores = np.concatenate([id_scores, ood_scores])
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]      # average 1-based rank
    n_id, n_ood = len(id_scores), len(ood_scores)
    return float((rank[:n_id].sum() - n_id * (n_id + 1) / 2.0) / (n_id * n_ood))


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Fraction of OOD inputs accepted at the threshold that accepts 95 % of ID inputs."""
    tau = np.quantile(id_scores, 0.05)
    return float((ood_scores > tau).mean())


def main() -> None:
    id_val, id_test = sample_id(N), sample_id(N)
    ood = {"spiky OOD": sample_spiky_ood(N // 2), "weak OOD": sample_weak_ood(N - N // 2)}
    ood_all = torch.cat(list(ood.values()))

    scorers = {
        "norm only": activation_norm,
        "entropy only": activation_entropy,
        "ANEW N(1+H)": anew_score,
        "ANEW N exp(H)": lambda a: anew_score(a, fusion="exp", beta=1.0),
    }
    print(f"synthetic penultimate activations: d = {D}, {N} ID test, {len(ood_all)} OOD (spiky + weak)")
    print(f"{'score':<15}{'AUROC %':>9}{'FPR95 %':>9}")
    for name, score in scorers.items():
        s_id, s_ood = score(id_test).numpy(), score(ood_all).numpy()
        print(f"{name:<15}{100 * auroc(s_id, s_ood):9.2f}{100 * fpr_at_95_tpr(s_id, s_ood):9.2f}")

    detector = ANEWDetector().fit(id_val, tpr=0.95)
    print(f"\nANEWDetector, tau = {detector.tau:.2f} fitted on held-out ID data; accepted as ID:")
    print(f"  {'ID test':<10}{100 * detector.predict(id_test).float().mean().item():7.2f} %")
    for name, a in ood.items():
        print(f"  {name:<10}{100 * detector.predict(a).float().mean().item():7.2f} %")


if __name__ == "__main__":
    main()
