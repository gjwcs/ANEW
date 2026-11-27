# Activation Shape Matters: OOD Detection with Norm-Entropy Fusion

Jiawei Gu, Ziyue Qiao, Zechao Li · ACM MM 2025 · [Paper](https://doi.org/10.1145/3746027.3754972)

Norm-based out-of-distribution (OOD) detectors trust the L2 norm of a classifier's penultimate activation, but an OOD input can reach a large norm through a few spiking channels, while an in-distribution (ID) input with a moderate, evenly spread activation gets a low norm.
ANEW (Activation Norm and Entropy Weighting) also reads the *shape* of the activation: it turns the rectified activation into a distribution over channels and measures its spread with the Shannon entropy H.
The score S(x) = ‖a‖₂ · (1 + H(a)) scales the norm by this spread, so a few spiking channels no longer earn a high score and balanced ID activations are not penalised for a moderate norm.
ANEW is post-hoc: it needs one forward pass of a trained classifier, no retraining, and O(d) extra work per input.

This repository provides a concise implementation of the method. The detailed version with the full experimental code will be released later.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python demo.py
```

The demo builds synthetic penultimate activations (ID activations spread over many channels; OOD activations either spiky or weak), prints AUROC and FPR95 for the norm alone, the entropy alone and ANEW, and reports the acceptance rates of a detector whose threshold is fitted on held-out ID data. It needs no downloads and runs on a CPU.

## Usage

Score inputs with the penultimate features of any trained classifier, i.e. the input of its final linear layer:

```python
import torch
import torchvision  # only for this example; any classifier works

from anew import ANEWDetector, penultimate_features

model = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()
head = model.fc                                    # final linear layer

# x_val: held-out in-distribution images, x_test: images to screen (normalised, N x 3 x 224 x 224)
a_val = penultimate_features(model, head, x_val)   # (N, 2048) penultimate activations
detector = ANEWDetector().fit(a_val, tpr=0.95)     # threshold tau accepts 95 % of held-out ID inputs

a_test = penultimate_features(model, head, x_test)
scores = detector.score(a_test)                    # larger = more in-distribution
is_id = detector.predict(a_test)                   # True where S(x) > tau
```

If you already have penultimate features as a tensor of shape (N, d), call `anew_score(features)` directly.
Scores are oriented so that larger values mean in-distribution (Appendix D.2.3 and Fig. 2 of the paper).
The CIFAR and ImageNet benchmarks in the paper report KNN+ANEW, which uses this score in place of the norm term of KNN+NAN (Park et al., ICCV 2023); this repository provides the ANEW score itself.

## API

All functions take activations of shape (..., d), with channels in the last dimension.

| Function | Definition |
|---|---|
| `activation_norm(a)` | ‖a‖₂ (Eq. 6) |
| `channel_distribution(a)` | p_i = max(a_i, 0) / Σ_j max(a_j, 0) (Eq. 7); all-zero if no channel is positive |
| `activation_entropy(a, eps=1e-9, normalize=False)` | H(a) = −Σ_i p_i log(p_i + ε) (Eq. 8, natural log); `normalize=True` divides by log d |
| `anew_score(a, fusion="mul", beta=1.0, eps=1e-9, normalize=False)` | `"mul"`: ‖a‖₂ · (1 + H(a)) (Eq. 9, default); `"exp"`: ‖a‖₂ · exp(β H(a)) (Eq. 10) |
| `penultimate_features(model, head, x)` | input of the final layer `head` during one forward pass, flattened to (N, d) |
| `ANEWDetector(fusion="mul", beta=1.0, normalize=False, eps=1e-9)` | `.score(a)`; `.fit(a_id, tpr=0.95)` sets `tau` on held-out ID activations; `.predict(a)` returns `S > tau` (True = ID) |

Typical heads for `penultimate_features`: `model.fc` (ResNet), `model.classifier` (DenseNet), `model.classifier[-1]` (MobileNetV2), `model.heads.head` (torchvision ViT), `model.head` (torchvision Swin).

## Citation

```bibtex
@inproceedings{gu2025activation,
  title     = {Activation Shape Matters: OOD Detection with Norm-Entropy Fusion},
  author    = {Gu, Jiawei and Qiao, Ziyue and Li, Zechao},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  year      = {2025},
  doi       = {10.1145/3746027.3754972}
}
```

## Contact

gjwcs@outlook.com
