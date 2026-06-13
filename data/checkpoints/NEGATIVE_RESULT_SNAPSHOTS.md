# DiT-large-hpwl negative-result checkpoints

These are the **57.8 M-parameter** scaled DiT variant (legacy internal label
`v4b`), trained with an auxiliary HPWL-matching loss. They are the controlled
*negative result* discussed in the paper: scaling the model and adding the
HPWL auxiliary loss regressed the benchmark score instead of improving it.

Because each file is 220 MB, they are published as **GitHub Release assets**
(not in the git tree, and not via Git LFS), to avoid bloating the repository
and the LFS quota:

> **https://github.com/shankch/rerank_floorplan/releases/tag/v1.0.0**

The Code Ocean capsule omits these snapshots for size; this release is their
public companion.

## Saved snapshots

Each checkpoint is a `dict` with keys `model_state`, `step`, `config`
(57,777,156 parameters). The step numbers below are the **actual** training
steps stored in each file.

| File | Training step | SHA-256 |
|------|--------------:|---------|
| `dit_large_hpwl_step40k.pt`  |  40,000 | `a5d04251281c459797aaecff07e7e036dd64b2b1fac5f9d236434f4e3a9b8798` |
| `dit_large_hpwl_step50k.pt`  |  50,000 | `f56c3689cc6bfff84d5d61e96b7a9fb5e0483dc0ce1bc2d1cf0b85a6293a058e` |
| `dit_large_hpwl_step248k.pt` | 248,000 | `ad19a367f19a3c2e59bcd9502c9ddd3c97fb2676c8bd6f641102f4a2acda920a` |
| `dit_large_hpwl_step308k.pt` | 308,000 | `84526261bd131c33a5cfad59497b76868b9814c4c0a3527c5dd8886d13472b51` |
| `dit_large_hpwl_step356k.pt` | 356,000 | `c72ad8f73ca33f03f9bbfc8da68a582bc1abc7dd63971a827d4b099dd1063051` |

`step308k` is the checkpoint that was saved as the training "final"; `step356k`
is the latest snapshot taken before the run was stopped.

## Loading

```python
import torch
ckpt = torch.load("dit_large_hpwl_step308k.pt", map_location="cpu")
model_state = ckpt["model_state"]
print(ckpt["step"], ckpt["config"])
```

To reproduce the negative-result evaluation, load one of these into the
DiT-large path of `code/project_code/.../src/train_dit_large_hpwl.py` /
the solver's DiT loader and run the standard evaluation harness.
