# Vision tower: phantom vs. trained — resolved

**Question:** does Ornith-1.5-35B-A3B's vision tower (`model.visual.*`,
`qwen3_5_moe_vision`, 27 blocks, hidden size 1152, present in `config.json`'s
`vision_config`) carry real trained weights, or is it dead architecture
inherited from the base multimodal model and never trained — the same way
the prior project's (KAT-Coder) vision tower turned out to be phantom and
free to strip?

**Answer: genuinely trained. Not phantom.** Confirmed empirically against
the real downloaded checkpoint (`~/models/Ornith-1.5-35B-A3B`), not inferred
from the published `mmproj-*.gguf` file alone.

## Method

Standard transformer weight init leaves unambiguous fingerprints that
training erases:

- LayerNorm/RMSNorm affine weight (`norm.weight`) is initialized to an exact
  all-ones vector. An untrained module keeps `frac(weight == 1.0) ≈ 1.0` and
  near-zero cross-layer variation.
- `nn.Linear` bias is initialized to exact zero. An untrained module keeps
  `frac(bias == 0.0) ≈ 1.0`.
- `nn.Linear` weight is initialized i.i.d. from `N(0, initializer_range²)`
  (`0.02` here) — untrained modules show flat, depth-independent std ≈ 0.02
  with no structure.

Loaded `model.visual.*` tensors directly from the safetensors shards via
`model.safetensors.index.json` (33/33 files, checkpoint integrity already
verified) and computed per-tensor stats at blocks 0, 1, 5, 13, 20, 26,
comparing against the known-trained text backbone (`model.language_model.
layers.*.input_layernorm.weight`) as a positive control.

## Findings

- `model.visual.blocks.*.norm1.weight` / `norm2.weight`: **0.0-2.6% exactly
  1.0** (vs. ~100% expected untrained), real cross-layer structure — mean
  climbs from 0.47 (block 0) to 1.32 (block 26) for `norm1`, and from 0.93
  to 6.33 (peaking 7.26 at block 20) for `norm2`, with per-tensor std up to
  3.0 and maxima up to 15.8. This is the same qualitative signature as the
  text backbone's `input_layernorm.weight` (mean drifts from +0.031 to
  -0.070 across depth, std climbs from 0.049 to 0.135, 0% exactly 1.0) —
  which is unambiguously trained (this is the model the benchmark table is
  built on).
- `patch_embed.proj.bias`: mean 0.011, std 0.326, range [-3.05, +2.95] — a
  zero-initialized bias does not acquire this spread without gradient
  updates.
- `attn.qkv.weight` / `mlp.linear_fc1.weight` stay close to the 0.02 init
  scale (0.015-0.021 std across blocks 0/13/26) but show a depth-dependent
  trend rather than a flat 0.02 — consistent with training that concentrates
  most of the scale change in the norm affine weights (common in ViT-family
  training) rather than the linear weights themselves.

None of this is consistent with an untouched random init. It matches the
text backbone's own known-trained statistics closely enough to conclude the
vision tower went through the same training pipeline, not just architectural
inheritance.

## Decision

Vision-tower removal for the v1 text-only 16GB build is **still the right
call** — this project's target is single-GPU agentic coding, not multimodal
serving, and there's no VRAM budget to spare — but it is now documented
correctly as **discarding a real, trained capability for scope reasons**,
not as removing dead weight the way KAT-Coder's phantom tower was. Anyone
building on this checkpoint later who wants image understanding needs to go
back to the unpruned `ornith-ai/Ornith-1.5-35B-A3B` upstream checkpoint —
this repo's output will not have it.

No change to `scripts/prune/run_prune.sh` or the overall pipeline order is
needed: REAP prunes `model.language_model.*` MoE experts only and never
touches `model.visual.*`, so the vision tower survives the prune step
untouched regardless of this finding. The strip is a separate, explicit
step (ROADMAP.md step 6) that should now say what it's actually discarding.
