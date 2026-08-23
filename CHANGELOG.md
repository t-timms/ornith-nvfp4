# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repo created. Base-model selection research (`docs/base_model_selection.md`):
  Ornith-1.5-35B-A3B chosen over the full Qwen3-Coder family and a broader
  HF search, with every quant variant checked at exact measured sizes and
  every benchmark number sourced as vendor vs. independent.
- REAP-compatibility on Ornith-1.5-35B-A3B confirmed empirically via a
  partial-checkpoint smoke test against real weights
  (`scripts/prune/smoke_ornith.py`) — adapter detection, fused expert
  slicing, shared-expert preservation, and the router-renormalization fix
  all pass on real Ornith weights, not just inferred from architecture tags.
- SOTA research pass across architecture/pruning/quantization/serving/
  sampling (`docs/optimization_research_2026-08-23.md`, 20+ citations),
  followed by an independent audit pass that re-verified or revised each
  finding against fresh 2026 sources rather than re-confirming it. One
  newer method considered and explicitly rejected as not applicable: ZEDA
  (arXiv 2605.18643) — doesn't shrink on-disk footprint, doesn't fit this
  project's 16GB-VRAM constraint.
- MTP-head pruning-compatibility investigation (`docs/mtp_pruning_decision.md`):
  confirmed Ornith's `mtp_num_hidden_layers: 1` head is expert-routed (a
  full extra decoder layer, its own 256-expert MoE + router + shared
  expert), and that the stock `reap-cuda` CLI would prune the backbone
  without touching the MTP head's weights while still overwriting the
  shared `text_config.num_experts` field it depends on — a router
  shape-mismatch that would only surface on reload, after a completed
  multi-hour run. Decided to disable the MTP head for v1 rather than
  attempt joint pruning (no published technique exists for the latter).
- RAM headroom check before the real prune (`docs/ram_headroom_check.md`):
  found `--residency cpu_full` (the prior build's validated path) leaves
  only ~2-6GiB margin for Ornith's ~71.9GiB of bf16 weights against this
  machine's 74GiB available — too tight to launch as-is. Switched to
  `--residency layerwise`.
- Serving notes (`docs/serving_notes.md`): `--max-cudagraph-capture-size`
  mamba-cache assertion flag, MTP-1 speculative-decoding availability
  (contingent on the MTP decision above), and an open vLLM issue (#40696)
  where prefix caching is fully ineffective below the 528-token Mamba
  block size.
- Full bf16 checkpoint downloaded (`~/models/Ornith-1.5-35B-A3B`, 67GB,
  verified complete — 33/33 files, no partial-download artifacts).
- Prune, MTP-strip, and quantize scripts staged (none executed — all
  blocked on GPU go-ahead): `scripts/prune/run_prune.sh`,
  `scripts/prune/strip_mtp.py`, `scripts/quantize/quantize_ornith_gptq.py`.

### Changed

- Quantization default corrected from RTN-first to GPTQ-NVFP4-first. The
  prior build's "GPTQ showed zero measurable improvement over RTN" finding
  doesn't transfer to Ornith's different expert topology (256 vs. 128
  experts); no evidence-based reason remained to default to the weaker
  method.
- Pruning residency mode corrected from `cpu_full` to `layerwise` per the
  RAM headroom check above. Disclosed tradeoff: `layerwise`'s bit-for-bit
  reproducibility hasn't been validated the way `cpu_full` has.

### Fixed

- Found and fixed a real bug in `reap-cuda`'s pairwise expert co-occurrence
  metric: it built a one-hot tensor as `torch.long` and matmul'd it
  directly, which CUDA has no integer GEMM kernel for. Not specific to
  this checkpoint — would hit any `qwen3_5_moe` model exercising that
  metric. Fixed upstream in the fork (cast to float32 for the matmul,
  round, cast back — exact for any realistic count):
  `t-timms/reap-cuda`, `fix/qwen3-5-router-renormalization`, commit `b67be1a`.
