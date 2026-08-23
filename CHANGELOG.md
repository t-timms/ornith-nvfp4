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

### Fixed

- Found and fixed a real bug in `reap-cuda`'s pairwise expert co-occurrence
  metric: it built a one-hot tensor as `torch.long` and matmul'd it
  directly, which CUDA has no integer GEMM kernel for. Not specific to
  this checkpoint — would hit any `qwen3_5_moe` model exercising that
  metric. Fixed upstream in the fork (cast to float32 for the matmul,
  round, cast back — exact for any realistic count):
  `t-timms/reap-cuda`, `fix/qwen3-5-router-renormalization`, commit `b67be1a`.
