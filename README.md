# ornith-nvfp4

Making `ornith-ai/Ornith-1.5-35B-A3B` run as a usable local agentic coding
model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120). Same
pipeline as this project's prior release
([`t-timms/kat-coder-nvfp4`](https://github.com/t-timms/kat-coder-nvfp4)):
REAP expert pruning, then NVFP4 quantization, served by vLLM.

**Status: in progress.** REAP-compatibility confirmed empirically; the full
prune has not started yet. This README is written to survive a crash or a
context reset — see "Where things stand" below for exactly what's done,
what's verified, and what's next.

## Why a new base model

The prior release (KAT-Coder-V2.5-Dev, REAP-50% + NVFP4A16) reached 52.0%
on SWE-bench Verified, below the 56.4% comparable bar (Devstral Small 2512,
same mini-swe-agent bash-only scaffold). Every reasonable lever tried to
close that gap on that base model failed at meaningful scale: GPTQ
requantization (no measurable accuracy difference vs. plain RTN),
`presence_penalty`/`top_k` sampling (regressed the score, 48.0% vs. 52.0%),
a custom "investigation discipline" prompt (regressed twice), and two
different chat-template swaps (both regressed). Three genuinely different
intervention types failing against the same non-convergence failure mode
reads as a capability ceiling for that specific base model, not a missing
configuration fix — see that project's `ROADMAP.md` for the full evidence
trail on each of these.

Ornith-1.5-35B-A3B was chosen as the next base model after an exhaustive
comparison (full writeup: `docs/base_model_selection.md`), not because its
own vendor-claimed benchmarks are trusted at face value — they explicitly
are not (see that doc for a demonstrated 20pp discrepancy between two
vendors rating the same checkpoint). It's a reasoned bet on architecture
similarity (same `qwen3_5_moe` family, same REAP-pruning approach already
proven to work) and a validated-safe pruning ratio, not a benchmark-proven
choice — that proof only comes from building and measuring it ourselves,
same as the prior release.

## Where things stand (2026-08-23)

- **Exhaustive base-model survey done.** Compared Ornith-1.5-35B-A3B against
  the full Qwen3-Coder family and a broader HF search. No candidate beats
  it on the metric that matters (an independently-comparable agentic-coding
  score); the strongest alternative (Qwen3-Coder-30B-A3B) has *better*
  documented REAP-pruning evidence but is independently confirmed too weak
  (18.8% vs. our current 52.0%/Devstral's 56.4% bar). Full comparison table,
  every quant variant checked with exact measured sizes, sourcing on every
  benchmark number (vendor vs. independent): `docs/base_model_selection.md`.
- **No pre-pruned shortcut exists.** Checked all 8 quant variants
  ornith-ai has published (GGUF at 4 bit-depths, FP8, NVFP4, MLX x3) with
  exact measured file sizes. Smallest is GGUF Q4_K_M at 21.71 GiB — still
  over budget. REAP pruning is required regardless of quantization format.
  As far as this search can tell, a REAP-pruned build of this model does
  not exist yet anywhere — if this succeeds, it's likely the first.
- **REAP-compatibility CONFIRMED empirically**, not just inferred from
  architecture tags. Ran a partial-checkpoint smoke test (layers 0-2 +
  embed + lm_head, ~13GiB of the real 71GiB weights, not a synthetic
  model) through the real `reap-cuda` pipeline: adapter detection, fused
  expert slicing (256→56 in the smoke test's ratio), shared-expert
  preservation, and — critically — the router-renormalization fix
  (`Qwen3_5MoeModelAdapter.renormalizes_router_weights`, architecture
  -detected, not hardcoded to the prior model) all confirmed working on
  real Ornith weights. Found and fixed one real, general bug along the way
  (`reap-cuda` had no CUDA path for integer matmul in its co-occurrence
  metric — not Ornith-specific, would hit any `qwen3_5_moe` checkpoint
  using that code path). Fixed and pushed to
  `t-timms/reap-cuda`, `fix/qwen3-5-router-renormalization`, commit
  `b67be1a`. Script: `scripts/prune/smoke_ornith.py` (mirrors
  `reap-cuda/scripts/smoke_ornith.py`, kept in both places).
- **Open question, not yet resolved**: whether Ornith's vision tower is
  phantom (untrained, like the prior model's — free to strip) or genuinely
  trained. Evidence leans toward "genuinely trained": ornith-ai's GGUF repo
  publishes a dedicated `mmproj-*.gguf` multimodal projector file, which is
  specifically what enables real image understanding in llama.cpp — the
  prior model never published an equivalent. If confirmed trained, stripping
  it for our text-only use case is still almost certainly the right call,
  but it's a deliberate capability trade-off, not a free removal of
  untrained weights the way it was last time. Verify before stripping.
- **Full prune not yet started.** Needs the full 71GiB bf16 download first
  (partial slice used for the smoke test is not enough for a real
  calibration run needing all 40 layers). RAM is tighter than the prior
  build: REAP calibration needs the full model resident in CPU RAM
  (`--residency cpu_full`), and Ornith's 71GB against this machine's 78GB
  leaves only ~6GB headroom (the prior model's 69.3GB left ~9GB). Not
  disqualifying, worth watching.

## Planned pipeline (not yet executed)

1. Download full bf16 checkpoint (71GB).
2. REAP prune at 50% (256→128 experts) via the real CLI:
   `reap prune layerwise --model <ornith> --dataset theblackcat102/evol-codealpaca-v1
   --compression-ratio 0.50 --prune-method reap --observe-backend bmm
   --residency cpu_full --batch-size 1 --batches-per-category 64
   --model-max-length 2048 --seed 42` — same recipe validated on the prior
   model, calibration alone took 57.5 min there.
3. Restore checkpoint files REAP's save path drops (same fix needed last time).
4. Quantize to NVFP4A16 via plain RTN first (82s, data-free) — GPTQ showed
   zero measurable improvement over RTN on the prior model's build, so it's
   not the default here either; test it only as a follow-up if RTN's
   accuracy suggests there's room.
5. Strip the vision tower, after resolving the phantom-vs-trained question above.
6. HumanEval+/MBPP+ accuracy suite.
7. SWE-bench validation ladder: single-instance smoke → small bounded
   sample → full 50-instance pilot. Never a straight jump to the full pilot
   on unvalidated evidence — same discipline the prior project used
   throughout, including reversing a config that looked good on one
   instance and turned out to regress the score at full-pilot scale.

## Relationship to the prior release

This is a separate repo, not a branch of `kat-coder-nvfp4`, because it's a
different base model with its own license (Ornith is MIT; the prior
release is Apache 2.0, inherited from KAT-Coder-V2.5-Dev) and its own
independent validation trail. The prior release stays as-is, clone-and-run
-ready, not touched by this work.

## License

MIT, inherited from the base model `ornith-ai/Ornith-1.5-35B-A3B`.
