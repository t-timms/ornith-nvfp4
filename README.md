# ornith-nvfp4

Making `ornith-ai/Ornith-1.5-35B-A3B` run as a usable local agentic coding
model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120). Same
pipeline as this project's prior release
([`t-timms/kat-coder-nvfp4`](https://github.com/t-timms/kat-coder-nvfp4)):
REAP expert pruning, then NVFP4 quantization, served by vLLM.

**Status: in progress.** REAP-compatibility confirmed empirically; a SOTA
research pass and an independent audit pass are both done; the full bf16
checkpoint is downloaded; the full prune has not been launched yet (blocked
on GPU time). This README is written to survive a crash or a context reset
— see "Where things stand" below for exactly what's done, what's verified,
and what's next.

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
- **SOTA research pass done, then independently audited.** Sourced review
  across architecture, pruning, quantization, serving, and sampling
  (`docs/optimization_research_2026-08-23.md`, 20+ citations), followed by
  a second, skeptical pass that re-verified or revised each finding against
  fresh 2026 literature rather than re-confirming it (same file's "audit
  addendum" sections plus `docs/serving_notes.md`). One newer method (ZEDA,
  arXiv 2605.18643) was found and explicitly rejected — it doesn't shrink
  on-disk footprint, so it doesn't fit this project's actual constraint.
- **MTP head resolved as a real risk, not a footnote.** Ornith's
  `mtp_num_hidden_layers: 1` head is confirmed **expert-routed** (its own
  256-expert MoE, router, shared expert, full attention block — a complete
  extra decoder layer), found by inspecting the live `config.json`/
  `model.safetensors.index.json` directly. `text_config.num_experts` is a
  single config field shared between the backbone and the MTP block, so
  pruning the backbone via the stock CLI without also handling the MTP head
  would very likely produce a checkpoint that **fails to reload**, not a
  silent degradation — caught before spending any GPU time on it. Decision:
  disable the MTP head for v1 (strip `mtp.*` tensors post-prune,
  reload-verify). Full writeup: `docs/mtp_pruning_decision.md`.
- **RAM headroom checked before launch — real risk found, mitigated.**
  REAP calibration under `--residency cpu_full` (the prior build's
  validated path) needs the full model resident in CPU RAM. Ornith's
  ~71.9GiB of bf16 weights against this machine's 74GiB available (inside
  an 80GB `.wslconfig` cap) leaves only ~2-6GiB margin — too tight to
  launch as-is, unlike the prior build's ~9GB headroom. **Switched to
  `--residency layerwise`** (block-wise observe + disk offload) instead.
  Real, disclosed tradeoff: `layerwise`'s bit-for-bit reproducibility
  hasn't been validated the way `cpu_full` has — flagged as a risk to
  watch during the real run, not resolved. Full reasoning:
  `docs/ram_headroom_check.md`.
- **Quantization default corrected: GPTQ-NVFP4, not RTN.** The prior
  build's "GPTQ showed zero measurable improvement over RTN" finding does
  not transfer here (different expert topology, 256 vs. 128 experts) — no
  evidence-based reason to default to the weaker method now that
  llm-compressor's GPTQModifier is equally available. Script staged:
  `scripts/quantize/quantize_ornith_gptq.py`.
- **Checkpoint downloaded.** Full bf16 snapshot (67GB on disk) of
  `ornith-ai/Ornith-1.5-35B-A3B` at `~/models/Ornith-1.5-35B-A3B`, verified
  complete (33/33 files, no partial-download artifacts).
- **Prune, MTP-strip, and quantize scripts all staged, none executed.**
  `scripts/prune/run_prune.sh` (guards on the checkpoint existing first),
  `scripts/prune/strip_mtp.py` (untested against a real pruned checkpoint —
  none exists yet), `scripts/quantize/quantize_ornith_gptq.py` (fails fast
  if the MTP-strip step wasn't done first). **Blocked on GPU go-ahead.**

## Planned pipeline

1. ~~Download full bf16 checkpoint (71GB).~~ **Done** — 67GB on disk at
   `~/models/Ornith-1.5-35B-A3B`, verified complete.
2. **REAP prune at 50% (256→128 experts)** via `scripts/prune/run_prune.sh`
   — single seed (42), `--residency layerwise` (corrected from `cpu_full`,
   see "Where things stand"). **Staged, not executed — blocked on GPU
   go-ahead.**
3. Restore checkpoint files REAP's save path drops (same fix needed last time).
4. **MTP-head checkpoint surgery** via `scripts/prune/strip_mtp.py` — sets
   `mtp_num_hidden_layers: 0`, drops `mtp.*` tensors, reload-verifies with a
   real forward pass. New step vs. the original plan; see "Where things
   stand" and `docs/mtp_pruning_decision.md`.
5. **Quantize to NVFP4A16 via GPTQ** (`scripts/quantize/quantize_ornith_gptq.py`
   — corrected default, see "Where things stand"). Fails fast if step 4
   wasn't done first.
6. Strip the vision tower, after resolving the phantom-vs-trained question above.
7. HumanEval+/MBPP+ accuracy suite.
8. SWE-bench validation ladder: single-instance smoke → small bounded
   sample → full 50-instance pilot. Never a straight jump to the full pilot
   on unvalidated evidence — same discipline the prior project used
   throughout, including reversing a config that looked good on one
   instance and turned out to regress the score at full-pilot scale. Check
   `docs/serving_notes.md` for known-before-launch vLLM flags first.

## Relationship to the prior release

This is a separate repo, not a branch of `kat-coder-nvfp4`, because it's a
different base model with its own license (Ornith is MIT; the prior
release is Apache 2.0, inherited from KAT-Coder-V2.5-Dev) and its own
independent validation trail. The prior release stays as-is, clone-and-run
-ready, not touched by this work.

## License

MIT, inherited from the base model `ornith-ai/Ornith-1.5-35B-A3B`.
