# ornith-nvfp4

Making `ornith-ai/Ornith-1.5-35B-A3B` run as a usable local agentic coding
model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120). Same
pipeline as this project's prior release
([`t-timms/kat-coder-nvfp4`](https://github.com/t-timms/kat-coder-nvfp4)):
REAP expert pruning, then NVFP4 quantization, served by vLLM.

**Status: released and fully evaluated, including agentic.** The
pipeline is complete: REAP 50% prune (256→128 experts), MTP-head strip,
GPTQ-NVFP4A16 quantization, vision-tower strip — 12.47 GiB, served
end-to-end by vLLM after three real upstream gaps were patched locally.
Quality: **HumanEval+ 84.15% / HumanEval 90.24% / MBPP+ 89.15%** (greedy,
instruct framing), and **SWE-bench Verified 22/50 = 44.0% resolved**
(81.5% resolved-of-completed, official harness, zero infra failures)
via `mini-swe-agent`'s bash-only scaffold at a 49K-token context ceiling —
on the same 50-instance slice as the prior release's 52.0%, making it an
apples-to-apples comparison: the entire gap is submission rate, not patch
quality (see `docs/model_card.md` for the full breakdown). Weights and
card are live at
[`Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16).
This README is written to survive a crash or a context reset — see
"Where things stand" below for exactly what's done, what's verified, and
what's next.

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

## Where things stand (2026-08-26)

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
- **Vision-tower question resolved: genuinely trained, not phantom.**
  Confirmed empirically against the real downloaded checkpoint, not just
  inferred from the published `mmproj-*.gguf` file. `model.visual.*` norm
  weights show real cross-layer structure and are essentially never exactly
  1.0 (the init value) — the same statistical signature as the text
  backbone's known-trained layernorm weights — and linear-layer biases have
  drifted far from their zero init. Unlike the prior model's confirmed-
  phantom tower, this one was actually trained on images. Stripping it for
  this project's text-only 16GB target is still the right call, but it's
  now documented as discarding a real capability for scope reasons, not
  removing dead weight. Full method and stats: `docs/vision_tower_decision.md`.
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
- **RAM headroom checked before launch (2026-08-23), then corrected
  (2026-08-24).** REAP calibration under `--residency cpu_full` (the prior
  build's validated path) needs the full model resident in CPU RAM.
  Original estimate: Ornith's ~71.9GiB of bf16 weights against this
  machine's 74GiB available (inside an 80GB `.wslconfig` cap) left only
  ~2-6GiB margin — judged too tight, switched to `--residency layerwise`
  (block-wise observe + disk offload) instead. That switch then surfaced
  three real `reap-cuda` bugs on this hybrid GDN+attention MoE model (see
  `docs/layerwise_prune_run_2026-08-24.md`); rather than keep debugging a
  code path this project doesn't actually need, re-measured the RAM math
  instead of trusting the original estimate a second time. The real prune
  run's own logs report an actual **~65.4GiB** weight footprint, not
  ~71.9GiB — a genuine ~11-12GiB margin against this box's ~77GiB free RAM.
  **Reverted to `--residency cpu_full`**, confirmed completing the full
  pipeline cleanly at both reduced and full scale before the real overnight
  run launched. Full reasoning and the correction: `docs/ram_headroom_check.md`.
- **Quantization default corrected: GPTQ-NVFP4, not RTN.** The prior
  build's "GPTQ showed zero measurable improvement over RTN" finding does
  not transfer here (different expert topology, 256 vs. 128 experts) — no
  evidence-based reason to default to the weaker method now that
  llm-compressor's GPTQModifier is equally available. Script staged:
  `scripts/quantize/quantize_ornith_gptq.py`.
- **Checkpoint downloaded.** Full bf16 snapshot (67GB on disk) of
  `ornith-ai/Ornith-1.5-35B-A3B` at `~/models/Ornith-1.5-35B-A3B`, verified
  complete (33/33 files, no partial-download artifacts).
- **Full REAP prune completed (2026-08-24).** GPU go-ahead given;
  small-scale invocation validation ahead of the real run caught three
  `reap-cuda` bugs before any GPU-hours were spent (two fixed and verified
  end-to-end under `--residency layerwise`, one found but not fixed one
  step further into the pipeline) — full incident writeup:
  `docs/layerwise_prune_run_2026-08-24.md`. `scripts/prune/run_prune.sh`
  ran under `--residency cpu_full` (see the RAM headroom correction above)
  to completion in ~48 minutes with zero errors: **256→128 experts, 36GB
  checkpoint** at `~/reap-stability/ornith_n64_s42/model_.../pruned_models/
  reap-renorm_true-seed_42-0.50`, `num_experts: 128` and
  `mtp_num_hidden_layers: 1` confirmed correct in `config.json`.
- **MTP-head stripped (2026-08-24).** `scripts/prune/strip_mtp.py`, run for
  the first time against a real checkpoint after being written and reviewed
  (but untested) the day before. Two assumptions turned out wrong and were
  fixed against the real thing: the checkpoint layout is a single ~35GiB
  `model.safetensors` with no index (the script assumed sharding), and the
  MTP head's weights were already entirely absent from the pruned output —
  not "present but shape-mismatched" as originally predicted (see
  `docs/mtp_pruning_decision.md`'s 2026-08-24 addendum). The config fix
  (`mtp_num_hidden_layers: 0`) was still necessary and correctly applied
  either way. Reload-verified with a real CPU forward pass, independently
  confirmed rather than just trusted from the script's exit code: logits
  shape `(1, 4, 248320)` matches the model's actual `vocab_size` exactly.
  Output: `<prune dir>-mtp-stripped`, 36GB, 1026 tensors, zero `mtp.*` keys.
- **GPTQ-NVFP4 quantization completed (2026-08-25, ~5.5 hours).** The
  mechanism-verification work below preceded it and paid off: the full run
  at the probed settings completed with zero errors, producing a **12.47
  GiB** compressed-tensors `nvfp4-pack-quantized` checkpoint at
  `~/models/ornith-nvfp4a16-gptq` (291 tensors unquantized via the ignore
  list — routers, gates, embeddings, linear-attention projections).
  Post-quant surgery produced the serving artifact:
  `ornith-nvfp4a16-gptq-text-only-norope`, i.e. vision tower stripped
  (`scripts/prune/strip_vision.py` — switch to transformers' real
  text-only `Qwen3_5MoeForCausalLM` class, `model.language_model.*` key
  renames, 333 visual tensors dropped) plus mrope position fields removed
  from config (text-only serving doesn't use them). What looked like a
  corrupted quantization right after the run was in fact a load-time bug:
  see the llm-compressor gap under "Fixed" in `CHANGELOG.md`.
- **First real accuracy evaluation (2026-08-25), via vLLM.** Getting there
  required patching three confirmed upstream gaps locally: the
  llm-compressor 2D-load gap above (silently dropped all quantized expert
  weights; local patch `scripts/patches/patch_qwen3_5_moe_2d_load.py`,
  cross-confirmed against llm-compressor PR #3080 which names an Ornith
  checkpoint as affected), a missing vLLM registry entry for
  `Qwen3_5MoeForCausalLM`, and three mamba-state classmethods missing from
  `Qwen3_5ForCausalLMBase` that `IsHybrid` requires (crash on first request,
  after model load and KV init succeed — the worst possible failure mode to
  debug cold). Results, greedy/instruct framing via lm-eval-harness on the
  patched vLLM backend:

  | task | score | n |
  |---|---|---|
  | HumanEval+ | **84.15%** | 164 |
  | HumanEval | 90.24% | 164 |
  | MBPP+ | 89.15% | 378 |

  kat-coder-16gb scored 96.34/89.63/89.42 on the same tasks — Ornith is
  lower across the board but same tier, reasonable given more aggressive
  (50%) REAP pruning on a different base model. Raw results:
  `~/eval-suite-ornith-vllm/`; launch scripts and task definitions:
  `scripts/eval/`.
- **SWE-bench Verified ladder completed and graded (2026-08-26).**
  Smoke → 10-instance gate → full 50-instance pilot, all rungs passed
  (`scripts/swebench/run_ladder_night.sh`; overnight run finished
  50/50 instances, 96.5% prefix-cache hit rate, served at the 49,152
  window with a 54,067-token KV budget). Graded with the official harness
  (`scripts/swebench/grade_pilot.sh`): **22/50 = 44.0% resolved, 22/27 =
  81.5% resolved-of-completed, zero infra or eval errors.** Rollout
  breakdown: 27 Submitted / 15 ContextWindowExceeded / 6 LimitsExceeded /
  2 RepeatedFormatError. The 50-instance slice is identical to the prior
  release's 52.0% run, so the comparison is same-instance under the same
  scaffold, window, and step limit: the prior release completed 33
  instances (26 resolved, 81.25% of completed) vs this build's 27 (22
  resolved, 81.5% of completed) — the headline gap is entirely submission
  rate, not resolve quality; 18 instances were resolved by both builds.
  Sampling followed the checkpoint's own `generation_config.json`
  (temperature 1.0 / top_p 0.95 / top_k 20), flagged as unvalidated in
  `scripts/swebench/ornith_overrides_ladder.yaml`. Model card updated and
  re-uploaded to the Hub with the full section.
- **Released on the Hugging Face Hub (2026-08-25).** Weights (12.47 GiB),
  config, and tokenizer live at
  `Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16`; card is
  `docs/model_card.md` uploaded as `README.md`.
- **Quantization mechanism verified and ETA measured first (2026-08-24).** `scripts/quantize/quantize_ornith_gptq.py` requires a
  separate environment (`~/quant-env` — `llmcompressor` isn't installed in
  the `reap-cuda-env` used for pruning). Before committing ~5 GPU-hours,
  ran two bounded checks: (1) an 8-sample smoke test confirmed GPTQ
  actually reaches the fused MoE expert weights — a real open question,
  since `targets="Linear"` normally matches `nn.Linear` module instances by
  type, and Ornith's experts are stored as raw fused parameters on a custom
  container, not individual `nn.Linear` submodules; the log showed correct
  per-expert quantization (`Quantizing ...experts.9.gate_proj using 8
  samples`) confirming `moe_calibrate_all_experts=True` does reach them,
  with zero errors across 5 layers' worth of experts. (2) A second probe at
  the real calibration settings (256 samples, 2048 seq length) measured
  actual per-module time: ~3.27s/expert (1.45s gate_proj + 1.45s up_proj +
  0.37s down_proj), essentially unchanged from the 8-sample measurement —
  confirming the cost is dominated by the fixed GPTQ solve (Hessian/
  Cholesky math scaling with matrix dimensions), not calibration data
  volume. With 40 layers × 128 experts = 5,120 experts needing individual
  quantization, that's **~4.65 hours for the expert MLPs alone, ~4.75-5
  hours total** — which the real run then came in just above (~5.5h wall
  clock). Both probe runs were killed once their measurement purpose was
  served, not left to run to completion.

## Planned pipeline

1. ~~Download full bf16 checkpoint (71GB).~~ **Done** — 67GB on disk at
   `~/models/Ornith-1.5-35B-A3B`, verified complete.
2. ~~REAP prune at 50% (256→128 experts)~~ **Done** (2026-08-24) via
   `scripts/prune/run_prune.sh` — single seed (42), `--residency cpu_full`
   (see "Where things stand" for the `layerwise` detour and why it was
   reverted). 36GB checkpoint, config verified, zero errors.
3. Restore checkpoint files REAP's save path drops (same fix needed last time).
4. ~~MTP-head checkpoint surgery~~ **Done** (2026-08-24) via
   `scripts/prune/strip_mtp.py` — sets `mtp_num_hidden_layers: 0`, drops any
   `mtp.*` tensors present (turned out there were none — see "Where things
   stand"), reload-verifies with a real forward pass. New step vs. the
   original plan; see "Where things stand" and `docs/mtp_pruning_decision.md`.
5. ~~Quantize to NVFP4A16 via GPTQ~~ **Done** (2026-08-24→25, ~5.6h wall
   clock per the run log) via `scripts/quantize/quantize_ornith_gptq.py` —
   12.47 GiB checkpoint, zero errors.
6. ~~Strip the vision tower~~ **Done** (2026-08-25) via
   `scripts/prune/strip_vision.py`, plus a config-only mrope strip for
   text-only serving. Confirmed genuinely trained, not phantom (see
   `docs/vision_tower_decision.md`) — removal remains a deliberate scope
   trade-off, not a free removal of dead weight.
7. ~~HumanEval+/MBPP+ accuracy suite.~~ **Done** (2026-08-25) — HumanEval+
   84.15%, HumanEval 90.24%, MBPP+ 89.15%; first genuine quality signal of
   the whole pipeline, served through locally-patched vLLM (three upstream
   gaps fixed along the way — see "Where things stand").
8. ~~SWE-bench validation ladder~~ **Done** (2026-08-26) — smoke →
   10-instance gate → full 50-instance pilot, then graded with the official
   harness: **22/50 = 44.0% resolved (81.5% of completed), zero infra
   failures** — same-instance comparison to the prior release's 52.0%;
   the gap is submission rate, not patch quality. Never a straight jump to
   the full pilot on unvalidated evidence — same discipline the prior
   project used throughout, including reversing a config that looked good
   on one instance and turned out to regress the score at full-pilot scale.
   Harness: `scripts/swebench/`.

## Relationship to the prior release

This is a separate repo, not a branch of `kat-coder-nvfp4`, because it's a
different base model with its own license (Ornith is MIT; the prior
release is Apache 2.0, inherited from KAT-Coder-V2.5-Dev) and its own
independent validation trail. The prior release stays as-is, clone-and-run
-ready, not touched by this work.

## License

MIT, inherited from the base model `ornith-ai/Ornith-1.5-35B-A3B`.
