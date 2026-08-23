# Base model selection: why Ornith-1.5-35B-A3B

Two research passes, 2026-08-23. First pass had real coverage gaps (caught
by hand-checking afterward, not by the research itself) — documented here
so the same gaps don't get repeated: it checked one of Ornith's 8 published
quant variants and used a rough size estimate instead of exact measurement.
Second pass was exhaustive and precise; this doc reflects the corrected,
verified findings.

## Hard constraints

- Must fit ~12-13GB after REAP pruning + 4-bit quantization + stripping any
  phantom multimodal weights, leaving ~3GB+ headroom for KV cache.
- REAP pruning ratio ceiling: this project's own prior research found >50%
  pruning starts risking measurable long-horizon-editing degradation, never
  validated past that. Candidates needing much more than ~50-55% are weaker
  choices, not disqualified outright, but flagged.
- Apache-2.0 or similarly permissive license preferred.
- Needs a plausible REAP-pruning path — either llm-compressor's native REAP
  modifier or the standalone `reap`/`reap-cuda` tool supports the model's
  MoE implementation, or it would need adapter work like the prior model did.

## Comparison table

| Model | Total/Active | Prune ratio needed | Smallest existing quant (exact, measured) | REAP compatibility | Best independently-comparable score | License |
|---|---|---|---|---|---|---|
| **Ornith-1.5-35B-A3B** (chosen) | 35.95B/~3B | ~50% | GGUF Q4_K_M, 21.71 GiB | **Confirmed empirically** — see below | 79 (vendor-only, OpenHands, wrong scaffold vs. our bar) | MIT |
| Qwen3-Coder-30B-A3B-Instruct | 30.53B/3.3B | ~25-50% (shallower) | NVFP4/AWQ 16.85 GiB | Better *documented* evidence than Ornith at selection time (a real published llm-compressor REAP example on this exact family, 99.80% accuracy recovery at 50%) | **18.8%** (independent, bash-only scaffold) — far below our 52.0%/Devstral's 56.4% bar | Apache 2.0 |
| Qwen3-Coder-Next | 79.67B/~3B | ~85%+ (far beyond validated ceiling) | FP8 ~74.3 GiB | Different architecture (`qwen3_next`), unconfirmed | 70.6 (vendor, SWE-Agent scaffold — wrong instrument, not comparable) | Apache 2.0 |
| Moonlight-16B-A3B-Instruct | 15.96B/3B | ~25-50% | — | **Tested and fails**: REAP's own published numbers show 50% sparsity collapses to 16.99% accuracy recovery | Not evaluated (disqualified on pruning grounds) | MIT |
| Qwen3-Coder-480B-A35B | 480B/35B | >97% | — | Not evaluated — ratio alone disqualifies | — | Apache 2.0 |

No independently-verified SWE-bench-Verified-or-comparable number exists
for any candidate under a scaffold matching our actual bar (mini-swe-agent
bash-only). Every vendor number cited above is flagged as such because this
project has already caught a concrete case of vendor cross-table
unreliability: Kwaipilot's own card rates a prior Ornith generation
(Ornith-1.0-35B) at 55.80; Ornith's own card rates that same checkpoint at
75.6 — a 20pp discrepancy from which vendor is reporting, for the same
model.

## The real finding: best pruning evidence isn't the same as best model

Qwen3-Coder-30B-A3B had *stronger* REAP-compatibility evidence than Ornith
at the time of selection — a real published accuracy-recovery number, not
an inferred claim. It loses anyway: its actual independently-verified score
under our comparable scaffold is 18.8%, a full model generation behind.
Good pruning behavior doesn't rescue a weak base model. Moonlight is a
clean, concrete disqualification — tested by REAP's own authors, collapses
at 50% sparsity.

## No pre-pruned shortcut exists for Ornith

All 8 variants ornith-ai has published, checked directly with exact
measured sizes (not vendor README claims):

| Variant | Exact size | Fits 16GB? |
|---|---:|:---:|
| BF16 (base) | 71.07 GiB | no |
| GGUF Q8_0 | 37.80 GiB | no |
| FP8 | 36.66 GiB | no |
| GGUF Q6_K | 29.21 GiB | no |
| GGUF Q5_K_M | 25.35 GiB | no |
| NVFP4 | 21.81 GiB | no |
| GGUF Q4_K_M (smallest offered) | 21.71 GiB | no |
| MLX (all bit-widths) | — | n/a, Apple Silicon only |

Even the most aggressive published quant is 5.7GB over budget. One
curiosity chased down and ruled out: the FP8 repo has a file named
`.model_only_pruned` — it's an empty (0-byte) marker, and the actual
measured size (36.66 GiB) matches a full unpruned model almost exactly, so
whatever that flag means in ornith-ai's build pipeline, it isn't meaningful
REAP-style pruning.

**As far as this search can tell, a REAP-pruned build of this model does
not exist anywhere yet.**

## REAP-compatibility: confirmed empirically, not just inferred

Checked at three levels, escalating from cheapest to most direct:

1. **Code-level**: `reap-cuda`'s router-renormalization fix
   (`Qwen3_5MoeModelAdapter.renormalizes_router_weights`) is
   architecture-detected — it triggers on the `Qwen3_5MoeSparseMoeBlock`
   module class any `qwen3_5_moe`-tagged model instantiates via stock
   `transformers`, not hardcoded to the prior model specifically.
2. **Config-level**: Ornith's `config.json` declares
   `architectures: ["Qwen3_5MoeForConditionalGeneration"]`,
   `model_type: "qwen3_5_moe"`, **no `auto_map`** — confirming it uses that
   exact stock modeling code, not custom remote code that could diverge
   from what the adapter expects. Same hybrid 3:1 linear/full attention
   pattern, same `shared_expert`/`shared_expert_gate` structure.
3. **Live, on real weights**: adapted `reap-cuda`'s existing smoke-test
   pattern (`scripts/smoke_qwen35.py`, originally written for a
   structurally-identical 256-expert case) into `scripts/smoke_ornith.py`,
   fetched a partial checkpoint slice (layers 0-2 + embed + lm_head, ~13GiB
   of the real 71GiB, identified via `model.safetensors.index.json` rather
   than guessing which shards were needed), and ran the real pipeline
   against real Ornith weights. **Passed clean**: adapter detection, fused
   expert slicing, shared-expert preservation, and the renormalization fix
   all confirmed. One real bug found along the way — `reap-cuda` had no
   CUDA path for integer matmul in its pairwise-cooccurrence metric (not
   Ornith-specific, would hit any `qwen3_5_moe` checkpoint exercising that
   code path) — fixed and pushed to `t-timms/reap-cuda`,
   `fix/qwen3-5-router-renormalization`, commit `b67be1a`.

## Sources

- Ornith-1.5-35B-A3B: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
- Qwen3-Coder-30B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Qwen3-Coder-Next: https://huggingface.co/Qwen/Qwen3-Coder-Next (name as found; verify on revisit)
- Moonlight-16B-A3B-Instruct REAP results: from REAP's own published evaluation
- `reap-cuda` fork: https://github.com/t-timms/reap-cuda, branch `fix/qwen3-5-router-renormalization`
