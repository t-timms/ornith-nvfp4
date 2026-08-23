# Ornith-1.5-35B-A3B: SOTA Optimization Research (2026-08-23)

Sourced literature/ecosystem review across architecture, pruning, quantization, serving,
and sampling. Findings are labeled **[SOURCED]** (verified against a primary doc/repo/paper),
**[CITED, unverified against our checkpoint]** (a public claim not yet checked against our
own artifacts), or **[UNVERIFIED — needs GPU time]** (no empirical test exists yet, ours or
anyone else's, for this exact combination). Nothing here is a benchmark number we measured —
this doc is planning input for the full-scale REAP prune, not a results writeup.

## 1. Architecture specifics

**[SOURCED]**, from `ornith-ai/Ornith-1.5-35B-A3B`'s live `config.json`:

- `num_hidden_layers: 40`, hidden_size 2048
- `layer_types`: repeating 3:1 pattern — 3x `linear_attention` (Gated DeltaNet) then
  1x `full_attention`, across all 40 layers → **30 GDN layers / 10 full-attention layers**
  (confirms the "30 of 40" figure from secondary sources and the general Qwen3.5-family 3:1
  ratio; this is architecturally identical in shape to upstream Qwen3.5-MoE, consistent with
  our own `reap-cuda` finding that Ornith instantiates `Qwen3_5MoeSparseMoeBlock` via stock
  `transformers`).
- Gated DeltaNet layers: `linear_key_head_dim: 128`, `linear_num_key_heads: 16`,
  `linear_value_head_dim: 128`, `linear_num_value_heads: 32`.
- Full-attention layers: `head_dim: 256`, `num_attention_heads: 16`, `num_key_value_heads: 2`
  (GQA).
- MoE: `num_experts: 256`, `num_experts_per_tok: 8`, `moe_intermediate_size: 512`,
  `shared_expert_intermediate_size: 512`.
- **`mtp_num_hidden_layers: 1`** — Ornith ships a native Multi-Token-Prediction draft head,
  same mechanism as Qwen3.5/Qwen3.6/DeepSeek-V3-style MTP (predicts token *t+2* from the
  backbone's hidden state at *t*, trained jointly with the main model). This was **not
  previously flagged** in project memory and is a real, free-standing optimization lever —
  see §4.
- Context: 262,144 native, YaRN-extendable to ~1M (official guide's own caveat: YaRN applies
  one fixed scale factor per request, "can slightly hurt quality on ordinary-length inputs" —
  don't enable it by default for the SWE-bench harness, which runs well under native context).
- Runtime floors per the official card: `transformers >= 5.8.1`, `vllm >= 0.19.1`,
  `sglang >= 0.5.9`.

Sources: [Ornith-1.5-35B-A3B HF card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
config.json (live fetch), [Qwen3.5 GDN architecture analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15),
[MTP-as-spec-decode-draft writeup](https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/).

**Practical implication**: our `reap-cuda` fork's expert-slicing logic (256→N experts) needs
to explicitly account for the MTP head as a *separate* small transformer block, not a copy of
layer 40 — confirm whether it shares the router/expert weights with the backbone or has its
own independent (typically dense, non-MoE) FFN before pruning.

**[RESOLVED 2026-08-23]** — checked directly against the live `config.json`/
`model.safetensors.index.json`: the MTP head is **expert-routed** (own 256-expert MoE, router,
shared expert, and full self-attention block — a complete extra decoder layer, not a copy of a
GDN layer). Deeper finding: `text_config.num_experts` is a single config field shared with the
backbone, so pruning the backbone via the stock CLI without also handling the MTP head risks a
checkpoint that fails to reload (router weight shape mismatch), not just a suboptimal one. Full
writeup, code citations, and the resulting decision (disable MTP for v1, revisit later as its
own experiment): `docs/mtp_pruning_decision.md`.

## 2. Pruning SOTA

**[SOURCED]** REAP (Cerebras + U. Calgary, arXiv 2510.13999, accepted ICLR 2026) remains the
current SOTA one-shot MoE pruning criterion — router-gate-value-weighted expert-activation-norm
scoring, no fine-tuning required, near-lossless at 50% compression on code-gen/tool-calling
benchmarks for 20B–1T-param SMoE models (Qwen3-Coder-480B, Kimi-K2 cited specifically).
No newer method supersedes it as of this search; a companion paper
([2606.15716](https://arxiv.org/pdf/2606.15716), "How to Score Experts for One-Shot MoE
Pruning") generalizes REAP's scoring into a unified framework but doesn't claim to beat it
outright.

**[SOURCED]** Checked `CerebrasResearch/reap`'s upstream README directly: its `MODEL_ATTRS`
list covers Qwen3-30B-A3B/Coder-480B/Coder-30B, GLM-4.5-Air/4.6, Mixtral-8x7B, Llama-4-Scout,
ERNIE-4.5-21B-A3B, Kimi-K2/Kimi-Linear, MiniMax-M2, DeepSeek-V3.2 — **`qwen3_5_moe`, Qwen3.5,
and Gated DeltaNet are not mentioned anywhere.** This confirms (independent of our own earlier
empirical finding) that our project's `reap-cuda` fork's Qwen3.5-hybrid support and the
router-renormalization fix are genuinely upstream-absent work, not a duplicated effort — worth
stating plainly in the Ornith repo's docs and worth upstreaming as a PR once validated at scale.

**[CITED, relevant precedent]** Literature specific to pruning *hybrid* SSM/linear-attention +
MoE architectures (as opposed to pure-MoE): `Minitron-SSM` (arXiv 2504.11409, "Group-Aware SSM
Pruning") and `Mamba-Shedder` (arXiv 2501.17088) both target pruning the recurrent/SSM layers
themselves (width/group pruning within Mamba blocks), not expert pruning — a **different,
complementary axis** to REAP. `SparseSSM` (2506.09613) does one-shot SSM pruning without
retraining, same spirit as REAP but for the recurrent layers. `Nemotron-Labs-3-Puzzle-75B-A9B`
(2607.04371) is the closest direct precedent — explicitly compresses a *hybrid MoE LLM*
(their own terminology) and is worth reading in full before the real prune run, since it's the
one paper found that addresses both axes (expert count + hybrid-layer structure) together.
**[UNVERIFIED]**: none of these give a validated recipe for combining REAP (expert-axis) with
any SSM/GDN-layer-axis pruning on this exact architecture — recommend staying expert-axis-only
for this build (matches the KAT-Coder precedent and keeps the change auditable/reproducible)
rather than layering in an unvalidated second compression axis.

Sources: [REAP paper](https://arxiv.org/abs/2510.13999), [REAP GitHub](https://github.com/CerebrasResearch/reap),
[Cerebras blog](https://www.cerebras.ai/blog/reap), [Minitron-SSM](https://arxiv.org/pdf/2504.11409),
[Mamba-Shedder](https://arxiv.org/pdf/2501.17088), [SparseSSM](https://arxiv.org/pdf/2506.09613),
[Nemotron-Labs-3-Puzzle](https://arxiv.org/pdf/2607.04371).

## 3. Quantization SOTA

**[SOURCED]** MR-GPTQ (arXiv 2509.23202, "Bridging the Gap Between Promise and Performance for
Microscaling FP4 Quantization," ICLR 2026) — block-wise Hadamard rotation + format-specific
GPTQ tailored to FP4's small group size, claims up to 96.1% FP16-accuracy recovery, and
specifically addresses why standard outlier mitigation fails on NVFP4's narrow group size.
**Its llm-compressor RFC (issue #2006) is still open as of this search** — assigned, labeled
`enhancement, keep-open`, **no implementation PR yet**, though the corresponding *inference*
side (vLLM kernel support, PR #24440) is already merged. So: the serving-side kernel exists,
but there is still no llm-compressor recipe to actually *produce* an MR-GPTQ checkpoint —
same conclusion as our prior check, re-confirmed, not stale.

**[SOURCED]** llm-compressor's plain GPTQModifier now supports both MXFP4 and NVFP4 output
schemes directly (v0.9–0.10 changelog), plus a real Hessian-numerics fix (+4% GSM8K on
Llama-3-8B cited) and distributed multi-GPU GPTQ (up to 3.8x on 4 GPUs — irrelevant to our
single-GPU box, but confirms the codepath is actively maintained). This is a strictly better
starting point than the plain-RTN NVFP4 path we shipped for KAT-Coder, and unlike MR-GPTQ it's
actually usable today.

**[CITED]** NVFP4 vs MXFP4: NVFP4 is the better-recovering of the two on Blackwell
(native E2M1 microscale, SM100/SM120-accelerated); a real large-MoE precedent exists —
`MiniMax-M2.7-NVFP4` (230B total/10B active, 256 experts) quantizes **only the MoE expert MLP
layers** to NVFP4, leaving attention/router/norms at higher precision — directly analogous to
the AWQ finding below and to our own KAT-Coder ignore-list pattern. Worth adopting the same
"quantize FFN experts, protect everything structurally load-bearing" default for Ornith.

**[SOURCED — directly answers the GDN-quantization question]** Public AWQ recipes for
Qwen3.5-family hybrid models (cross-checked against a published Qwen3.5-27B-AWQ config) keep
these Gated-DeltaNet-specific tensors at higher precision rather than quantizing them:
`linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba`, `linear_attn.out_proj`, `linear_attn.norm`,
`linear_attn.conv1d`, **`linear_attn.A_log`**, **`linear_attn.dt_bias`** — i.e., the decay-gate
and delta-rule state parameters specifically, not just "attention in general." This matches the
generic SSM-quantization sensitivity literature (`QS4D`, arXiv 2507.06079): the state-transition
analog is the most quantization-sensitive component, projections tolerate it much better, gating
is intermediate. **Caveat**: QS4D's numbers are for QAT on classic Mamba-style SSMs (A/B/C
matrices), not Gated DeltaNet's delta-rule formulation specifically, and not PTQ — treat the
*qualitative* conclusion (protect the gate/decay parameters) as solid, the *quantitative*
bit-width thresholds as **[UNVERIFIED]** for our architecture.

**Recommendation**: extend our KAT-Coder-derived NVFP4 ignore-list with the seven
`linear_attn.*` tensor patterns above (A_log and dt_bias are the two that matter most if
disk/VRAM budget forces a smaller ignore-list) before running calibration — this is a direct,
concrete, low-risk change validated across two independent sources (AWQ recipe + SSM-sensitivity
literature), not a guess.

Sources: [MR-GPTQ RFC #2006](https://github.com/vllm-project/llm-compressor/issues/2006),
[MR-GPTQ paper](https://arxiv.org/abs/2509.23202), [LLM Compressor v0.10 (Red Hat)](https://developers.redhat.com/articles/2026/03/18/llm-compressor-010-faster-compression-distributed-gptq),
[LLM Compressor v0.9 (Red Hat)](https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more),
[NVFP4 production inference overview](https://arunksingh16.medium.com/nvidia-nvfp4-quantization-blackwell-and-the-path-to-production-inference-12407e14e084),
[MiniMax-M2.7-NVFP4](https://huggingface.co/NinjaBoffin/MiniMax-M2.7-NVFP4),
[Qwen3.5-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ),
[QS4D](https://arxiv.org/pdf/2507.06079).

## 4. Serving / kernel optimization SOTA

**[SOURCED]** vLLM added native Qwen3.5-hybrid support at v0.17: Triton kernels ported from
Flash Linear Attention for the GDN layers, plus **a hybrid KV-cache manager that auto-tunes the
"logical" block size so full-attention and linear-attention layers occupy equal physical memory
per block** — this is the mechanism, not just a marketing claim, per the vLLM Qwen3-Next launch
post (same cache-manager design carried into Qwen3.5). Our runtime floor (`vllm >= 0.19.1`) is
comfortably past this.

**[SOURCED]** From vLLM's own Qwen3.5 serving recipe page: if you hit a **mamba-cache assertion
error**, the fix is lowering `--max-cudagraph-capture-size` below its 512 default — a concrete,
actionable flag to know about *before* first launch, not to debug reactively. Prefix caching in
"align" mode for the Mamba/GDN cache is still marked experimental upstream — worth testing
in isolation before relying on it for the SWE-bench harness's repeated-instance workload.

**[SOURCED — new, actionable]** vLLM ships MTP speculative decoding for Qwen3.5-family models
via `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`, using exactly the
`mtp_num_hidden_layers: 1` head confirmed in §1. Guidance from vLLM's own docs: MTP-1 helps
**low-concurrency, latency-sensitive** serving (lower TPOT, high acceptance rate) but *reduces*
throughput under load — our SWE-bench harness runs 2 concurrent rollout workers (per the
KAT-Coder `--workers 2` convention), which is low enough concurrency that MTP-1 is plausibly a
real win worth testing, not just a curiosity. **[UNVERIFIED]** — no public number for
GDN-hybrid + MTP acceptance rate on agentic/code workloads specifically; this needs an actual
A/B once the pruned+quantized checkpoint exists, same promotion discipline as every other lever
this project has tested (single-instance smoke test → repeated draws → full pilot, never a
default swap on n=1).

**[UNVERIFIED, flagging the open question rather than guessing]**: whether REAP-pruning the
expert count (256→fewer) requires re-deriving the MTP head's weights (since MTP is trained
jointly with the backbone and may reference specific expert indices through its own routing, if
it has one) is not answered by any source found — this is the single biggest unresolved
architecture question before the real prune run and should be checked directly against the
Ornith checkpoint's weight names/shapes, not assumed either way.

**Backend selection**: no source found states explicitly whether vLLM excludes FLASH_ATTN from
the candidate backend list for Qwen3.5-hybrid the same way it does for KAT-Coder's
Gated-DeltaNet architecture (documented in project memory as a live production-log finding, not
a general claim) — treat that KAT-Coder finding as **architecture-family-level evidence, likely
to transfer** (same GDN mechanism, same vLLM codepath) but re-verify via the actual candidate
backend list in Ornith's own server startup log rather than assuming byte-for-byte transfer.

Sources: [vLLM Qwen3-Next launch (cache manager)](https://vllm.ai/blog/2025-09-11-qwen3-next),
[vLLM Qwen3.5/3.6 recipe page](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html),
[vLLM MTP docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/),
[MTP-as-draft-head analysis](https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/).

## 5. Sampling / chat-template

**[SOURCED]**, from the live Ornith model card: general-purpose recommended sampling is
`temperature=0.6, top_p=0.95, top_k=20`; benchmark-reproduction runs used `temperature=1.0`.
**No `presence_penalty` or `repetition_penalty` value is published** — a real gap compared to
Kwaipilot/KAT-Coder-V2.5-Dev's card, which explicitly recommends `presence_penalty=1.5,
top_k=20` for Thinking mode (the fix that measurably suppressed literal-repetition loops on
KAT-Coder). Tool calls render as `<tool_call>` XML, reasoning as `<think>` blocks, parsed into
`reasoning_content`/`tool_calls` — same qwen3-family convention already handled by our existing
`--tool-call-parser qwen3_xml --reasoning-parser qwen3` flags, no new parser work expected.

**[UNVERIFIED]**: no published guidance, official or third-party, on agentic-coding-specific
looping/repetition behavior for Ornith specifically — this project's own three-strikes-and-out
finding on KAT-Coder (sampling params, prompt engineering, AND chat-template swaps all failed to
fix stuck-loop behavior, reading as a capability-ceiling issue rather than a config problem) is
the closest available evidence, and it argues *against* assuming a sampling-param tweak will be
the lever here either — if Ornith shows the same failure mode, this project's own prior result
predicts config-level fixes won't solve it, and that's worth testing early (single-instance) to
avoid repeating three already-falsified interventions from the KAT-Coder track.

Source: [Ornith-1.5-35B-A3B HF card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## Summary: priority-ordered action list before the real prune run

1. **[Do first, cheap, high-value]** Inspect the MTP head's weight names/shapes in the actual
   Ornith checkpoint to determine whether it's dense or expert-routed, before writing the prune
   script's exclusion list (§1/§4 — biggest open architecture question, currently unverified
   anywhere).
2. **[Do before quantization]** Extend the NVFP4 ignore-list with `linear_attn.A_log` and
   `linear_attn.dt_bias` at minimum, ideally all seven `linear_attn.*` patterns from the AWQ
   precedent (§3) — concrete, dual-sourced, low-risk.
3. **[Plan, don't skip]** Prefer GPTQ-NVFP4 (llm-compressor's current GPTQModifier, actively
   maintained) over plain RTN for the Ornith build from the start, rather than repeating
   KAT-Coder's RTN-first-then-GPTQ-retrofit sequence — the accuracy-neutral GPTQ result on
   KAT-Coder doesn't guarantee the same on Ornith's different expert topology, but there's no
   reason to default to the weaker method now that the stronger one is equally available.
   MR-GPTQ stays a documented future-watch item only (§3) — still unimplemented upstream.
4. **[Test once a checkpoint exists, cheap]** MTP-1 speculative decoding — real, sourced,
   architecturally-confirmed lever with no KAT-Coder precedent to compare against (KAT-Coder has
   no MTP head). Standard promotion discipline applies: smoke test → repeated draws → decide.
5. **[Verify, don't assume]** Re-check the FLASH_ATTN-exclusion and mamba-cache-assertion /
   `--max-cudagraph-capture-size` behavior against Ornith's own server startup log before the
   first real launch — both are architecture-family-level findings with a plausible-but-unproven
   transfer from KAT-Coder or from vLLM's own docs.
6. **[Explicit non-recommendation]** Do not layer SSM/GDN-layer-width pruning
   (Minitron-SSM/Mamba-Shedder-style) on top of REAP's expert pruning for this build — no
   validated recipe exists for combining both axes, and it would break this project's
   single-variable-at-a-time discipline for no evidenced gain (§2).
7. **[Low priority, informational]** No published presence_penalty/repetition guidance exists
   for Ornith; don't assume a sampling fix will solve agentic looping if it appears — KAT-Coder's
   own three failed interventions are the best available prior (§5).

## Audit addendum (2026-08-23, later pass — SOTA re-check before GPU go-ahead)

Skeptical re-audit of the above against fresh WebSearch, specifically hunting for anything
newer/better rather than confirming prior conclusions. Verdicts below.

**§2 Pruning — one real gap found, checked, correctly not applicable.** Missed the first pass:
`ZEDA` (arXiv 2605.18643, "Post-Trained MoE Can Skip Half Experts via Self-Distillation",
[GitHub](https://github.com/TsinghuaC3I/ZEDA)) beats the strongest *dynamic*-MoE baseline by
4-6pp on Qwen3-30B-A3B/GLM-4.7-Flash at ~50% expert-FLOP reduction. **Does not apply to this
project's actual constraint**: ZEDA is a *dynamic* token-level skip mechanism (self-distillation
training required, not one-shot) — it cuts compute, not the on-disk/VRAM footprint, since every
expert must stay resident for the router to conditionally skip it. This project's binding
constraint is fitting a 16GB card, not FLOPs, so REAP's static permanent-removal approach remains
correct. Documented here so this comparison doesn't need re-deriving later.

**§2 Pruning tooling — found an independent third-party CUDA REAP port, not adopted, worth
knowing about.** [`egesabanci/reap-cuda`](https://github.com/egesabanci/reap-cuda) is a separate,
actively-maintained fork of `CerebrasResearch/reap` (unrelated to our `t-timms/reap-cuda`) that
already claims Qwen3.5/3.6 support including the shared-expert architecture and the same
`auto/gpu_full/layerwise/cpu_full` residency modes our `run_prune.sh` uses. **Could not verify
whether it independently implements the router-renormalization fix** our own fork required
(no GDN/renormalization-specific code was found in the time available) — not a reason to switch
mid-project given our fork is already validated end-to-end via the smoke test, but worth a future
cross-check (does it produce the same pruned output on a small model?) as an independent
correctness signal if this work is ever upstreamed or disputed.

**§3 Quantization — MR-GPTQ RFC #2006 re-confirmed still open/unimplemented** (llm-compressor's
own Q1-2026 roadmap issue #2262 lists it as still-planned, not shipped). No change from the
original research pass — re-verified, not stale.

**§3 Calibration dataset — confirmed reasonable, no revision.** `evol-codealpaca-v1` is the same
family CodeLlama/DeepSeek-Coder-style GPTQ recipes use for code-domain calibration; no evidence
found of a meaningfully better alternative for this use case.

**§4 Serving — one new, real, currently-open issue found.** vLLM
[issue #40696](https://github.com/vllm-project/vllm/issues/40696) (open, affects our documented
floor v0.19.1): prefix caching is **completely ineffective, not just suboptimal**, for any prompt
under the 528-token Mamba-aligned block size on Qwen3.5-class hybrid models — a 479-token prompt
gets 0% cache hits purely from crossing the block boundary, confirmed performance-only (not a
correctness/corruption bug, verified directly from the issue thread, not just a search snippet).
Relevant to the SWE-bench harness: early turns in a trajectory are more likely to be short.
Logged in `serving_notes.md`, not actionable as a fix (no vLLM-side workaround exists yet beyond
awareness), just something to expect rather than debug cold.

**Everything else in this document holds up** — no other library version, RFC status, or vLLM
behavioral claim found to be stale in this pass.
