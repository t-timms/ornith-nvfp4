---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
library_name: transformers
tags:
- reap
- pruning
- expert-pruning
- nvfp4
- nvfp4a16
- fp4
- 4-bit precision
- compressed-tensors
- quantization
- vllm
- blackwell
- mixture-of-experts
- moe
- code
---

# Ornith-1.5-35B-A3B REAP-50 NVFP4A16 (16 GB)

**REAP expert-pruned (50%) + GPTQ-NVFP4A16 quantized build of `ornith-ai/Ornith-1.5-35B-A3B`, sized and served to run as a local coding model inside 16 GB of consumer VRAM** — 12.47 GiB, RTX 5070 Ti (SM120), vLLM. Text-only: the base model's genuinely-trained vision tower and its MTP draft head are deliberately removed (documented trade-offs below), which is what makes the text-only architecture (`Qwen3_5MoeForCausalLM`) servable by stock-class vLLM after two confirmed upstream gaps were patched locally (both independently fixed upstream since, in vLLM #50210).

**First quality signal — HumanEval+ 84.15% [77.8, 88.9], HumanEval 90.24% [84.7, 93.9], MBPP+ 89.15% [85.6, 91.9]**, greedy decoding, instruct framing, measured on exactly the released checkpoint. Agentic: **SWE-bench Verified 22/50 = 44.0% resolved** (81.5% resolved-of-completed), via `mini-swe-agent`'s official bash-only scaffold at a 49K-token context ceiling, graded by the official harness with zero infra failures — measured on the same 50-instance slice as the prior release's 52.0% figure. The run is context-limited (15/50 hit the ceiling before submitting); see the SWE-bench section below before citing the headline number without that context.

## Highlights

Result | Detail
--- | ---
**12.47 GiB** | REAP 50% expert pruning (256→128 experts) + GPTQ-NVFP4A16 (weight-only), vision tower and MTP head stripped
**84.15% [77.8, 88.9]** | HumanEval+, greedy, instruct framing, n=164 — first genuine accuracy measurement of this pipeline, on the released artifact
**89.15% [85.6, 91.9]** | MBPP+, same protocol, n=378
**44.0%** (22/50) | SWE-bench Verified, `mini-swe-agent` bash-only, 49K context — see caveats below
~19 B params (from 35.95 B) | Expert halving is what pays for the 16 GB budget; 8-of-128 expert routing preserved
MIT | Inherited from the base model

## Why 50 percent

Forced by arithmetic on a 16 GB card, not a tuning choice:

variant | size | fits 16 GB
--- | --- | ---
bf16 base (measured footprint) | ~65.4 GiB | no
NVFP4, unpruned (`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`) | ~21.8 GiB | no — before any KV cache
REAP 25% + NVFP4 | not built | KV-cache headroom at 25% was insufficient on the prior project's identical-budget build; not re-derived here
**REAP 50% + NVFP4A16** | **12.47 GiB** | **yes, with KV cache room**

Supporting evidence for the ratio: [*Half the Experts, All the Code*](https://arxiv.org/html/2607.16721) pruned Qwen3.6-35B-A3B — this base model's size-class cousin — at 50% with no statistically detectable loss on its primary code benchmark.

## Evaluation

HumanEval+ and MBPP+ via lm-eval-harness, greedy decoding, instruct framing, Wilson 95% confidence intervals, served through vLLM on the released checkpoint itself:

benchmark | score | 95% CI | n
--- | --- | --- | ---
HumanEval+ | **84.15%** | [77.8, 88.9] | 164
HumanEval | 90.24% | [84.7, 93.9] | 164
MBPP+ | 89.15% | [85.6, 91.9] | 378

The full 706-problem suite ran in ~12 minutes through vLLM's backend. For tier context, our previous release (KAT-Coder-V2.5-Dev REAP-50 NVFP4A16, a different and code-specialized base) scored 96.34 / 89.63 / 89.42 on the same three tasks: this build is lower across the board but in the same tier, consistent with more aggressive pruning (same 50%) applied to a general-purpose MoE base rather than a code-specialized one. No upstream HumanEval/MBPP numbers are published for `Ornith-1.5-35B-A3B` itself, so there is no published figure to compare against directly.

### SWE-bench Verified — read before citing the 44.0% figure alone

Measured 2026-08-26 via `mini-swe-agent`'s official bash-only scaffold (the prior release's protocol), graded by the official SWE-bench harness with zero infra or eval errors, on the released checkpoint itself:

| metric | value |
|---|---:|
| resolved | **22/50 = 44.0%** |
| resolved of completed (valid patch produced and tested) | 22/27 = **81.5%** |
| ContextWindowExceeded | 15 (49K ceiling) |
| LimitsExceeded (agent turns exhausted, step_limit 65) | 6 |
| RepeatedFormatError (no usable tool call) | 2 |
| ran, tests failed (genuinely unresolved) | 5 |

The 50 instances are the same shuffled slice the prior release's 52.0% figure comes from — a same-instance comparison under the same scaffold, window, and step limit. Prior release (KAT-Coder-V2.5-Dev REAP-50 NVFP4A16, code-specialized base): 26/50 resolved, 26/32 = 81.25% resolved-of-completed. This build: 22/50, 22/27 = 81.5% resolved-of-completed. The headline gap is entirely submission rate (33 vs 27 instances reached a real completion attempt), not patch quality once submitted — resolved-of-completed is indistinguishable. Where the missing submissions went: 15 ContextWindowExceeded (both builds fail this way — the 16 GB KV budget is the binding constraint, not the model), 6 LimitsExceeded (the prior release had none at step_limit 65; instances this base finishes less turn-efficiently), 2 RepeatedFormatError. 18 instances were resolved by both builds, 8 only by the prior release, 4 only by this one. Consistent with the code-benchmark tier picture above: same tier, slightly behind the code-specialized prior release.

**Config note.** Sampling followed the checkpoint's own `generation_config.json` (temperature 1.0 / top_p 0.95 / top_k 20) — the vendor-documented default, adopted as the starting point and flagged as unvalidated in `scripts/swebench/ornith_overrides_ladder.yaml`; the prior project's precedent is that this class of choice must survive a full pilot before being trusted, and no sampling sweep has been run here. Window: `max_model_len` 49,152 with a measured 54,067-token total KV budget (1.10x concurrency at workers=1; the ceiling varies 40–61K tokens with desktop VRAM contention, and the ladder harness falls back to smaller windows automatically). Read the headline as context-limited, not unconditional: 15 of 50 instances never submitted because they ran out of window. Raw artifacts (trajectories, `preds.json`, grading report) are preserved with the run; reproduction entry points: `scripts/swebench/run_ladder_night.sh`, `scripts/swebench/grade_pilot.sh`.

## Quantization and pruning details

Field | Value
--- | ---
Base model | `ornith-ai/Ornith-1.5-35B-A3B` (35.95 B params, MIT)
Pruning | REAP, expert-level, 50% compression ratio (256→128 experts), seed 42, single seed
Pruning residency | `--residency cpu_full` — validated deterministic path; a `layerwise` detour surfaced three real `reap-cuda` bugs and was reverted after the RAM math that motivated it was re-measured and found overly conservative (~65.4 GiB actual vs ~71.9 GiB estimated)
Pruning calibration | `theblackcat102/evol-codealpaca-v1`
Router renormalization | Fixed (upstream REAP adapter silently disables it for this architecture; fix carried from the prior release)
MTP head | Removed (`mtp_num_hidden_layers: 0`). Confirmed expert-routed — a full extra decoder layer with its own 256-expert MoE — meaning joint pruning was unvalidated; disabling it forfeits speculative-decoding speedups, documented as accepted trade-off
Vision tower | Removed. Verified genuinely trained (not phantom weights) by statistical comparison against known-init fingerprints; removal is a deliberate capability trade-off, not dead-weight stripping. Checkpoint switched to transformers' real text-only class (`Qwen3_5MoeForCausalLM`)
Quantization method | compressed-tensors / llm-compressor 0.13.0, GPTQ rounding (chosen over RTN from the start — the prior project's "GPTQ≈RTN" finding doesn't transfer to this expert topology)
Quantization scheme | NVFP4A16 — weight-only 4-bit float, group size 16, fp8_e4m3 scales, static actorder, tensor_group strategy; activations bf16
Quantization calibration | evol-codealpaca, 256 samples, 2048 sequence length, all experts calibrated
Kept unquantized | 291 tensors: routers, shared-expert gates, embeddings, linear-attention projections, lm_head
Files | Single `model.safetensors` (12.47 GiB) plus tokenizer/config
Serving validation | Two confirmed upstream gaps were required to serve this checkpoint and were patched locally; both have since been fixed upstream independently (vLLM #50210): a missing model registry entry for `Qwen3_5MoeForCausalLM`, and three `IsHybrid`-required `get_mamba_state_*` classmethods missing from `Qwen3_5ForCausalLMBase`. The pinned, locally-patched vLLM tree that produced the numbers below is referenced in the eval scripts. Separately, loading this checkpoint family through stock llm-compressor ≤0.13.0 silently drops every quantized expert weight (no `qwen3_5_moe` entry in `ARCH_TO_2D_MAPPINGS`) — if you load these weights outside vLLM, verify your loader keeps them; upstream fix tracked in llm-compressor PR #3080 (approved, unmerged as of 2026-08-25)
Built on | RTX 5070 Ti, 16 GB VRAM, SM120 (compute capability 12.0)

## Usage

Requires vLLM with SM120 support. The configuration below is the one actually exercised end-to-end during evaluation (via vLLM's engine, greedy):

```bash
vllm serve Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16 \
  --dtype bfloat16 --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-cudagraph-capture-size 8 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --mamba-block-size 528 --block-size 528
```

Notes, each earned the hard way:

- `--max-cudagraph-capture-size` below vLLM's default of 512 avoids a documented mamba-cache assertion for this architecture family; 8 is what the evaluated configuration used.
- `--mamba-block-size 528 --block-size 528` alignment is required for correct hybrid-cache sizing on this checkpoint.
- Prefix caching works, but is ineffective for prompts shorter than the 528-token mamba block boundary (open upstream issue vllm-project/vllm #40696) — expect no prefix-cache benefit on short early turns in agentic workloads.
- The code benchmarks ran with `max_model_len` 2048; the SWE-bench pilot served the same released checkpoint at 49,152 with a measured 54,067-token total KV budget — the measured serving ceiling on this 16 GB card (it varies 40–61K tokens with desktop VRAM contention).
- Greedy (`temperature=0`) is what the HumanEval+/MBPP+ numbers were measured with. The SWE-bench pilot instead used the checkpoint's own `generation_config.json` sampling (temperature 1.0 / top_p 0.95 / top_k 20) — the vendor-documented default, adopted as the starting point, not a validated choice; see the config note in the SWE-bench section.

## Known limitations

- **No pruning-ablation baseline measured.** The unpruned model does not fit this hardware; the accuracy cost of REAP itself (independent of quantization) is not isolated here. The comparison against the prior release above is tier context, not an ablation.
- **Text-only by construction.** The vision tower was trained and worked; this build cannot see images. Documented trade-off, reversible in principle by rebuilding from the unstripped variant.
- **No MTP / speculative decoding.** Forfeited by the MTP-strip decision above.
- **Context-limited agentic results.** The SWE-bench pilot served at a 49K window; 15/50 instances hit ContextWindowExceeded before submitting, so 44.0% is a floor under this hardware budget rather than an unconditional capability number. The ceiling itself moves (40–61K tokens total) with desktop VRAM contention on this card.
- These are self-reported numbers with published reproduction scripts, independently re-runnable from the eval suite in the companion repository; they are not leaderboard submissions.

## Prior art and scope of claims

Verified against the Hugging Face Hub on 2026-08-25:

- Unpruned NVFP4 of this base exists officially: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (~21.8 GiB).
- REAP-pruned Ornith builds did not exist on the Hub as of that date.

What is distinct, and all that is claimed: a **vLLM-servable Ornith that fits 16 GB with KV-cache room**, with published HumanEval+/MBPP+ and SWE-bench Verified numbers, Wilson intervals, and the exact serving configuration — none of which the unpruned quant above publishes. The upstream-gap fixes required to serve text-only Qwen3.5-MoE checkpoints have landed upstream (vLLM #50210), so current vLLM builds serve this architecture without local patching.

## License

MIT, inherited from the base model `ornith-ai/Ornith-1.5-35B-A3B`.

## Citation

This checkpoint is derived from `ornith-ai/Ornith-1.5-35B-A3B`. If you use it, please cite the base model:

```bibtex
@misc{ornith15_2026,
  title={Ornith-1.5-35B-A3B},
  author={{Ornith AI}},
  year={2026},
  url={https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B}
}
```

And the pruning method:

```bibtex
@misc{reap2025,
  title={{REAP} the Experts: Why Pruning Prevails for One-Shot {MoE} compression},
  author={Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
  year={2025},
  eprint={2510.13999},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.13999}
}
```

