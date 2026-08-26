# Roadmap

Status snapshot as of 2026-08-25. Not a changelog (see `CHANGELOG.md` for
what shipped) — this is where the project is headed and why, kept current.

## Done

- Exhaustive base-model comparison (Ornith-1.5-35B-A3B vs. full Qwen3-Coder
  family and a broader HF search) — see `docs/base_model_selection.md`.
- REAP-compatibility confirmed empirically on real Ornith weights (partial
  checkpoint slice, real `reap-cuda` pipeline, not a synthetic test) — same
  doc.
- Found and fixed a real bug in `reap-cuda`'s pairwise co-occurrence metric
  (no CUDA path for integer matmul) — pushed upstream to the fork,
  `t-timms/reap-cuda` commit `b67be1a`.
- **SOTA research pass across architecture/pruning/quantization/serving/sampling**,
  see `docs/optimization_research_2026-08-23.md` (sourced, 20+ citations). Key
  outputs folded into this roadmap below.
- **MTP-head pruning question resolved** (was the single biggest open item from
  the research pass): Ornith's `mtp_num_hidden_layers: 1` head is confirmed
  **expert-routed** (its own 256-expert MoE + router + shared expert + full
  self-attention block — a complete extra decoder layer, not a lightweight dense
  draft head), found by inspecting the live `config.json`/
  `model.safetensors.index.json` directly. Deeper finding: `text_config.num_experts`
  is a single config field shared between the backbone and the MTP block — pruning
  the backbone via the stock `reap-cuda` CLI without also handling the MTP head
  would very likely produce a checkpoint that **fails to reload** (router weight
  shape mismatch), not a silent degradation. Full writeup, code citations, and the
  decision: `docs/mtp_pruning_decision.md`. **Decision: disable the MTP head for
  the v1 build** (`mtp_num_hidden_layers: 0` post-prune, strip `mtp.*` tensors) —
  re-enabling it is a separate, later, explicitly-scoped experiment, not bundled
  into the base prune.
- **Quantization plan corrected**: originally planned RTN-by-default (below,
  reasoning was "GPTQ showed zero measurable improvement over RTN on the prior
  model's build") — that reasoning doesn't transfer to Ornith's different expert
  topology (256 vs. 128 experts) and there's no evidence-based reason to default
  to the weaker method now that llm-compressor's GPTQModifier is equally available
  and actively maintained (v0.9–0.10). **Corrected default: GPTQ-NVFP4 from the
  first run**, not an RTN-first-then-retrofit sequence. Script staged (not yet
  runnable — needs a real pruned checkpoint first):
  `scripts/quantize/quantize_ornith_gptq.py`. `linear_attn.*` ignore-list pattern
  verified against Ornith's actual tensor names (not assumed) — see that script's
  docstring.
- Confirmed `~/quant-env` (llm-compressor 0.13.0, same version validated for the
  prior project) is ready to use for Ornith — no new env needed.
- **Checkpoint download launched** (2026-08-23, user go-ahead given while GPU
  was occupied by gaming — network/disk only, no GPU used): full bf16 snapshot
  of `ornith-ai/Ornith-1.5-35B-A3B` to `~/models/Ornith-1.5-35B-A3B`, via
  `~/reap-cuda-env/bin/python ~/download_ornith.py` (resumable
  `snapshot_download`), log at `~/download_ornith.log`.
- **RAM headroom checked before launch (2026-08-23), then corrected
  (2026-08-24).** Original check: 74GiB available inside the 80GB
  `.wslconfig` cap vs. an estimated ~71.9GiB of bf16 weights (35.95B params
  × 2 bytes) left only ~2-6GiB margin under `--residency cpu_full`
  (KAT-Coder's validated path) — judged too tight, switched to `--residency
  layerwise` instead. That estimate turned out to be too conservative:
  `layerwise` surfaced three real `reap-cuda` bugs (see the "Full REAP
  prune" item below), and re-checking the RAM math against the real prune
  run's own logs showed an actual ~65.4GiB weight footprint against ~77GiB
  free RAM — a genuine ~11-12GiB margin. Reverted to `--residency cpu_full`
  (also resolves the reproducibility question this item originally flagged
  as unchecked for `layerwise`). Full reasoning and the correction:
  `docs/ram_headroom_check.md`.
- **Prune launch script staged and run to completion**:
  `scripts/prune/run_prune.sh` — single seed (42), 50% ratio, `--residency
  cpu_full` (see the corrected headroom finding above), points at
  `~/models/Ornith-1.5-35B-A3B`. Guards on the model directory existing
  before running (so it fails fast if the download isn't done yet rather
  than mid-run). GPU go-ahead given 2026-08-24; ran successfully in ~48
  minutes — see the "Full REAP prune" item below for the validation work
  that preceded launch and the resulting checkpoint.
- **MTP-strip script staged**: `scripts/prune/strip_mtp.py` — sets
  `mtp_num_hidden_layers: 0`, drops `mtp.*` tensors from the safetensors
  shards + index, reload-verifies with a real forward pass on CPU. Reviewed
  against the base checkpoint's actual tensor names, but **untested against a
  real pruned checkpoint** (none exists yet) — see the script's own docstring
  for what's verified vs. assumed.
- Env readiness confirmed for the prune step: `~/reap-cuda-env` has `reap`
  0.1.0 installed editable from `~/reap-cuda` at commit `b67be1a` (the
  router-renorm fix, correct fork/commit), torch 2.13.0+cu130.

## Next (in order)

1. ~~Resolve the vision-tower question.~~ **Done** — confirmed genuinely
   trained, not phantom, by comparing `model.visual.*` weight statistics
   (norm-weight drift from init, non-zero linear biases) against the
   known-trained text backbone. Full method and stats:
   `docs/vision_tower_decision.md`. Removal in step 6 below is a deliberate
   scope trade-off, not a free strip of dead weight.
2. ~~Checkpoint download.~~ **Done** — 67GB on disk at
   `~/models/Ornith-1.5-35B-A3B`, verified complete (33/33 files).
3. ~~Full REAP prune at 50%~~ **Done** (2026-08-24, 256→128 experts).
   GPU go-ahead given 2026-08-24; small-scale invocation validation (not
   the real run) caught three real `reap-cuda` bugs under `--residency
   layerwise` before any GPU-hours were spent — two fixed and verified
   end-to-end (attention-mask format reused across incompatible layer
   types; MoE expert weights read after `accelerate` re-offloads them to
   meta), one found but not fixed (same class of bug in the pruning/
   mutation step). Residency mode has no effect on the pruned model's
   quality — it's purely how weights are staged in memory — so rather than
   keep debugging a code path this project doesn't need, reran the
   RAM-headroom estimate that had originally ruled out `--residency
   cpu_full` and found it was too conservative (measured ~65.4GiB weight
   footprint, not the ~71.9GB estimate — comfortable margin against
   ~77GiB free RAM). Ran to completion under `--residency cpu_full` in
   ~48 minutes with zero errors. **Output**: `~/reap-stability/
   ornith_n64_s42/model_.../pruned_models/reap-renorm_true-seed_42-0.50`,
   36GB, `num_experts: 128` confirmed, `mtp_num_hidden_layers: 1`
   confirmed untouched (as intended — step 4 handles it). Full incident
   writeup: `docs/layerwise_prune_run_2026-08-24.md`; corrected RAM math:
   `docs/ram_headroom_check.md`'s 2026-08-24 addendum.
4. ~~MTP-head checkpoint surgery~~ **Done** (2026-08-24):
   `scripts/prune/strip_mtp.py`. Two assumptions written before any real
   checkpoint existed turned out wrong and were fixed against the real
   thing: the checkpoint layout is a single ~35GiB `model.safetensors`
   with no index (the script assumed sharding); and the MTP head's
   weights were already entirely absent from the pruned output (not
   "untouched but shape-mismatched" as `docs/mtp_pruning_decision.md`
   originally predicted — see that doc's 2026-08-24 addendum). Config
   fix (`mtp_num_hidden_layers: 0`) still applied and necessary either
   way. Reload-verified with a real CPU forward pass — confirmed
   independently, not just trusted from the script's exit code:
   `logits shape (1, 4, 248320)` matches the model's actual `vocab_size`.
   Output: `<prune output dir>-mtp-stripped`, 36GB, 1026 tensors, zero
   `mtp.*` keys remaining.
5. ~~Quantize to NVFP4A16 via GPTQ~~ **Done** (2026-08-24, ~5h37m wall
   clock per the run log). The script fails fast if step 4 wasn't done
   (checks `mtp_num_hidden_layers == 0` before spending calibration time).
   **Initially believed corrupted** — a direct sanity check produced
   incoherent garbage on a trivial prompt, byte-for-byte-identical garbage
   on the quantized checkpoint and its vision-stripped derivative, versus
   coherent output on the pre-quantization bf16 checkpoint. **Root cause
   found 2026-08-25: NOT corruption.** llm-compressor 0.13.0 has a
   confirmed, named upstream gap (vllm-project/llm-compressor PR #3080,
   open/unmerged, explicitly cites ornith-ai/Ornith-1.0-35B as a real
   checkpoint hitting this) — qwen3_5_moe/qwen3_5_moe_text are registered
   for GPTQ calibration but have no entry in ARCH_TO_2D_MAPPINGS, so
   from_pretrained silently drops every quantized expert weight as
   unexpected, leaving the MoE layers at random init. The GPTQ math itself
   was correct the whole time. Fixed locally with a monkeypatch
   (`scripts/patches/patch_qwen3_5_moe_2d_load.py`, applied automatically
   by `scripts/eval/run_lm_eval_patched.py`) — confirmed working: patched
   load reproduces the bf16 checkpoint's coherent output. No
   re-quantization needed. Output: `~/models/ornith-nvfp4a16-gptq`, 14GB,
   from the mtp-stripped 36GB checkpoint. The bounded-probe discipline that
   preceded launch (mechanism verification + ETA measurement) is preserved
   below because it's what made the ~5.6h window predictable:
   an 8-sample run confirmed GPTQ's `moe_calibrate_all_experts=True`
   correctly reaches the fused MoE expert weights (a real open question —
   `targets="Linear"` normally matches `nn.Linear` module instances by
   type, and Ornith's experts are raw fused parameters on a custom
   container, not `nn.Linear` submodules; log confirmed per-expert
   quantization with zero errors across 5 layers). A second probe at real
   settings (256 samples, 2048 seq len) measured ~3.27s/expert
   (1.45s+1.45s+0.37s for gate/up/down), essentially unchanged from the
   8-sample measurement — the cost is dominated by the fixed per-matrix
   GPTQ solve, not calibration volume. 40 layers × 128 experts = 5,120
   experts × 3.27s ≈ **4.65 hours for the expert MLPs, ~4.75-5 hours
   total** — the real run came in at ~5h37m wall clock. Both probes were
   killed once they'd measured what they needed to.
6. ~~Strip vision tower~~ **Done** (2026-08-24), against the quantized
   checkpoint: `scripts/prune/strip_vision.py`. A naive tensor-delete would
   NOT have worked — `Qwen3_5MoeForConditionalGeneration`’s wrapper
   unconditionally builds `self.visual = AutoModel.from_config(config.
   vision_config)` in `__init__` with no guard, so deleting the weights
   while keeping that class/config shape would either crash on load or
   silently rebuild a full random-init vision tower that still eats VRAM.
   Switched to `Qwen3_5MoeForCausalLM`, a real text-only class already in
   this transformers version: promoted the checkpoint’s already-complete
   `text_config` dict to the top-level config, renamed every
   `model.language_model.*` tensor key to `model.*` (that class expects
   `model.embed_tokens.weight`/`model.layers.*`, not the language_model-
   nested naming — `lm_head.weight` was already unprefixed, no change
   needed there), dropped all 333 `model.visual.*` tensors, and filtered
   `quantization_config.ignore` (401 → 291 entries, dropping the
   now-nonexistent visual module names, renaming the rest). Reload-verified
   with a real CPU forward pass: `logits shape (1, 4, 248320)` matches
   `vocab_size`. **Output**: `~/models/ornith-nvfp4a16-gptq-text-only`,
   13GB (down from the 14GB vision-inclusive quantized checkpoint), this is
   the release-candidate checkpoint going into step 7.
7. ~~HumanEval+/MBPP+ accuracy suite.~~ **Done (2026-08-25)** — first real
   quality signal on the whole pipeline. Served via locally-patched vLLM
   (registry gap + missing `Qwen3_5ForCausalLMBase` mamba classmethods
   fixed in `~/vllm-src`, both confirmed real upstream gaps, not
   project-specific bugs; checkpoint served as the mrope-stripped
   config-only variant since text-only serving does not need them).
   humaneval_instruct **90.24%** (148/164), humaneval_plus_instruct
   **84.15%** (138/164 — the headline number per this project's
   convention), mbpp_plus_instruct **89.15%** (337/378), greedy, Wilson
   stderr 2.32/2.86/1.60pp respectively. Full 706-problem suite ran in
   about 12 minutes via vLLM (vs. an extrapolated 40+ hours on the hf
   backend before the vLLM fixes). For comparison, kat-coder-16gb scored
   96.34%/89.63%/89.42% on the same three tasks — Ornith is lower across
   the board but in the same tier, reasonable given more aggressive (50%)
   REAP pruning on a different base model. Raw results:
   `~/eval-suite-ornith-vllm/`.
8. ~~SWE-bench validation ladder~~ **Done (2026-08-26)** — single-instance
   smoke → small bounded sample → full 50-instance pilot. Same promotion discipline as the prior
   project throughout — no step skipped, no full-pilot claim made on
   single-instance evidence. Before first launch, check
   `docs/serving_notes.md` for known-before-launch vLLM flags
   (`--max-cudagraph-capture-size` mamba-cache assertion) rather than
   debugging them cold.

   **2026-08-26 night, rung progress.** Smoke instance ran end-to-end:
   server stable, qwen3_xml tool parsing confirmed viable through the real
   litellm path (preflight), trajectory saved, clean teardown. Two config
   facts measured, both now recorded here instead of guessed: hybrid-KV
   ceiling on this card is ~40-61K tokens total depending on desktop VRAM
   contention (1.23-1.85x concurrency at a 32K window), so workers are
   serialized to protect prefix caching; and prefix caching WORKS in align
   mode - 93.9% hit rate over the smoke rollout (307,296/327,137), closing
   the serving_notes open question with evidence. The smoke instance
   exhausted step_limit 40 while progressing sanely (no loop), matching the
   prior project's reference-scaffold data (~87 calls/instance); the ladder
   config raises it to 65 on that evidence. Overnight chained run launched
   (scripts/swebench/run_ladder_night.sh): 1-instance submission check ->
   gated -> 10-instance bounded sample -> gated -> 50-instance pilot,
   gates on patch mechanics only, scoring deferred to human review.

   **2026-08-26 completion.** The chained overnight run relaunched clean
   after the gaming shutdown and finished all rungs: Gate A 1/1 Submitted;
   Gate B 10/10 instances, 8/10 nonempty; pilot 50/50 instances, 27/50
   nonempty (27 Submitted / 15 ContextWindowExceeded / 6 LimitsExceeded /
   2 RepeatedFormatError), 96.5% prefix-cache hit rate, served at the
   49,152 window (54,067-token KV budget). Graded with the official
   harness (`scripts/swebench/grade_pilot.sh`): **22/50 = 44.0% resolved,
   22/27 = 81.5% resolved-of-completed, zero infra/error instances** — on
   the same 50-instance slice as the prior project's 52.0%, so directly
   comparable: the gap is submission rate (33 vs 27), not resolve quality
   (81.25% vs 81.5% resolved-of-completed). Card updated with the
   SWE-bench section, ceiling, and sampling disclosure; re-uploaded to
   the Hub.

## Longer-horizon / not scheduled

- **MTP-1 speculative decoding** (`--speculative-config '{"method": "mtp", ...}'`)
  — real, vLLM-documented, architecturally confirmed for Ornith, but not usable
  against the v1 checkpoint (MTP disabled per the decision above). Follow-up
  experiment once the base build is validated: does the *original* unpruned MTP
  head work against the *pruned* backbone at all, before asking whether it needs
  pruning too. See `docs/mtp_pruning_decision.md` SS3.
- Re-verify whether the KAT-Coder FLASH_ATTN-exclusion finding transfers to
  Ornith's own vLLM server startup log rather than assuming it does — see
  `docs/serving_notes.md`.
- Explicit non-recommendation carried from the research pass: do not layer
  SSM/GDN-width pruning (Minitron-SSM/Mamba-Shedder-style) on top of REAP's
  expert pruning — no validated recipe exists for combining both axes.
- A from-scratch comparison against `kat-coder-nvfp4`'s published numbers,
  once this build has its own validated SWE-bench score.
