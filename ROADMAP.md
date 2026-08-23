# Roadmap

Status snapshot as of 2026-08-23. Not a changelog (see `CHANGELOG.md` for
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
- **RAM headroom checked before launch — real risk found, mitigated.** 74GiB
  available inside the 80GB `.wslconfig` cap vs. ~71.9GiB of bf16 weights
  alone (35.95B params) leaves only ~2-6GiB margin under `--residency
  cpu_full` (KAT-Coder's validated path) — too tight to launch as-is. Switched
  the launch script to `--residency layerwise` (block-wise observe + disk
  offload, not a full CPU pin) instead. Full reasoning, including the
  **unverified tradeoff** (layerwise's determinism vs. cpu_full's validated
  reproducibility has not been checked) in `docs/ram_headroom_check.md`.
- **Prune launch script staged**: `scripts/prune/run_prune.sh` — single seed
  (42), 50% ratio, `--residency layerwise` per the headroom finding above,
  points at `~/models/Ornith-1.5-35B-A3B`. Guards on the model directory
  existing before running (so it fails fast if the download isn't done yet
  rather than mid-run). **Not yet executed** — needs the download to finish
  and GPU go-ahead.
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

1. **Resolve the vision-tower question** before assuming it's free to
   strip. Check whether Ornith's vision weights show signs of real training
   (the published `mmproj-*.gguf` file suggests they might, unlike the
   prior model's confirmed-phantom tower) before treating removal as a
   zero-cost step.
2. **Checkpoint download** — in progress (see above), check
   `~/download_ornith.log` for `DOWNLOAD_COMPLETE`.
3. **Full REAP prune at 50%** (256→128 experts): `bash
   scripts/prune/run_prune.sh`. Staged and RAM-checked (see above). **Blocked
   on step 2 finishing and on explicit user GPU go-ahead** (multi-hour GPU
   run, matches this project's standing practice for expensive operations).
4. **MTP-head checkpoint surgery**: `python3 scripts/prune/strip_mtp.py
   --checkpoint <run_prune.sh's output dir> --output-dir <output>-mtp-stripped`
   (script staged, see above; `run_prune.sh` prints the exact invocation with
   real paths when it finishes).
5. **Quantize to NVFP4A16 via GPTQ** (`quantize_ornith_gptq.py`, corrected
   default — see above). The script fails fast if step 4 wasn't done
   (checks `mtp_num_hidden_layers == 0` before spending calibration time).
6. **Strip vision tower** (or don't, per step 1's finding) and assemble the
   release candidate.
7. **HumanEval+/MBPP+ accuracy suite.**
8. **SWE-bench validation ladder**: single-instance smoke → small bounded
   sample → full 50-instance pilot. Same promotion discipline as the prior
   project throughout — no step skipped, no full-pilot claim made on
   single-instance evidence. Before first launch, check
   `docs/serving_notes.md` for known-before-launch vLLM flags
   (`--max-cudagraph-capture-size` mamba-cache assertion) rather than
   debugging them cold.

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
