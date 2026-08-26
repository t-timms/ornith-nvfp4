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
  `--residency layerwise`. **Corrected 2026-08-24**: the estimate was too
  conservative (measured ~65.4GiB actual footprint vs. 77GiB free RAM, a
  genuine ~11-12GiB margin); reverted to `cpu_full` after `layerwise`
  surfaced three real `reap-cuda` bugs — see below and
  `docs/layerwise_prune_run_2026-08-24.md`.
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
- Vision-tower phantom-vs-trained question resolved
  (`docs/vision_tower_decision.md`): loaded `model.visual.*` tensors
  directly from the downloaded checkpoint and compared their statistics
  against known transformer-init fingerprints (all-ones norm weight,
  zero linear bias, flat 0.02 std) and against the text backbone's
  known-trained layernorm weights as a positive control. Result: genuinely
  trained, not phantom — norm weights show real cross-layer structure and
  are essentially never exactly 1.0, biases have drifted far from zero
  init. Decision unchanged (still strip it for the text-only 16GB build),
  but it's now documented as a deliberate capability trade-off rather than
  a free removal of dead weight.
- WSL2 overnight-job reliability fixes (2026-08-24, this machine's host
  config, not project-specific): `vmIdleTimeout=-1` added to `.wslconfig`
  to stop the WSL2 VM tearing itself down ~60s after the launching session
  ends; background jobs now launched via `tmux new-session -d` instead of
  `nohup`+`disown`, which was verified not to reliably survive session
  teardown even with the VM fix in place.
- Real REAP prune completed (2026-08-24): `--residency cpu_full`, seed 42,
  50% compression (256→128 experts), 64 calibration batches/category
  against `theblackcat102/evol-codealpaca-v1`. Ran to completion in ~48
  minutes with zero errors. Output: `~/reap-stability/ornith_n64_s42/
  model_.../pruned_models/reap-renorm_true-seed_42-0.50`, 36GB —
  `num_experts: 128` and `mtp_num_hidden_layers: 1` confirmed correct in
  `config.json`. One benign startup warning (`residency=cpu_full but
  model~65.4GiB may exceed safe host RAM`) did not materialize into any
  actual problem — RAM stayed above ~73GiB available throughout.
- MTP head stripped from the pruned checkpoint (2026-08-24):
  `scripts/prune/strip_mtp.py`, run for the first time against a real
  checkpoint. Reload-verified with a real CPU forward pass — logits shape
  `(1, 4, 248320)` matches the model's actual `vocab_size`, independently
  confirmed rather than just trusted from the script's exit code. Output:
  `<prune dir>-mtp-stripped`, 36GB, 1026 tensors, zero `mtp.*` keys
  remaining. See Fixed, below, for the two script bugs this run exposed.
- GPTQ-NVFP4 quantization mechanism verified and ETA measured (2026-08-24),
  launch paused. `scripts/quantize/quantize_ornith_gptq.py` requires a
  separate environment (`~/quant-env`, `llmcompressor` 0.13.0 — not
  installed in `reap-cuda-env`). Two bounded probes: an 8-sample run
  confirmed `moe_calibrate_all_experts=True` correctly reaches Ornith's
  fused MoE expert weights despite `targets="Linear"` normally matching
  `nn.Linear` module instances by type (experts here are raw fused
  parameters on a custom container, not individual `nn.Linear`
  submodules) — zero errors across 5 layers. A second probe at real
  settings (256 samples, 2048 seq length) measured ~3.27s/expert
  (1.45s+1.45s+0.37s gate/up/down), essentially unchanged from the
  8-sample measurement, confirming the cost is dominated by the fixed
  per-matrix GPTQ solve, not calibration volume. 5,120 experts (40 layers
  × 128) × 3.27s ≈ 4.65 hours for the expert MLPs, ~4.75-5 hours total.
   Both probes killed once they'd measured what was needed — no quantized
   checkpoint exists yet.
- **GPTQ-NVFP4 quantization completed (2026-08-25, ~5.5 hours).** Full run
  at the probed settings (256 samples, 2048 seq length,
  `moe_calibrate_all_experts=True`). Output: 12.47 GiB checkpoint
  (`~/models/ornith-nvfp4a16-gptq`), compressed-tensors
  `nvfp4-pack-quantized` format, 291 tensors left unquantized via the
  ignore list (routers, gates, embeddings, linear-attention projections).
  The "suspected corruption" scare after the run resolved as this load-time
  bug below — the quantization itself was correct all along.
- Text-only serving variants of the checkpoint (2026-08-25):
  `ornith-nvfp4a16-gptq-text-only` (`scripts/prune/strip_vision.py`: vision
  declaration stripped per `docs/vision_tower_decision.md`, switching the
  checkpoint to transformers' real text-only
  `Qwen3_5MoeForCausalLM` class — config surgery + `model.language_model.*`
  key renames + dropping the 333 visual tensors) and
  `ornith-nvfp4a16-gptq-text-only-norope` (a further config-only edit:
  multi-axis mrope position fields removed, which text-only serving does
  not use; no tensors touched by this step). Both are the evaluated
  artifact.
- Local fix for a confirmed llm-compressor load-time gap (2026-08-25):
  `qwen3_5_moe` / `qwen3_5_moe_text` checkpoints with 2D per-expert
  quantized weights fail to load — llmcompressor's `ARCH_TO_2D_MAPPINGS`
  has no entry for either model type, so `from_pretrained()` silently
  drops every quantized expert weight as an "unexpected key" and the MoE
  layers run at random init. Shapes look correct; output is garbage.
  Confirmed real upstream gap, not project-specific:
  vllm-project/llm-compressor PR #3080 (open) fixes it and explicitly names
  an Ornith checkpoint as affected. Patched locally
  (`scripts/patches/patch_qwen3_5_moe_2d_load.py`); patched load produces
  coherent output matching the pre-quantization bf16 checkpoint on a sanity
  prompt. This project is the real-world validation PR #3080's author said
  they had no hardware for.
- Two vLLM gaps fixed locally to serve the text-only checkpoint
  (`~/vllm-src`, 2026-08-25), both confirmed real upstream gaps by reading
  upstream code, not project misconfigurations. Note (added 2026-08-25,
  later): both have since been fixed upstream independently by vLLM PR
  #50210 (text-only Qwen3.5 dense/MoE support), which is newer than this
  project's pinned checkout — no upstream PRs needed from here, and the
  local patches remain for reproducibility of tonight's runs:
  - Missing model registry entry: the checkpoint declares
    `Qwen3_5MoeForCausalLM` (the text-only architecture transformers maps
    from `model_type qwen3_5_moe_text`), which vLLM's registry did not list.
  - `Qwen3_5ForCausalLMBase` was missing the three
    `get_mamba_state_*_from_config` classmethods that `IsHybrid` requires;
    the implementations only existed on `Qwen3_5ForConditionalGeneration`.
    Copied verbatim (they reference only `vllm_config` and generic Mamba
    state calculators). Without them, vLLM crashes with an AttributeError
    on the first executed request — after model load and KV-cache init
    both succeed.
- First real accuracy evaluation of the whole pipeline (2026-08-25), served
  through patched vLLM: HumanEval+ **84.15%** (138/164), HumanEval
  90.24% (148/164), MBPP+ 89.15% (337/378) — greedy, instruct framing,
  lm-eval-harness, Wilson stderr 2.86/2.32/1.60pp respectively. Full
  706-problem suite in ~12 minutes on the vLLM
  backend (vs. an extrapolated 40+ hours on hf before the fixes above).
  Raw results in `~/eval-suite-ornith-vllm/`; task definitions and launch
  scripts committed under `scripts/eval/`. For tier context:
  kat-coder-16gb scored 96.34/89.63/89.42 on the same three tasks — Ornith
  is lower across the board but same tier, reasonable given more aggressive
  (50%) REAP pruning on a different base model.

### Changed

- Quantization default corrected from RTN-first to GPTQ-NVFP4-first. The
  prior build's "GPTQ showed zero measurable improvement over RTN" finding
  doesn't transfer to Ornith's different expert topology (256 vs. 128
  experts); no evidence-based reason remained to default to the weaker
  method.
- Pruning residency mode corrected from `cpu_full` to `layerwise` per the
  RAM headroom check above. Disclosed tradeoff: `layerwise`'s bit-for-bit
  reproducibility hasn't been validated the way `cpu_full` has.
- **Reverted (2026-08-24): pruning residency mode corrected back from
  `layerwise` to `cpu_full`.** `layerwise` surfaced three real `reap-cuda`
  bugs on this hybrid GDN+attention MoE model (see Fixed, below, and
  `docs/layerwise_prune_run_2026-08-24.md`); the RAM concern that
  originally motivated `layerwise` was re-measured and found overly
  conservative (see the RAM headroom entry above). Residency mode does not
  affect the REAP algorithm, calibration data, or resulting model quality
  — this is a pure risk/efficiency decision, not a compromise. `cpu_full`
  also resolves the reproducibility question `layerwise` left open, being
  the prior release's own validated deterministic path.

### Fixed

- Found and fixed a real bug in `reap-cuda`'s pairwise expert co-occurrence
  metric: it built a one-hot tensor as `torch.long` and matmul'd it
  directly, which CUDA has no integer GEMM kernel for. Not specific to
  this checkpoint — would hit any `qwen3_5_moe` model exercising that
  metric. Fixed upstream in the fork (cast to float32 for the matmul,
  round, cast back — exact for any realistic count):
  `t-timms/reap-cuda`, `fix/qwen3-5-router-renormalization`, commit `b67be1a`.
- Found and fixed two real bugs in `reap-cuda`'s `--residency layerwise`
  path, exposed by Ornith's hybrid Gated-DeltaNet + attention MoE
  architecture (2026-08-24, uncommitted in `reap-cuda`'s working tree —
  full writeup in `docs/layerwise_prune_run_2026-08-24.md`):
  - `LayerwiseMoEObserver` captured `attention_mask` once from block 0 and
    replayed the same tensor into every block regardless of layer type,
    crashing the instant the model switched from `linear_attention` to
    `full_attention` blocks. Fixed by capturing and selecting the correct
    mask per layer type (`src/reap/layerwise_observer.py`).
  - `get_stacked_expert_weights` read MoE expert weights directly, bypassing
    `accelerate`'s forward-hook-based on-demand materialization, so any
    disk-offloaded block's weights were already back on `meta` by the time
    they were read — `NotImplementedError: Cannot copy out of meta tensor`.
    Fixed by explicitly bracketing the read with `accelerate`'s own
    `AlignDevicesHook.pre_forward`/`post_forward` (`src/reap/kernels/
    weight_cache.py`).
  - A third, related bug (same class — direct `.data` mutation on
    possibly-offloaded expert tensors, this time in the pruning/mutation
    step, `model_adapters.py`'s `slice_experts`) was found but not fixed;
    see `docs/layerwise_prune_run_2026-08-24.md` for why.
- Found and fixed two real bugs in this project's own
  `scripts/prune/strip_mtp.py`, written and reviewed the day before any
  real pruned checkpoint existed to test it against (2026-08-24):
  - Assumed a sharded checkpoint layout (`model.safetensors.index.json` +
    multiple shards); the real `--residency cpu_full` output is a single
    ~35GiB `model.safetensors` with no index. Added a single-file code
    path (`strip_single_file`) alongside the existing sharded one.
  - Treated "zero `mtp.*` tensors found" as an error (either a naming
    mismatch or an already-stripped checkpoint). On the real checkpoint
    this was neither — `reap-cuda`'s prune/publish step had already
    omitted the MTP head's weights entirely while `config.json` still
    claimed `mtp_num_hidden_layers>0` (see
    `docs/mtp_pruning_decision.md`'s 2026-08-24 addendum). Fixed to treat
    this as a legitimate case: copy the weights through unchanged and
    rely on the config fix, both in the single-file and sharded paths.
