# First real REAP prune attempt (2026-08-24) — three reap-cuda bugs found under `layerwise`; resolved by reverting to `cpu_full`

User gave GPU go-ahead overnight ("start the ornith project while I sleep"). Before launching the
real multi-hour `scripts/prune/run_prune.sh`, validated the exact invocation at a drastically
reduced scale (`--batches-per-category 2` instead of 64, scratch `--artifacts-dir`) against the
**real** full checkpoint — not the earlier partial-slice `smoke_ornith.py` compatibility check,
which only exercised layers 0-2 and never reached a `full_attention` block. That validation step
is what caught all three bugs below before they could burn multi-hour GPU time; nothing here
required an actual hours-long run to surface a single one of them.

**Bottom line**: two of the three bugs are fixed and verified end-to-end under `--residency
layerwise` (all 40 blocks pass observation cleanly); the third was found one step further into
the pipeline, during pruning rather than observation, and is not fixed. Rather than continue
patching `layerwise`-specific bugs one at a time — a code path this project does not actually
need — the decision was to re-verify the RAM estimate that originally ruled out `--residency
cpu_full` (see `docs/ram_headroom_check.md`'s 2026-08-24 addendum: it was too conservative) and
revert to `cpu_full`, which has completed the entire pipeline cleanly, twice, with zero bugs.
Residency mode is a memory-staging mechanism, not part of the REAP algorithm — it has no bearing
on the resulting model's quality, so this is a pure efficiency/risk decision, not a compromise.

## Infra fixed first (not reap-cuda bugs — this machine's WSL2 config)

1. **WSL2 VM idle-shutdown**: defaults to tearing down the whole lightweight VM ~60s after the
   last attached `wsl.exe` session ends, killing any `nohup`/`disown`'d background job with it.
   Fixed: added `vmIdleTimeout=-1` to `C:\Users\ttimm\.wslconfig`, `wsl --shutdown` + restart to
   apply. Verified with a live 2-minute background-sleep survival test.
2. **`nohup`+`disown` alone still wasn't enough** even after the VM fix — a background job
   launched that way died with no VM reboot logged (session-teardown, not VM-teardown). Switched
   to `tmux new-session -d`, which does fully detach; verified with the same kind of survival
   test before trusting it with the real job.

Both are one-time host fixes, not project-specific; they'll hold for future overnight runs on
this machine. See `reference_wsl_overnight_job_survival` in this operator's memory system for the
generalized version of this finding.

## Bug 1 (FIXED, verified): layerwise observer replays block-0's attention_mask into every block

**Symptom**: first real full-scale invocation crashed almost immediately —
`RuntimeError: Expected attn_mask dtype to be bool or float or to match query dtype, but got
attn_mask.dtype: long int and query.dtype: c10::BFloat16 instead.` — at block 3
(`model.language_model.layers.3`), the model's *first* `full_attention` block. Blocks 0-2
(Gated-DeltaNet `linear_attention` blocks) had processed fine.

**Root cause**: `reap.layerwise_observer.LayerwiseMoEObserver._capture_first_block_inputs`
captures `attention_mask` exactly once, from a real forward pass hooked at block 0, then
`ReplayCache.materialize` replays that *same* tensor into every subsequent block regardless of
each block's own `layer_types` entry. Ornith's own top-level `Qwen3_5MoeModel.forward` builds
*two* distinct mask representations per real forward call
(`transformers.masking_utils.create_causal_mask` for `full_attention` blocks,
`create_recurrent_attention_mask` for `linear_attention` blocks — see
`modeling_qwen3_5_moe.py` lines ~1285-1297) precisely because the two block types need
differently-shaped/typed masks. The layerwise per-block replay path never replicated that
branch — it silently assumed one mask format works for the whole model, which is true for
homogeneous architectures (why this never showed up on KAT-Coder) but not for a hybrid
GDN+attention model like Ornith.

**Fix** (`reap-cuda` working tree, `src/reap/layerwise_observer.py`, uncommitted — see below):
detect `config.text_config.layer_types` (or `config.layer_types`) at observer init; if more than
one distinct type is present, additionally capture `attention_mask` from a real forward pass
hooked at the first block of *each* other distinct type (same technique already used for block
0, generalized via a new `entry_block_idx`/`mask_only` mode on `_capture_first_block_inputs`);
`_forward_block` now selects the type-correct captured mask per block instead of always reusing
block 0's.

**Verified**: block 3 (`full_attention`) completes cleanly, and every subsequent block —
including transitions back to `linear_attention` — also completes cleanly, through the full
40-block pass. Confirmed twice, both at reduced scale (2 calibration batches).

## Bug 2 (FIXED, verified): MoE expert weights read after accelerate re-offloads them to meta

**Symptom**: crashed at block 6 (`model.language_model.layers.6`) with `NotImplementedError:
Cannot copy out of meta tensor; no data!` inside `reap.kernels.weight_cache.
get_stacked_expert_weights`, called from `observe_moe_batch` → `_process_moe_activations` →
`_after_forward`.

**Root cause**: the model is loaded with `device_map=auto` + disk offload (`accelerate.
big_modeling` logged "Some parameters are on the meta device because they were offloaded to the
cpu" at load time — some but not all parameters ended up disk-backed rather than RAM-resident,
decided automatically by `accelerate` based on available memory). `reap-cuda`'s own manual
per-block loader (`LayerwiseMoEObserver._move_block`, `layerwise_observer.py` ~L328-335)
explicitly skips moving any block that still has meta tensors, deferring to `accelerate`'s own
forward hooks (`AlignDevicesHook`) to materialize real data on-demand during the block's actual
`forward()` call — and that part works fine: the decoder layer's own attention/GDN computation
succeeds even for disk-offloaded blocks, because it runs inside the hooked forward.

`get_stacked_expert_weights` is the problem: it reads `moe.gate_up_proj`/`.down_proj` (or each
expert's `.weight`) directly, from `_after_forward`, which runs **after** `block(*block_input,
**block_kwargs)` has already returned — i.e. after `accelerate`'s post-forward hook has already
re-offloaded that block's weights back to meta. Blocks 0-5 happened to have RAM-resident (not
disk-offloaded) experts, so reading `.weight` directly off them worked by luck; block 6 was the
first block `accelerate` chose to disk-back, and hit the gap.

Confirmed by reading `accelerate`'s actual source (`accelerate/hooks.py`), not guessed: `
add_hook_to_module` monkey-patches `module.forward` itself (not PyTorch's native
`register_forward_hook`) — `new_forward` calls `hook.pre_forward` (materialize), then the real
forward, then `hook.post_forward` (re-offload to meta), all before `new_forward` returns. So by
the time *any* code outside that module's own `forward()` call reads its weights — including a
PyTorch-native `register_forward_hook` on the same module, which always fires after `.forward()`
returns — the weights are already back on meta. `preload_module_classes` (accelerate's documented
mechanism for "classes with submodules registered but not called directly during forward") does
not solve this either: it only changes *what granularity* gets materialized/offloaded together,
not *when* — the offload still happens the instant the (now coarser) module's own forward call
finishes, which is still before `_after_forward` runs. There is no way to get accelerate to leave
weights materialized past their owning module's `forward()` return by construction.

**Fix** (`reap-cuda/src/reap/kernels/weight_cache.py`, uncommitted): `AlignDevicesHook.
pre_forward(module)` / `.post_forward(module, None)` are plain, safely-standalone-callable
methods (confirmed from source: `pre_forward` takes `*args, **kwargs` and is safe with none;
`post_forward(module, output)` just returns `output` unchanged when `io_same_device=False`, the
default for `device_map="auto"` dispatch). Added `_materialize_offloaded(module)` /
`_release_offloaded(...)`, which walk every submodule looking for one with an accelerate
`_hf_hook` where `.offload` is true, and manually call the same `pre_forward`/`post_forward`
accelerate already uses internally — bracketing exactly the weight read in
`get_stacked_expert_weights` with a `try/finally`. Safe because every returned tensor
(`torch.stack(...)` or `.contiguous()`) is already an independent copy by the time the source is
released back to meta — no view aliasing.

**Verified**: re-ran the full 40-block reduced-scale pass with both fixes applied. Block 6 —
the exact block that crashed twice before — now completes cleanly, and so does the entire rest
of the pass through block 40. Both fixes hold together, not just individually.

## Bug 3 (NOT fixed): same class of bug, different call site, found one step further in

**Symptom**: with bugs 1 and 2 fixed, the full 40-block observation phase passed cleanly for the
first time — but the run then crashed during the *pruning* step (mutating the observed model to
drop unselected experts), not observation:
`RuntimeError: Attempted to call variable.set_data(tensor), but variable and tensor have
incompatible tensor type.` in `reap/model_adapters.py:366`, inside `slice_experts`:

```python
all_experts.gate_up_proj.data = all_experts.gate_up_proj.data[keep_indices]
```

**Root cause (characterized, not fixed)**: same underlying class of problem as bug 2 — code that
reads/mutates expert `.data` directly, outside of any `forward()` call, on a block whose weights
may still be on meta. `slice_experts` runs during `apply_pruning`, a separate pass over the model
after observation completes, in a different file (`model_adapters.py`, not `weight_cache.py`) —
the bug-2 fix does not cover it. Unlike bug 2 (a pure read), this is an in-place *mutation* of the
actual expert tensors, which makes the same materialize/release bracketing pattern riskier to
apply blind: assigning into `.data` while a hook still owns that parameter's device/offload state
needs to interact correctly with accelerate's bookkeeping, not just avoid the meta-tensor error.

**Why not fixed**: found late in the same investigation, and by this point there was a clear
better option on the table — `cpu_full` had already completed the entire pipeline (through this
exact pruning step) successfully, twice. Continuing to find and fix bugs one call site at a time
in a code path this project doesn't need was judged not worth it, especially since — unlike bugs
1 and 2 — the checkpoint-save step still hadn't even been reached, so there was no way to know
how many more such spots existed.

## Decision: reverted to `--residency cpu_full`

Residency mode is purely how `reap-cuda` stages weights in memory during observation and
pruning — it does not change the REAP algorithm, the calibration data, the seed, or the
compression ratio, so it has **no effect on the resulting model's quality**. Given that, and
given three bugs found in `layerwise`-specific code paths against zero bugs in two full
`cpu_full` runs, continuing to debug `layerwise` bought nothing toward this project's actual
goal. See `docs/ram_headroom_check.md`'s 2026-08-24 addendum for the corrected RAM math that
made this switch possible (measured ~65.4 GiB weight footprint, not the ~71.9 GiB estimate that
originally ruled `cpu_full` out).

`scripts/prune/run_prune.sh` now uses `--residency cpu_full` again. The bug-1 and bug-2 fixes
remain in the `reap-cuda` working tree (uncommitted) — real, verified work, kept in case a
future, larger model ever needs `layerwise` residency for real. Bug 3 remains open and
undocumented anywhere except here if that day comes.

## State as of this writing

- `reap-cuda` working tree has both fixes uncommitted (`src/reap/layerwise_observer.py`,
  `src/reap/kernels/weight_cache.py`) — not committed or pushed, left for review.
- `ornith-nvfp4` repo: `scripts/prune/run_prune.sh`, `ROADMAP.md`, `README.md`,
  `CHANGELOG.md`, and this doc updated in the working tree to reflect tonight's findings and the
  final `cpu_full` decision — also uncommitted, left for review (per standing instruction: only
  commit when explicitly asked).
- The real prune run (`--residency cpu_full`, 64 batches/category, seed 42, 50% compression)
  **completed successfully** in ~48 minutes with zero errors: 256→128 experts, 36GB checkpoint at
  `~/reap-stability/ornith_n64_s42/model_.../pruned_models/reap-renorm_true-seed_42-0.50`,
  `num_experts: 128` and `mtp_num_hidden_layers: 1` confirmed correct in `config.json`. Next step:
  `scripts/prune/strip_mtp.py` (not run — a separate, deliberate step).
