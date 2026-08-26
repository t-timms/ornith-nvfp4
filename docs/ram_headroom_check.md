# RAM headroom check for the Ornith REAP prune (2026-08-23)

> **Superseded 2026-08-24 — see the addendum at the bottom.** The ~71.9 GiB
> weight estimate below was a param-count calculation, not a measurement. The
> real prune run's own logs report the actual loaded footprint at **~65.4 GiB**
> - meaningfully smaller, and `--residency cpu_full` turned out to be
> perfectly viable. `run_prune.sh` uses `cpu_full` again as of 2026-08-24. The
> reasoning below is kept as-is for the record; it was a sound precaution
> given the information available at the time.

Checked before launching the real prune run, per this project's "validate before
spending a multi-hour run" standing practice - not a launch-time surprise.

## Verified numbers

- **WSL memory cap**: `C:\Users\ttimm\.wslconfig` sets `memory=80GB`, `swap=16GB`.
  Confirmed live via `free -h` inside WSL: `78Gi total, 74Gi available` (idle,
  no other job running).
- **Ornith-1.5-35B-A3B size**: 35.95B total params confirmed via the HF API
  (`safetensors.total`), bf16 = 2 bytes/param -> **~71.9 GiB of weights alone**.
  This matches the repo's own published shard sizes.
- **`--residency cpu_full`** (KAT-Coder's validated path, see
  `stability_run.sh`/`seqlen_test.sh`: "the validated deterministic path") pins
  the *entire* bf16 model resident in host RAM for the whole observe+prune run.

## The arithmetic

74 GiB available - ~72 GiB for weights alone leaves roughly **2-6 GiB** for
everything else `cpu_full` residency needs concurrently: calibration batch
activations, the REAP observer's running statistics across all layers/experts,
Python/PyTorch/CUDA-host overhead, and normal OS/WSL background usage. That is
a genuinely tight margin, not a comfortable one - this is the basis for the
"~6GB free vs ~9GB last time" note already in ROADMAP.md before this check.

**Could not get an exact comparative number for KAT-Coder's own prune-time peak
RSS** - the original `~/models/KAT-Coder-V2.5-Dev` bf16 checkpoint is no longer
on disk to check its exact size directly, and no peak-memory figure was found
logged in `~/reap-stability/*/prune.log`. The "~9GB last time" figure in
ROADMAP predates this check and is not independently re-verified here - flagging
as **unverified**, not restating it as confirmed.

## Verdict: real risk, not comfortable - mitigated by switching residency mode

`reap prune layerwise --help` documents a third option beyond `cpu_full`/
`gpu_full`: **`layerwise`** - "block-wise observe + disk offload (not full CPU
pin)". This avoids pinning the full 72GiB model in RAM at once.

**Recommendation, now reflected in `scripts/prune/run_prune.sh`: use
`--residency layerwise` for the real Ornith prune run, not `cpu_full`.**

**Real tradeoff, not a free win** — flagging honestly rather than presenting
this as a strict improvement: KAT-Coder's scripts chose `cpu_full` specifically
BECAUSE it's documented as the deterministic, reproducible path (see
`stability_run.sh`: "residency=cpu_full deliberately: it is the validated,
reproducible path"). Whether `layerwise` residency is equally bit-for-bit
deterministic has **not been checked** here (would need reading `reap-cuda`'s
actual observe/offload implementation, not just the CLI help text, or running
two `layerwise` prunes with the same seed and diffing outputs). If a future
stability re-run (multi-seed disagreement check, like KAT-Coder's own
`prune_both_seeds.sh` experiment) is planned for Ornith, verify `layerwise`'s
determinism *before* trusting a seed-to-seed comparison run under it.

## Bottom line for whoever launches this

- Running `cpu_full` as-is on this box is a real OOM risk for a model this
  size - not launched.
- `layerwise` is the safer default and is what `run_prune.sh` now uses.
- If the prune run OOMs anyway under `layerwise`, or if reproducibility needs
  become a concern later, the next lever to check is whether closing other WSL
  memory consumers (or temporarily raising `.wslconfig`'s cap, if total system
  RAM allows) buys enough room to go back to `cpu_full` - not evaluated here
  since it wasn't needed for the `layerwise` recommendation.

## Addendum 2026-08-24: the estimate was too conservative, reverted to `cpu_full`

The `layerwise` recommendation above was never wrong given the information
available on 2026-08-23 — it was a reasonable precaution against an
unverified worst case. But `layerwise` residency then surfaced three real
bugs in `reap-cuda` on Ornith's specific hybrid GDN+attention MoE
architecture (attention-mask handling, and two separate spots that read/
mutate expert weights directly, bypassing `accelerate`'s offload hooks —
full writeup: `docs/layerwise_prune_run_2026-08-24.md`). Rather than keep
patching bugs in a code path this project doesn't need, the RAM estimate
itself was re-checked empirically instead of taken on faith a second time:

- **Actual measured weight footprint under `cpu_full`: ~65.39 GiB** (logged
  directly by `reap-cuda`'s own `reap.prune: Loaded model weight footprint`
  line), not the ~71.9 GiB param-count estimate above. Against ~77 GiB free
  RAM, that is a genuine **~11-12 GiB margin** — comfortable, not tight —
  plus 16 GiB of configured swap as a further backstop.
- Confirmed twice: once at reduced scale (a 2-batch smoke invocation
  completed the entire prune successfully), and once at full scale (the
  real 64-batch, 40-block run stayed at ~75 GiB available RAM throughout
  before being intentionally stopped for the `layerwise` bug investigation
  — not stopped because of any memory pressure).
- `cpu_full` is also KAT-Coder's own "validated deterministic path" (see
  above) — reverting to it resolves the reproducibility question this doc
  flagged as unchecked for `layerwise`, rather than leaving it open.

**Current state**: `scripts/prune/run_prune.sh` uses `--residency cpu_full`
again. The `layerwise` fixes made during the investigation remain in the
`reap-cuda` working tree (uncommitted) in case a future, larger model ever
needs `layerwise` residency for real — see
`docs/layerwise_prune_run_2026-08-24.md` for what's fixed and what isn't.
