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

## Next (in order)

1. **Resolve the vision-tower question** before assuming it's free to
   strip. Check whether Ornith's vision weights show signs of real training
   (the published `mmproj-*.gguf` file suggests they might, unlike the
   prior model's confirmed-phantom tower) before treating removal as a
   zero-cost step.
2. **Download the full bf16 checkpoint** (71GB).
3. **Full REAP prune at 50%** (256→128 experts), real CLI, not the smoke
   test path. Watch RAM headroom closely — tighter than the prior build
   (~6GB free vs. ~9GB last time, at `--residency cpu_full`).
4. **Quantize to NVFP4A16 via RTN** (not GPTQ by default — GPTQ showed zero
   measurable improvement over RTN on the prior model's build).
5. **Strip vision tower** (or don't, per step 1's finding) and assemble the
   release candidate.
6. **HumanEval+/MBPP+ accuracy suite.**
7. **SWE-bench validation ladder**: single-instance smoke → small bounded
   sample → full 50-instance pilot. Same promotion discipline as the prior
   project throughout — no step skipped, no full-pilot claim made on
   single-instance evidence.

## Longer-horizon / not scheduled

- GPTQ requantization test, only if RTN's accuracy suite leaves visible room.
- A from-scratch comparison against `kat-coder-nvfp4`'s published numbers,
  once this build has its own validated SWE-bench score.
