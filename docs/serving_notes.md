# Ornith serving notes (running log, started 2026-08-23)

Durable home for concrete serving-time flags/gotchas found during research, before a real
serving script exists for this project (no GPU-produced checkpoint yet - see ROADMAP.md).
Move each item into the actual launch script's own comments once that script is written and
the flag has been confirmed against a real server startup log, not just vLLM's docs.

## Known-before-launch flags

- **`--max-cudagraph-capture-size` (mamba-cache assertion)**: vLLM's own Qwen3.5/3.6 serving
  recipe page documents a mamba-cache assertion error that is fixed by lowering
  `--max-cudagraph-capture-size` below its default of 512. Source:
  [vLLM Qwen3.5/3.6 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html).
  **Not yet triggered or confirmed on this box** - no Ornith server has been started here.
  Know this before the first launch attempt rather than debugging it cold if the assertion
  fires.

- **Prefix caching "align" mode for the Mamba/GDN cache is marked experimental upstream.**
  Worth testing in isolation (a single request, prefix-cache hit/miss checked directly) before
  relying on it for the SWE-bench harness's repeated-instance workload pattern - same
  single-instance-before-full-spend discipline as the rest of this project.

- **`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`**: real,
  vLLM-documented mechanism for Ornith's native `mtp_num_hidden_layers: 1` head. **Contingent
  on the MTP-head pruning decision** (see `docs/mtp_pruning_decision.md`) - the v1 pruned
  checkpoint is planned to ship with `mtp_num_hidden_layers: 0` (MTP disabled), so this flag
  is NOT usable against that checkpoint. Only relevant if/when the "re-enable the original
  unpruned MTP head against the pruned backbone" follow-up experiment happens.

## Open questions, re-verify against a real startup log (don't assume KAT-Coder transfers)

- **FLASH_ATTN backend exclusion**: KAT-Coder's production server log showed vLLM excluding
  FLASH_ATTN from its candidate backend list for its Gated-DeltaNet architecture
  (`potential backends: ['FLASHINFER', 'TRITON_ATTN']`), auto-selecting FlashInfer. No source
  found states this explicitly for Qwen3.5-hybrid/Ornith specifically - treat as
  architecture-family-level evidence likely to transfer (same GDN mechanism, same vLLM
  codepath), but check Ornith's own startup log's candidate-backend line before assuming it,
  the same way the KAT-Coder finding was itself confirmed from a live log rather than inferred.

- **Hybrid KV-cache manager**: vLLM's Qwen3-Next launch post describes a hybrid cache manager
  that auto-tunes logical block size so full-attention and linear-attention layers occupy equal
  physical memory per block - carried into Qwen3.5 per the same post. Our runtime floor
  (`vllm >= 0.19.1` per Ornith's model card) is past the v0.17 introduction point, so this
  should just work, but has not been observed directly on this box yet.

## Audit addendum (2026-08-23, later pass)

- **New: prefix caching is fully ineffective (not just degraded) below the 528-token Mamba
  block size.** [vLLM issue #40696](https://github.com/vllm-project/vllm/issues/40696), open,
  confirmed against our documented floor (v0.19.1): a 479-token prompt gets 0% prefix-cache
  hits purely from falling short of the 528-token block boundary (552 tokens → 95.4% hits).
  Verified this is a performance-only gap, not correctness/corruption, by reading the actual
  issue thread rather than trusting a search snippet. Relevant to the SWE-bench harness since
  early trajectory turns are more likely to be short — expect no prefix-cache benefit on those,
  not a bug to chase if it happens. No workaround exists upstream yet.
