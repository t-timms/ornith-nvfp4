# Ornith MTP head: pruning-compatibility decision (2026-08-23)

Resolves the "dense vs expert-routed" open question flagged as **[UNVERIFIED]** in
`optimization_research_2026-08-23.md` §1/§4. Findings below come from directly inspecting
`ornith-ai/Ornith-1.5-35B-A3B`'s live `config.json` and `model.safetensors.index.json`
(metadata only, no shard weights downloaded), plus reading `reap-cuda`'s actual adapter and
prune-orchestration source (`src/reap/model_adapters.py`, `src/reap/prune.py`) — not assumed
from the library's docs.

## 1. The MTP head is expert-routed, not dense

`mtp.layers.0.mlp.*` in the safetensors index contains, per expert index `0..255`:
`mtp.layers.0.mlp.experts.{i}.{down,gate,up}_proj.weight` (768 tensors, all 256 indices
present) — plus its own router (`mtp.layers.0.mlp.gate.weight`), shared expert
(`mtp.layers.0.mlp.shared_expert.{down,gate,up}_proj.weight`,
`mtp.layers.0.mlp.shared_expert_gate.weight`), and a full self-attention block
(`mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` + `q_norm`/`k_norm` — standard multi-head
attention naming, **not** `linear_attn.*`, so the MTP block behaves like one of Ornith's
`full_attention` layers, not a GDN layer). It is structurally a complete extra decoder layer
with the same MoE-FFN shape as the 40 backbone layers, not a lightweight dense draft head.

## 2. The real risk: `num_experts` is a single shared config field

`text_config.num_experts: 256` is the only expert-count field in the config schema — there is
no separate `mtp_num_experts` (or similar) key. Confirmed in `reap-cuda`'s own
`Qwen3MoeModelAdapter.update_config()` (`model_adapters.py:384-404`): it correctly targets
`config.text_config` for this wrapper shape (the docstring explicitly names Qwen3.5/3.6
conditional-generation wrappers as the reason this method exists) and writes the pruned expert
count there. `reap prune`'s orchestrator (`prune.py:278`) calls this once, using the *backbone's*
retained-expert count.

**Consequence, not yet exercised by anyone (KAT-Coder has no MTP head, so this code path has
never run against a model like Ornith before):** since `get_model_layers()`
(`model_adapters.py:147-170`) only ever resolves `model.layers` /
`model.language_model.layers` / equivalents — never a top-level `mtp` attribute — the MTP
block's expert `nn.ModuleList` and router are **never touched** by the prune loop itself. But
`update_config()` still overwrites the *shared* `text_config.num_experts` field that the model
class presumably uses to instantiate every MoE-shaped submodule, MTP included, at
`from_pretrained` time. Net effect if the checkpoint is saved and reloaded as-is: the MTP
block's on-disk router weight (`mtp.layers.0.mlp.gate.weight`, shape `(256, hidden_size)`) and
its 256 on-disk expert triplets would be loaded against a freshly-instantiated MTP module built
for the *pruned* expert count — a genuine shape mismatch on the router weight specifically
(experts are separate `nn.Module` entries per index, so those would show up as merely
"unexpected keys," but the router is one dense tensor and a size mismatch on a present key is a
hard `RuntimeError` in standard `from_pretrained` loading, not a silent warning). **This would
most likely surface as a checkpoint that fails to reload** — i.e., a multi-hour prune run
completing successfully by its own logs, then breaking at the very next load (quantization,
eval, or serving), the exact class of costly-late-failure this project's "verify at small scale
first" standing practice exists to catch before it happens for real.

## 3. Decision: exclude the MTP head from this build (v1)

Given:
- MTP/speculative-decoding is already a standalone, not-yet-tested optimization lever (research
  doc §4, item 4) with **no KAT-Coder precedent** to compare against.
- No validated recipe exists anywhere (checked) for jointly pruning a backbone MoE and a
  dependent MTP-head MoE consistently — this would be a second, unvalidated compression axis
  stacked onto REAP, which the research doc's §2/§6 already argues against on separate grounds
  (SSM-width pruning).
- The project's own standing discipline is single-variable-at-a-time changes with explicit
  promotion gates, not bundling an untested mechanism into the same run as the core prune.

**Recommended default for the real prune run: disable the MTP head entirely rather than try to
prune it in lockstep with the backbone.** Concretely, as a post-processing step after
`reap prune layerwise` completes (not a `reap-cuda` code change — this is a checkpoint-surgery
step the project's own launch script should perform, and it needs a real GPU-produced checkpoint
to test against before being trusted):

1. Set `text_config.mtp_num_hidden_layers: 0` in the pruned checkpoint's `config.json` (mirrors
   how the base model would represent "no MTP head" — verify this is actually what a from-scratch
   non-MTP Qwen3.5 config looks like before shipping, don't assume the key name's semantics).
2. Drop the now-orphaned `mtp.*` tensors from the saved safetensors shards (they'd otherwise sit
   on disk unused — non-trivial size, since one MTP layer's 256-expert MoE is roughly 1/40th of
   the backbone's *pre-prune* expert weight, i.e. comparable in scale to a single backbone layer).
3. Re-verify the checkpoint reloads cleanly (`AutoModelForCausalLM`/`AutoModelForImageTextToText.
   from_pretrained` + a single forward pass) before treating the prune run as done — same
   single-instance-before-full-spend discipline as everywhere else in this project.

This is a **documented decision, not yet executed** — no GPU run has been made, no checkpoint
exists yet to perform this surgery on. When MTP-1 speculative decoding is revisited later (per
research doc item 4), the natural follow-up experiment is: does re-enabling the *original*
(unpruned, full 256-expert) MTP head against the *pruned* backbone work at all (hidden_size is
unchanged by expert pruning, so this is plausible), and only then ask whether the MTP head itself
should also be pruned to match. That is a separate, explicitly scoped experiment — not bundled
into the base prune run this decision covers.

## Addendum: independent re-verification (2026-08-23, audit pass)

Re-fetched `model.safetensors.index.json` fresh (not reusing the earlier cached copy) and
confirmed the MTP tensor set is fully namespace-clean: **785 tensors total under the `mtp.`
prefix**, of which 4 sit outside `mtp.layers.0.*` — `mtp.fc.weight`, `mtp.norm.weight`,
`mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight` (the standard
DeepSeek-V3-style MTP wiring: a projection combining the next-token embedding with the
backbone's hidden state, plus its own pre-norms and final norm). None of these collide with or
alias a backbone tensor name (no shared/tied-weight naming pattern found) — the concern that
`strip_mtp.py` might need special handling for a tied embedding/lm_head was checked and ruled
out. Confirmed directly against `strip_mtp.py`'s actual matching logic
(`key.startswith("mtp.") or ".mtp." in key`) that all 785 tensors, including the 4 outside
`layers.0`, are caught. No code change needed — this is a positive confirmation, not a fix.

Also searched for any published technique for pruning an MoE-based MTP/speculative head jointly
with its backbone (DeepSeek-V3/R1's MTP being the closest real-world precedent for this exact
structure). **Found none.** The "disable MTP for v1" decision above remains the only
evidence-based option — not revised.
