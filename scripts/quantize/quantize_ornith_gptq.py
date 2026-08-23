"""Quantize the REAP-pruned Ornith-1.5-35B-A3B checkpoint to NVFP4A16 via GPTQ.

WHY GPTQ FROM THE START (not RTN-first like KAT-Coder)
    KAT-Coder shipped RTN first, then retrofitted GPTQ later and found it
    accuracy-neutral on that checkpoint (2026-08-22 result, see
    ~/kat-coder-nvfp4's ROADMAP.md). That neutrality doesn't transfer as a
    prediction for Ornith - different expert topology (256 experts here vs
    128), different hybrid-layer ratio - so there's no evidence-based reason
    to default to the weaker method here. llm-compressor's GPTQModifier
    supports NVFP4A16 directly (v0.9-0.10, actively maintained; see
    ~/ornith-nvfp4/docs/optimization_research_2026-08-23.md SS3) and is used
    from the first run, not as a follow-up experiment.

    MR-GPTQ (arXiv 2509.23202, better claimed recovery) stays a documented
    watch-item only - its llm-compressor RFC (issue #2006) is still open/
    unimplemented as of the 2026-08-23 research pass. Not usable here.

IGNORE LIST - carried over from quantize_kat_gptq.py, verified against
Ornith's ACTUAL tensor names (not assumed) before writing this script:
    `re:.*linear_attn.*` - confirmed via
    ~/ornith-nvfp4/tmp_meta/index.json that every Gated-DeltaNet-specific
    tensor name contains this substring: in_proj_qkv, in_proj_a, in_proj_b,
    in_proj_z, out_proj, conv1d, norm, A_log, dt_bias. One nuance the
    research doc's AWQ-recipe analogy didn't need to deal with: A_log,
    dt_bias, and norm are bare nn.Parameter / RMSNorm weights, not
    nn.Linear - GPTQModifier's targets="Linear" was never going to touch
    them regardless of the ignore list. conv1d is nn.Conv1d, also not a
    Linear-target candidate. The tensors that ARE real nn.Linear modules
    needing the explicit ignore are in_proj_qkv/in_proj_a/in_proj_b/
    in_proj_z/out_proj - all covered by the one regex. Net effect: the
    single `.*linear_attn.*` pattern is necessary and sufficient; no
    additional per-tensor patterns needed beyond what KAT-Coder already had.

    `re:.*mtp.*` - carried over from KAT-Coder's script (where it matched
    nothing, since KAT-Coder has no MTP head - a defensive pattern that
    turned out prescient). For Ornith this pattern DOES matter, but the
    real plan (see docs/mtp_pruning_decision.md) is to strip the mtp.*
    tensors out of the checkpoint entirely during REAP post-processing,
    before this script ever runs - so by the time quantization happens,
    this pattern should be matching nothing here too. Left in as a
    defensive no-op in case that stripping step is skipped or deferred.

WHAT IS DELIBERATELY UNCHANGED FROM THE KAT-CODER PRECEDENT
    GPTQ hyperparameters (actorder="static", block_size=128,
    dampening_frac=0.01, offload_hessians=False) and the
    moe_calibrate_all_experts=True safeguard against Hessian breakdowns on
    ill-conditioned expert layers - same reasoning as
    quantize_kat_gptq.py's docstring, not re-derived here since nothing
    about Ornith's architecture changes that reasoning. Calibration set
    (evol-codealpaca-v1, same seed) also unchanged for consistency with the
    sister project's methodology, not because it's been re-validated for
    Ornith's code-domain fit specifically.

NOT YET RUNNABLE AS-IS
    SRC below is a placeholder - it must point at the REAP-pruned, MTP-
    stripped checkpoint, which does not exist yet (no GPU prune run has
    been made - see ROADMAP.md). This script is staged/reviewed ahead of
    that run, not executed.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# TODO: update once the REAP prune run (blocked on user go-ahead, see
# ROADMAP.md) produces a real pruned+MTP-stripped checkpoint directory.
SRC = pathlib.Path(
    os.environ.get(
        "ORNITH_PRUNED_SRC",
        str(pathlib.Path.home() / "reap-stability" / "ornith_TBD" / "pruned_model"),
    )
)
DST = pathlib.Path(
    os.environ.get("GPTQ_DST", str(pathlib.Path.home() / "models" / "ornith-nvfp4a16-gptq"))
)

NUM_CALIB = int(os.environ.get("GPTQ_NUM_CALIB", "256"))
MAX_SEQ = int(os.environ.get("GPTQ_MAX_SEQ", "2048"))

print(f"source : {SRC}", flush=True)
print(f"dest   : {DST}", flush=True)
if not SRC.is_dir():
    raise SystemExit(
        f"source checkpoint missing: {SRC}\n"
        "This script is staged ahead of the real REAP prune run - see "
        "docs/mtp_pruning_decision.md and ROADMAP.md for what has to happen first."
    )

cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
archs = getattr(cfg, "architectures", None) or []
print(f"config : {cfg.__class__.__name__}  architectures={archs}", flush=True)

if any("ConditionalGeneration" in a or "ImageText" in a for a in archs):
    from transformers import AutoModelForImageTextToText as AutoCls

    print("using AutoModelForImageTextToText", flush=True)
else:
    from transformers import AutoModelForCausalLM as AutoCls

    print("using AutoModelForCausalLM", flush=True)

# Sanity check the MTP-stripping decision actually happened before spending
# GPU time on calibration - fail fast rather than quantize a checkpoint that
# will error on reload later (see docs/mtp_pruning_decision.md SS2-3).
tc = getattr(cfg, "text_config", cfg)
mtp_layers = getattr(tc, "mtp_num_hidden_layers", None)
if mtp_layers not in (0, None):
    raise SystemExit(
        f"mtp_num_hidden_layers={mtp_layers} in source config - expected 0. "
        "The MTP-head stripping step from docs/mtp_pruning_decision.md must run "
        "BEFORE quantization, not after."
    )

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoCls.from_pretrained(
    SRC,
    dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"MODEL_LOADED in {time.time() - t0:.1f}s", flush=True)

ds = load_dataset("theblackcat102/evol-codealpaca-v1", split=f"train[:{NUM_CALIB * 2}]")
ds = ds.shuffle(seed=42).select(range(NUM_CALIB))


def preprocess(example):
    text = f"{example['instruction']}\n\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=MAX_SEQ)


ds = ds.map(preprocess, remove_columns=ds.column_names)
print(f"calibration: {len(ds)} samples, max_seq {MAX_SEQ}", flush=True)

recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
        "re:.*conv1d.*",
        "re:.*mtp.*",
    ],
    actorder="static",
    block_size=128,
    dampening_frac=0.01,
    offload_hessians=False,
)

t1 = time.time()
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQ,
    num_calibration_samples=NUM_CALIB,
    moe_calibrate_all_experts=True,
    output_dir=str(DST),
)
print(f"ONESHOT_DONE in {time.time() - t1:.1f}s", flush=True)

tokenizer.save_pretrained(DST)

for fname in (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
):
    src_f = SRC / fname
    dst_f = DST / fname
    if src_f.is_file() and not dst_f.is_file():
        dst_f.write_bytes(src_f.read_bytes())
        print(f"  copied {fname}", flush=True)

print("\n=== verification ===", flush=True)
shards = sorted(DST.glob("*.safetensors"))
total = sum(p.stat().st_size for p in shards)
print(f"  shards: {len(shards)}  total: {total / 2**30:.2f} GiB")

cfg_out = json.loads((DST / "config.json").read_text())
q = cfg_out.get("quantization_config", {})
print(f"  format: {q.get('format')}")
for gname, g in (q.get("config_groups") or {}).items():
    acts = g.get("input_activations")
    print(
        f"  {gname}: weights={g.get('weights', {}).get('num_bits')} "
        f"acts={acts.get('num_bits') if acts else 'None (weight-only)'}"
    )
tc_out = cfg_out.get("text_config", cfg_out)
print(f"  num_experts: {tc_out.get('num_experts')}")
print(f"  mtp_num_hidden_layers: {tc_out.get('mtp_num_hidden_layers')} (expect 0)")

if total / 2**30 > 15.0:
    print("  !! larger than 15 GiB - will not leave room for KV cache on a 16.3 GB card")
elif len(shards) == 0:
    print("  !! NO SHARDS WRITTEN")
else:
    print("  OK: fits a 16 GB card")
print("QUANTIZE_COMPLETE", flush=True)
