#!/usr/bin/env python3
"""Strip Ornith's vision tower from the quantized checkpoint (post-quant surgery).

WHY THIS SCRIPT EXISTS
    docs/vision_tower_decision.md already established the vision tower is
    genuinely trained (not phantom) and that stripping it for the v1 16GB
    text-only build is a real capability trade-off, deliberately accepted.
    This is the script that actually does it, against the real quantized
    checkpoint (~/models/ornith-nvfp4a16-gptq).

WHY A NAIVE "DELETE model.visual.* TENSORS" DOES NOT WORK
    Checked the real modeling code (transformers/models/qwen3_5_moe/
    modeling_qwen3_5_moe.py). Qwen3_5MoeModel.__init__ (used by
    Qwen3_5MoeForConditionalGeneration, the class this checkpoint currently
    declares) unconditionally does:
        self.visual = AutoModel.from_config(config.vision_config)
    with no guard. Deleting the visual.* tensors while keeping this class
    and vision_config would either crash on load (if vision_config is
    removed) or silently rebuild a full random-init 27-block ViT that still
    eats VRAM (if vision_config is kept but weights are gone) - the second
    failure mode would not raise an error, it would just quietly defeat the
    entire point of stripping.

WHAT ACTUALLY WORKS
    transformers ships a real, separate text-only class for this model
    family: Qwen3_5MoeForCausalLM (config: Qwen3_5MoeTextConfig,
    self.model = Qwen3_5MoeTextModel(config) - no self.visual attribute at
    all, confirmed by reading the class body). Switching to it requires:
      1. Config surgery: this checkpoint's config.json already carries a
         complete, self-contained Qwen3_5MoeTextConfig-shaped dict under
         "text_config" (model_type: qwen3_5_moe_text, vocab_size,
         hidden_size, all the MoE/linear-attn fields, its own bos/eos/pad
         token ids) - confirmed by inspection, not assumed. The new
         top-level config.json IS that dict, promoted, plus
         architectures=["Qwen3_5MoeForCausalLM"] and a filtered
         quantization_config (see below). vision_config, image_token_id,
         video_token_id, vision_start/end_token_id are dropped - they only
         mean anything to the ConditionalGeneration wrapper.
      2. Tensor renaming: Qwen3_5MoeForCausalLM expects
         "model.embed_tokens.weight" / "model.layers.*" / "model.norm.weight",
         not "model.language_model.*" (confirmed via
         _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
         on the class). Every "model.language_model.X" key is renamed to
         "model.X". "lm_head.weight" is already top-level and unprefixed in
         this checkpoint (confirmed - it is NOT nested under
         language_model), so it needs no change. Every "model.visual.*" key
         (333 tensors, confirmed count) is dropped.
      3. quantization_config["ignore"] (401 entries) is a concrete list of
         module names, not a regex - confirmed by inspection. All non-visual
         entries are "model.language_model.*"-prefixed (plus a bare
         "lm_head") and get the same rename as the weights; visual entries
         are dropped since those modules no longer exist post-strip.
         config_groups.group_0.targets is just ["Linear"], not per-tensor,
         so it needs no change.

WHAT THIS SCRIPT DOES NOT DO
    Nothing about REAP pruning, MTP, or quantization is touched. This is
    strictly the vision-tower removal + model-class switch, operating on
    the already-quantized checkpoint.

Usage:
    python3 scripts/prune/strip_vision.py \
        --checkpoint ~/models/ornith-nvfp4a16-gptq \
        --output-dir ~/models/ornith-nvfp4a16-gptq-text-only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

LM_PREFIX = "model.language_model."
VISUAL_PREFIX = "model.visual."


def is_visual_key(key: str) -> bool:
    return key.startswith(VISUAL_PREFIX)


def rename_key(key: str) -> str:
    if key.startswith(LM_PREFIX):
        return "model." + key[len(LM_PREFIX):]
    return key  # lm_head.weight and anything else already correctly named


def build_config(src_cfg: dict) -> dict:
    text_cfg = src_cfg.get("text_config")
    if not isinstance(text_cfg, dict):
        raise SystemExit(
            "config.json has no text_config dict - this script assumes the "
            "quantized checkpoint's current multimodal-wrapper shape; "
            "re-check by hand before proceeding."
        )
    if text_cfg.get("model_type") != "qwen3_5_moe_text":
        raise SystemExit(
            f"text_config.model_type is {text_cfg.get('model_type')!r}, not "
            "'qwen3_5_moe_text' - schema may have changed, do not trust the "
            "promotion below blindly."
        )

    new_cfg = dict(text_cfg)  # already self-contained per docstring
    new_cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
    if "transformers_version" in src_cfg:
        new_cfg["transformers_version"] = src_cfg["transformers_version"]

    qc = src_cfg.get("quantization_config")
    if qc is not None:
        qc = json.loads(json.dumps(qc))  # deep copy
        old_ignore = qc.get("ignore", [])
        new_ignore = []
        dropped = 0
        for entry in old_ignore:
            if "visual" in entry:
                dropped += 1
                continue
            new_ignore.append(rename_key(entry) if entry != "lm_head" else entry)
        qc["ignore"] = new_ignore
        print(
            f"  quantization_config.ignore: {len(old_ignore)} -> {len(new_ignore)} "
            f"entries ({dropped} visual entries dropped, rest renamed)",
            flush=True,
        )
        new_cfg["quantization_config"] = qc

    return new_cfg


def strip_weights(checkpoint: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = checkpoint / "model.safetensors"
    with safe_open(src, framework="pt") as f:
        all_keys = list(f.keys())
        visual_keys = {k for k in all_keys if is_visual_key(k)}
        print(f"  total tensors: {len(all_keys)}, visual tensors to drop: {len(visual_keys)}", flush=True)
        tensors = {}
        renamed = 0
        for k in all_keys:
            if k in visual_keys:
                continue
            new_k = rename_key(k)
            if new_k != k:
                renamed += 1
            tensors[new_k] = f.get_tensor(k)
    print(f"  kept {len(tensors)} tensors, renamed {renamed} (model.language_model.* -> model.*)", flush=True)
    save_file(tensors, str(out_dir / "model.safetensors"))
    total_size = (out_dir / "model.safetensors").stat().st_size
    print(f"  wrote model.safetensors: {len(tensors)} tensors, {total_size / 2**30:.2f} GiB", flush=True)


def copy_rest(checkpoint: pathlib.Path, out_dir: pathlib.Path) -> None:
    skip = {"model.safetensors", "config.json"}
    for p in checkpoint.iterdir():
        if p.name in skip:
            continue
        dst = out_dir / p.name
        if p.is_file() and not dst.exists():
            shutil.copy2(p, dst)


def reload_verify(out_dir: pathlib.Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    print(f"  reloaded config class: {type(cfg).__name__}, model_type={cfg.model_type}", flush=True)
    if getattr(cfg, "model_type", None) != "qwen3_5_moe_text":
        raise SystemExit("!! reloaded config is not qwen3_5_moe_text - do not trust this checkpoint")
    if hasattr(cfg, "vision_config") or hasattr(cfg, "text_config"):
        raise SystemExit("!! reloaded config still carries vision_config/text_config nesting - strip failed")

    print("  loading model on CPU for a real forward pass (this is slow, that's expected)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        out_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    print(f"  model class: {type(model).__name__}", flush=True)
    if hasattr(model, "visual"):
        raise SystemExit("!! reloaded model still has a .visual submodule - strip failed")

    inputs = tokenizer("def hello_world():", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    print(f"  FORWARD_OK logits shape: {tuple(out.logits.shape)}", flush=True)
    print("  REVERIFY_PASSED - checkpoint loads as text-only and runs with no vision tower", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--output-dir", required=True, type=pathlib.Path)
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the reload/forward-pass check. For fast iteration on this "
        "script's logic ONLY - never use this on a real run.",
    )
    args = ap.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    out_dir = args.output_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output dir {out_dir} already exists and is non-empty - refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"output    : {out_dir}", flush=True)

    print("[1/3] config", flush=True)
    src_cfg = json.loads((checkpoint / "config.json").read_text())
    new_cfg = build_config(src_cfg)
    (out_dir / "config.json").write_text(json.dumps(new_cfg, indent=2))

    print("[2/3] weights", flush=True)
    strip_weights(checkpoint, out_dir)

    print("[3/3] copying remaining files (tokenizer, etc.)", flush=True)
    copy_rest(checkpoint, out_dir)

    if args.skip_verify:
        print("SKIPPED reload-verify (--skip-verify) - do NOT trust this checkpoint as-is", flush=True)
        return 0

    print("[verify] reload + forward pass", flush=True)
    reload_verify(out_dir)
    print("STRIP_VISION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
