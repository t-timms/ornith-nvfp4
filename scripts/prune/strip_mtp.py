#!/usr/bin/env python3
"""Strip Ornith's MTP head from a REAP-pruned checkpoint (post-prune surgery).

WHY THIS SCRIPT EXISTS
    See docs/mtp_pruning_decision.md. reap-cuda's prune loop never touches
    the MTP block's weights (Qwen3MoeModelAdapter.get_model_layers() only
    resolves the backbone's `model.layers`), but its update_config() call
    DOES overwrite `text_config.num_experts` - a field the model class
    shares between the backbone and the MTP head. If a pruned checkpoint is
    saved and reloaded as-is, the MTP router weight
    (`mtp.layers.0.mlp.gate.weight`, shape (256, hidden_size) on disk) would
    be loaded against a freshly-instantiated MTP module built for the PRUNED
    expert count - a hard RuntimeError on load (size mismatch on a present
    key), discovered only after the multi-hour prune run already completed.
    Decision: disable the MTP head entirely for the v1 build rather than
    prune it in lockstep with the backbone (no validated recipe exists for
    that, and MTP-1 spec decoding is a separate, later, explicitly-scoped
    experiment - see the decision doc SS3).

WHAT THIS SCRIPT DOES
    1. Sets mtp_num_hidden_layers = 0 in config.json - both at the
       top level and under text_config if present, since it's unverified
       which one (or both) the model class actually reads at load time and
       getting this wrong silently would defeat the whole point.
    2. Drops every tensor whose key starts with "mtp." (or contains
       ".mtp." for wrapped naming) from the checkpoint's weights - handles
       both a sharded checkpoint (model.safetensors.index.json + multiple
       shards; only shards actually containing an mtp.* tensor are
       rewritten, the rest hard-linked/copied unchanged) and a single-file
       one (one model.safetensors, no index - what this project's own
       cpu_full prune run actually produced, ~35GiB, apparently under
       whatever size threshold triggers sharding). If zero mtp.* tensors
       are found at all, that is NOT an error: confirmed real on this
       project's first real pruned checkpoint (2026-08-24) that reap-cuda's
       prune/publish step can omit the MTP head's weights entirely while
       config.json still claims mtp_num_hidden_layers>0 (base checkpoint:
       1811 tensors incl. 785 mtp.*; pruned output: exactly 1026 =
       1811-785). In that case the weights are copied through unchanged
       and only the config fix from step 1 does any real work.
    3. Reload-verifies: AutoConfig + the correct Auto class
       (ForImageTextToText if the architecture name says so, else
       ForCausalLM) .from_pretrained on the stripped checkpoint, then one
       forward pass on a short dummy input, on CPU. This is the actual
       proof the surgery worked - matches this project's "verify at small
       scale before trusting the big run" discipline used everywhere else
       (see quantize_ornith_gptq.py's own fail-fast check, which assumes
       THIS script already ran).

STATUS: Run successfully 2026-08-24 against this project's first real
    pruned checkpoint (`--residency cpu_full`, seed 42, 50% compression -
    see docs/layerwise_prune_run_2026-08-24.md). Two assumptions written
    before any real checkpoint existed turned out wrong and were fixed
    against the real thing rather than left as documented-but-untested:
    (a) the checkpoint layout assumed sharding with an index.json - the
    real output is a single ~35GiB model.safetensors with no index; (b)
    the MTP head's weights were assumed "untouched but shape-mismatched"
    per docs/mtp_pruning_decision.md - empirically they are absent
    entirely from the pruned output, not mismatched (see point 2 above).
    Both are handled now, not just assumed. Step 3's reload-verify remains
    load-bearing - do not skip it (--skip-verify exists only for fast
    iteration on step 1/2's logic, never for a real run).

Usage:
    python3 scripts/prune/strip_mtp.py \
        --checkpoint ~/reap-stability/ornith_n64_s42/reap-<ratio> \
        --output-dir ~/reap-stability/ornith_n64_s42/reap-<ratio>-mtp-stripped
    # or, to modify in place (only after you trust this script):
    python3 scripts/prune/strip_mtp.py --checkpoint <path> --in-place
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def is_mtp_key(key: str) -> bool:
    return key.startswith("mtp.") or ".mtp." in key


def strip_config(src_cfg_path: pathlib.Path, dst_cfg_path: pathlib.Path) -> None:
    cfg = json.loads(src_cfg_path.read_text())

    touched = []
    if "mtp_num_hidden_layers" in cfg:
        cfg["mtp_num_hidden_layers"] = 0
        touched.append("top-level")
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and "mtp_num_hidden_layers" in tc:
        tc["mtp_num_hidden_layers"] = 0
        touched.append("text_config")

    if not touched:
        raise SystemExit(
            "mtp_num_hidden_layers not found anywhere in config.json - "
            "either the schema changed since the 2026-08-23 research pass, "
            "or this checkpoint never had an MTP head. Aborting rather than "
            "silently doing nothing."
        )
    print(f"  config: set mtp_num_hidden_layers=0 at: {', '.join(touched)}", flush=True)
    dst_cfg_path.write_text(json.dumps(cfg, indent=2))


def strip_single_file(checkpoint: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Handle a checkpoint saved as one model.safetensors file, no index.

    Small enough total size (this project's pruned Ornith checkpoint is
    ~35GiB) apparently doesn't get sharded by reap-cuda's save path - no
    model.safetensors.index.json is written at all. HF's from_pretrained
    loads single-file checkpoints fine without one, so the fix mirrors
    strip_shards' filtering logic but writes a single output file and
    skips the index entirely, rather than assuming a sharded layout that
    doesn't match what a real prune run actually produced.
    """
    src = checkpoint / "model.safetensors"

    with safe_open(src, framework="pt") as f:
        all_keys = list(f.keys())
        mtp_keys = {k for k in all_keys if is_mtp_key(k)}
        tensors = None if not mtp_keys else {
            k: f.get_tensor(k) for k in all_keys if k not in mtp_keys
        }

    if tensors is None:
        # No mtp.* tensors at the weight level at all - confirmed real on this
        # project's checkpoint (2026-08-24): reap-cuda's prune/publish step
        # never wrote the MTP head's weights in the first place (base
        # checkpoint has 1811 tensors incl. 785 mtp.*; the pruned output has
        # exactly 1026 = 1811-785, matching to the tensor). This differs from
        # what docs/mtp_pruning_decision.md assumed ("untouched but
        # shape-mismatched") - it's "absent but config-mismatched" instead,
        # which strip_config's fix above still correctly resolves. Nothing to
        # filter here; hardlink the file through unchanged rather than
        # treating this as an error.
        dst = out_dir / "model.safetensors"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        total_size = dst.stat().st_size
        print(
            f"  NOTE: no mtp.* tensors present in model.safetensors - the MTP head's "
            f"weights were already absent from this checkpoint (only the config still "
            f"claimed mtp_num_hidden_layers>0). Linked model.safetensors unchanged: "
            f"{len(all_keys)} tensors, {total_size / 2**30:.2f} GiB.",
            flush=True,
        )
        return

    print(f"  found {len(mtp_keys)} mtp.* tensors to drop (single-file checkpoint, no index)", flush=True)
    save_file(tensors, str(out_dir / "model.safetensors"))
    total_size = (out_dir / "model.safetensors").stat().st_size
    print(f"  wrote model.safetensors: {len(tensors)} tensors, {total_size / 2**30:.2f} GiB", flush=True)


def strip_shards(checkpoint: pathlib.Path, out_dir: pathlib.Path) -> None:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        single_file = checkpoint / "model.safetensors"
        if single_file.is_file():
            strip_single_file(checkpoint, out_dir)
            return
        raise SystemExit(f"no model.safetensors.index.json or model.safetensors at {checkpoint} - unexpected layout")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    mtp_keys = {k for k in weight_map if is_mtp_key(k)}
    if not mtp_keys:
        # See strip_single_file's matching branch: confirmed real on this
        # project's checkpoint (2026-08-24) that reap-cuda's prune/publish
        # step can omit the MTP head's weights entirely while config.json
        # still claims mtp_num_hidden_layers>0 - not an error, just nothing
        # to filter at the tensor level. Copy every shard through unchanged.
        print(
            "  NOTE: no mtp.* tensors found in the weight_map - the MTP head's weights "
            "were already absent from this checkpoint (only the config still claimed "
            "mtp_num_hidden_layers>0). Linking all shards through unchanged.",
            flush=True,
        )
        for shard_name in sorted(set(weight_map.values())):
            src_shard = checkpoint / shard_name
            dst_shard = out_dir / shard_name
            try:
                os.link(src_shard, dst_shard)
            except OSError:
                shutil.copy2(src_shard, dst_shard)
        shutil.copy2(index_path, out_dir / "model.safetensors.index.json")
        print(f"  linked {len(set(weight_map.values()))} shards, {len(weight_map)} tensors unchanged", flush=True)
        return
    print(f"  found {len(mtp_keys)} mtp.* tensors to drop", flush=True)

    shards_with_mtp = {weight_map[k] for k in mtp_keys}
    all_shards = sorted(set(weight_map.values()))
    print(
        f"  {len(shards_with_mtp)}/{len(all_shards)} shards contain mtp.* "
        "tensors and will be rewritten; the rest are linked unchanged",
        flush=True,
    )

    new_weight_map: dict[str, str] = {}

    for shard_name in all_shards:
        src_shard = checkpoint / shard_name
        dst_shard = out_dir / shard_name

        if shard_name not in shards_with_mtp:
            try:
                os.link(src_shard, dst_shard)
            except OSError:
                shutil.copy2(src_shard, dst_shard)
            with safe_open(src_shard, framework="pt") as f:
                for k in f.keys():
                    new_weight_map[k] = shard_name
            continue

        tensors = {}
        with safe_open(src_shard, framework="pt") as f:
            for k in f.keys():
                if is_mtp_key(k):
                    continue
                tensors[k] = f.get_tensor(k)
                new_weight_map[k] = shard_name

        if tensors:
            save_file(tensors, str(dst_shard))
        else:
            print(f"  shard {shard_name} was ENTIRELY mtp.* tensors - dropped, no file written")
            # Any key still pointing at this shard name would be a bug; there
            # should be none since we only assign new_weight_map for kept keys.

    # total_size in the index is metadata-only (HF tooling tolerates it being
    # approximate/stale), but compute it properly from the tensors we kept
    # rather than the bogus running sum above.
    total_size = 0
    for shard_name in {v for v in new_weight_map.values()}:
        total_size += (out_dir / shard_name).stat().st_size

    new_index = {"metadata": {**index.get("metadata", {}), "total_size": total_size}, "weight_map": new_weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
    print(f"  wrote new index: {len(new_weight_map)} tensors, {total_size / 2**30:.2f} GiB", flush=True)


def copy_rest(checkpoint: pathlib.Path, out_dir: pathlib.Path) -> None:
    skip = {"model.safetensors.index.json"} | {
        p.name for p in checkpoint.glob("*.safetensors")
    }
    for p in checkpoint.iterdir():
        if p.name in skip or p.name == "config.json":
            continue
        dst = out_dir / p.name
        if p.is_file() and not dst.exists():
            shutil.copy2(p, dst)


def reload_verify(out_dir: pathlib.Path) -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    if any("ConditionalGeneration" in a or "ImageText" in a for a in archs):
        from transformers import AutoModelForImageTextToText as AutoCls
    else:
        from transformers import AutoModelForCausalLM as AutoCls

    tc = getattr(cfg, "text_config", cfg)
    mtp_layers = getattr(tc, "mtp_num_hidden_layers", getattr(cfg, "mtp_num_hidden_layers", None))
    print(f"  reloaded config: mtp_num_hidden_layers={mtp_layers} (expect 0)", flush=True)
    if mtp_layers not in (0, None):
        raise SystemExit("!! config strip did not take effect on reload - do not trust this checkpoint")

    print("  loading model on CPU for a real forward pass (this is slow, that's expected)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    model = AutoCls.from_pretrained(out_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("def hello_world():", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    print(f"  FORWARD_OK logits shape: {tuple(out.logits.shape)}", flush=True)
    print("  REVERIFY_PASSED - checkpoint loads and runs with the MTP head removed", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=pathlib.Path)
    group.add_argument("--in-place", action="store_true")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the reload/forward-pass check. For fast iteration on this "
        "script's logic ONLY - never use this on a real prune output.",
    )
    args = ap.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    if args.in_place:
        out_dir = checkpoint
    else:
        out_dir = args.output_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"output    : {out_dir}{' (in place)' if args.in_place else ''}", flush=True)

    print("[1/3] config", flush=True)
    strip_config(checkpoint / "config.json", out_dir / "config.json")

    print("[2/3] shards + index", flush=True)
    strip_shards(checkpoint, out_dir)

    print("[3/3] copying remaining files (tokenizer, etc.)", flush=True)
    copy_rest(checkpoint, out_dir)

    if args.skip_verify:
        print("SKIPPED reload-verify (--skip-verify) - do NOT trust this checkpoint as-is", flush=True)
        return 0

    print("[verify] reload + forward pass", flush=True)
    reload_verify(out_dir)
    print("STRIP_MTP_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
