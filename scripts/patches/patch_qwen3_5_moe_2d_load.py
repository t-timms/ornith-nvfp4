"""Local fix for a confirmed llm-compressor gap: qwen3_5_moe / qwen3_5_moe_text
checkpoints saved with 2D per-expert quantized weights (gate_proj/up_proj/down_proj
split by expert index) fail to load correctly, because llmcompressor's
ARCH_TO_2D_MAPPINGS table (modeling/moe/conversion_mappings.py) has no entry for
either model_type. The architecture IS registered in ARCH_TO_IMPORT_PATHS (enough
for GPTQ calibration to linearize the fused experts and quantize them correctly),
but with no 2D mapping, from_pretrained() silently drops every quantized expert
weight as an "unexpected key" and the model's MoE feed-forward layers are left at
random, uninitialized values - shapes and architecture are correct, output is
garbage. Confirmed against this project's own checkpoint (2026-08-24/25): the
underlying GPTQ math was correct all along; this is a load-time bug only.

Confirmed via GitHub, not assumed: vllm-project/llm-compressor PR #3080 (open,
unmerged as of 2026-08-25) adds a real fix, patch_moe_mappings(), and explicitly
names ornith-ai/Ornith-1.0-35B as a real checkpoint hitting this exact issue. The
PR author states they had no hardware to test an actual 2D load - this project is
that real-world validation, and it works: patched load produces coherent output
matching the pre-quantization bf16 checkpoint byte-for-byte on a sanity prompt.

llmcompressor 0.13.0 (installed here, and PyPI's current latest - no newer release
exists yet) predates PR #3080, so `from llmcompressor.modeling import
patch_moe_mappings` is not available. This module reimplements the same fix by
directly patching ARCH_TO_2D_MAPPINGS and by monkeypatching
get_linearize_load_mappings, since the installed version's copy has a second, real
bug on top of the missing mapping: it does `get_checkpoint_conversion_mapping(
model_type)` with no `or []` fallback, and qwen3_5_moe has no default conversion
mapping at all (its upstream release ships fused 3D natively - no default mapping
was ever needed until now), so the unpatched function crashes with
`TypeError: 'NoneType' object is not iterable` before ever reaching our mapping.
PR #3080 fixes this exact line the same way.

Two model_type strings both need registering, not one - transformers'
`_MODEL_TO_CONVERSION_PATTERN` remaps each of them to a DIFFERENT third string
before it's used as the ARCH_TO_2D_MAPPINGS lookup key, confirmed by direct
inspection rather than assumed:
    "qwen3_5_moe"      -> ARCH_TO_IMPORT_PATHS key; remaps to itself ("qwen3_5_moe")
    "qwen3_5_moe_text" -> ARCH_TO_IMPORT_PATHS key (not present by default - this
                           project's vision-strip surgery, scripts/prune/
                           strip_vision.py, produces checkpoints with this
                           model_type); remaps to "qwen3_5_text" (a THIRD string,
                           not equal to either of the above)
Getting either remap wrong silently produces has_linearize_load_mappings() ==
False with no error - it just falls back to the slow, warning-logged post-load
linearize path, which is what happened on the first two attempts before this was
caught by checking _MODEL_TO_CONVERSION_PATTERN directly instead of assuming
identity mapping.

USAGE
    from patch_qwen3_5_moe_2d_load import apply_patch
    apply_patch()

    from transformers import AutoModelForCausalLM
    from llmcompressor.utils import load_context

    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_DIR, dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True,
        )

Call apply_patch() once, before any AutoModelForCausalLM.from_pretrained /
AutoModelForImageTextToText.from_pretrained call for an Ornith NVFP4A16-GPTQ
checkpoint. Safe to call multiple times (idempotent - just re-sets dict entries
and re-assigns the same replacement function).
"""

from __future__ import annotations

from transformers.core_model_loading import WeightRenaming
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    _MODEL_TO_CONVERSION_PATTERN,
)

from llmcompressor.modeling.moe import conversion_mappings as _cm
from llmcompressor.modeling.moe import linearize as _linearize

# The checkpoint's raw per-expert quantized weight keys already use exactly this
# shape (model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_packed etc,
# confirmed by direct inspection of the safetensors keys) - these renamings are
# intentionally near-identity, they exist to register the layout as "known 2D",
# not to actually rename anything.
_RENAMINGS = [
    WeightRenaming(
        source_patterns=r"\.experts\.(\d+)\.gate_proj\.",
        target_patterns=r".experts.\1.gate_proj.",
    ),
    WeightRenaming(
        source_patterns=r"\.experts\.(\d+)\.up_proj\.",
        target_patterns=r".experts.\1.up_proj.",
    ),
    WeightRenaming(
        source_patterns=r"\.experts\.(\d+)\.down_proj\.",
        target_patterns=r".experts.\1.down_proj.",
    ),
]
_REMOVE_TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def _fixed_get_linearize_load_mappings(model_type: str):
    """Same as llmcompressor 0.13.0's version, with PR #3080's `or []` fix for
    architectures (qwen3_5_moe/qwen3_5_moe_text) that have no default checkpoint
    conversion mapping at all."""
    _config_paths, expert_paths = _cm.ARCH_TO_IMPORT_PATHS[model_type]
    experts_cls = _cm.import_or_none(expert_paths)

    mapping = get_checkpoint_conversion_mapping(model_type) or []
    remapped = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    remove_targets, new_mappings = _cm.ARCH_TO_2D_MAPPINGS[remapped]

    save_mappings = [
        c for c in mapping if not any(t in remove_targets for t in c.target_patterns)
    ]
    load_mappings = save_mappings + new_mappings
    return experts_cls, load_mappings, save_mappings


def apply_patch() -> None:
    # qwen3_5_moe_text is not registered in ARCH_TO_IMPORT_PATHS by default -
    # only the multimodal wrapper's "qwen3_5_moe" is. Text-only checkpoints
    # produced by scripts/prune/strip_vision.py declare model_type
    # "qwen3_5_moe_text" and need their own entry, pointed at the same
    # Qwen3_5MoeExperts class.
    _cm.ARCH_TO_IMPORT_PATHS["qwen3_5_moe_text"] = _cm.ARCH_TO_IMPORT_PATHS["qwen3_5_moe"]

    # Register under the REMAPPED keys, not the raw model_type strings - see
    # module docstring. Verified by direct inspection of
    # transformers.conversion_mapping._MODEL_TO_CONVERSION_PATTERN, not assumed.
    _cm.ARCH_TO_2D_MAPPINGS["qwen3_5_moe"] = (_REMOVE_TARGETS, _RENAMINGS)
    _cm.ARCH_TO_2D_MAPPINGS["qwen3_5_text"] = (_REMOVE_TARGETS, _RENAMINGS)

    # linearize.py imported get_linearize_load_mappings by name at its own
    # import time, so patching conversion_mappings' copy alone would not be
    # seen by the code path that actually calls it during from_pretrained.
    _cm.get_linearize_load_mappings = _fixed_get_linearize_load_mappings
    _linearize.get_linearize_load_mappings = _fixed_get_linearize_load_mappings
