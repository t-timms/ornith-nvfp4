#!/usr/bin/env python3
"""Same entry point as the `lm_eval` console script, except it applies
patch_qwen3_5_moe_2d_load.apply_patch() first. lm_eval's hf backend loads models
internally via AutoModelForCausalLM.from_pretrained - there's no CLI flag to hook
in a pre-load patch, so this wraps the real entry point instead. See
patch_qwen3_5_moe_2d_load.py for what the patch does and why it's needed.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "patches"))
from patch_qwen3_5_moe_2d_load import apply_patch  # noqa: E402

apply_patch()

from lm_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    sys.exit(cli_evaluate())
