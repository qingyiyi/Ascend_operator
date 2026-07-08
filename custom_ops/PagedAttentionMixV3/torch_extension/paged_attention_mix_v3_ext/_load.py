import pathlib

import torch


def load_extension():
    package_dir = pathlib.Path(__file__).resolve().parent
    project_dir = package_dir.parent
    so_name = "libpaged_attention_mix_v3_ext.so"

    candidates = [
        package_dir / "lib" / so_name,
        project_dir / "build" / "lib" / "paged_attention_mix_v3_ext" / "lib" / so_name,
        project_dir / "build" / "lib" / so_name,
    ]

    for so_path in candidates:
        if so_path.exists():
            torch.ops.load_library(str(so_path))
            return

    raise ImportError(
        "Cannot find libpaged_attention_mix_v3_ext.so. "
        "Run `python setup.py build_ext` or install the wheel under "
        "custom_ops/PagedAttentionMixV3/torch_extension first."
    )