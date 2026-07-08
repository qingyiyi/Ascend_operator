# custom_ops

Ascend C custom operator playground for learning, building, and validating custom CANN operators.

This repository currently contains:

- `AddCustom/`: minimal custom add operator sample.
- `pagedattention/`: paged attention splitfuse/chunk prefill operator.
- `PagedAttentionMixV3/`: custom `PagedAttentionMixV3` operator, including the Ascend C kernel, host tiling code, packaging scripts, and a PyTorch/torch-npu extension wrapper.
- `AclNNInvocation/`: aclnn single-operator invocation sample used to validate `AddCustom`.
- `vendors/`: generated OPP install tree. This directory is produced by build/install flows and is ignored by Git.

## Repository Layout

```text
.
├── AddCustom/
│   ├── op_host/            # Operator definition and tiling code
│   ├── op_kernel/          # Ascend C kernel implementation
│   ├── framework/          # Framework adapter build files
│   ├── cmake/              # Shared build helpers
│   └── build.sh            # Build/package entry
├── pagedattention/
│   ├── op_host/
│   ├── op_kernel/
│   ├── framework/
│   └── build.sh
├── PagedAttentionMixV3/
│   ├── op_host/
│   ├── op_kernel/
│   ├── framework/
│   ├── torch_extension/    # torch-npu extension loader/wrapper
│   ├── build.sh
│   └── run.sh              # Local deployment helper for one environment
├── AclNNInvocation/
│   ├── inc/
│   ├── src/
│   ├── scripts/
│   └── run.sh
└── vendors/                # Generated CANN custom OPP output
```

## Prerequisites

You need an Ascend C/CANN development environment with the Ascend toolkit installed. The build scripts look for these environment variables in order:

```bash
BASE_LIBS_PATH
ASCEND_HOME_PATH
ASCEND_AICPU_PATH
```

If none of them is set, `build.sh` exits with `please set env.`. A typical setup is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

For the PyTorch extension, install compatible `torch` and `torch_npu`, and make sure `cmake >= 3.18` is available.

## Build Operators

Each operator project has its own `build.sh`.

```bash
cd AddCustom
bash build.sh
```

```bash
cd pagedattention
bash build.sh
```

```bash
cd PagedAttentionMixV3
bash build.sh
```

By default the scripts build the `binary` target and then package the custom OPP into `build_out/`. You can pass another target when needed:

```bash
bash build.sh install
```

Generated outputs such as `build_out/`, `*.run`, CMake files, object files, and shared libraries are ignored by Git.

## Install Custom OPP

After a successful build, install the generated package into your Ascend toolkit OPP path:

```bash
cd PagedAttentionMixV3
./build_out/custom_opp_ubuntu_aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If you prefer a local generated vendor tree, set the custom OPP path and library path explicitly:

```bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/custom_ops/vendors/customize
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
```

## Build the PagedAttentionMixV3 PyTorch Extension

The torch extension wraps the custom `PagedAttentionMixV3` operator for torch-npu.

```bash
cd PagedAttentionMixV3/torch_extension
python setup.py build_ext
```

The Python package loads `libpaged_attention_mix_v3_ext.so` from either the package `lib/` directory or the local `build/lib` output.

## Run the aclnn AddCustom Sample

`AclNNInvocation` generates test data, builds a small executable, runs the custom op, and compares the result with golden data.

```bash
cd AclNNInvocation
bash run.sh
```

The input/output `.bin` files and local build directory are generated artifacts and are ignored by Git.

## Git Hygiene

The repository ignores generated content, including:

- CMake outputs: `build/`, `build_out/`, `CMakeFiles/`, `CMakeCache.txt`
- compiled artifacts: `*.o`, `*.so`, `*.run`
- generated OPP install tree: `vendors/customize/`
- Python cache/build outputs
- local invocation data under `AclNNInvocation/input` and `AclNNInvocation/output`
- debug dumps and tensor cache files

If ignored files were already committed before `.gitignore` was added, remove them from Git tracking while keeping them locally:

```bash
git ls-files -ci --exclude-standard
git ls-files -ci --exclude-standard -z | xargs -0 git rm -r --cached --
git add .gitignore
git commit -m "chore: ignore generated build artifacts"
```

This removes the files from future commits, but does not erase them from older Git history. If a committed artifact contains credentials, tokens, or other sensitive data, rewrite repository history with a tool such as `git filter-repo` or BFG and force-push with care.

## Notes

- Keep source changes in `op_host/`, `op_kernel/`, `framework/`, `cmake/`, and `torch_extension/`.
- Treat `build_out/` and `vendors/customize/` as reproducible outputs.
- Re-source the Ascend toolkit environment after installing a custom OPP package.
