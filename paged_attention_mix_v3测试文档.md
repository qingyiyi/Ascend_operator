# PagedAttentionMixV3 测试文档

本文记录当前自研 `PagedAttentionMixV3` 与 ATB `_npu_paged_attention_splitfuse` 的测试环境准备、测试方法、正确性判断方式，以及一次固定规模下的 profiler 对比结果。

## 1. 创建 Docker 测试环境

在宿主机上创建测试容器。用户目录只挂载 `~`，容器内映射为 `/root`，因此后续在容器中拉取的 `~/var` 会对应宿主机的 `$HOME/var`。

```bash
CROSSING_DOCKER_IMAGE=crossing_npu_benchmark:310p-aarch64-8.3.RC2-b92ca26
CONTAINER_NAME=dev_test

docker run -d --net=host --ipc=host \
        --name $CONTAINER_NAME \
        --hostname huawei-arm-01 \
        --runtime ascend \
        --privileged \
        --ulimit stack=67108864 \
        --ulimit memlock=-1 \
        -u root \
        --device /dev/davinci_manager \
        --device /dev/hisi_hdc \
        --device /dev/devmm_svm \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /var/queue_schedule:/var/queue_schedule \
        -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v $HOME:/root \
        $CROSSING_DOCKER_IMAGE \
        tail -f /dev/null
```

进入容器：

```bash
docker exec -it $CONTAINER_NAME bash
```

在容器内拉取测试仓库：

```bash
cd ~
git clone --recurse-submodules https://github.com/qingyiyi/Ascend_operator.git
```

## 2. 如何运行

在测试目录执行：

```bash
cd /root/Ascend_operator/test_op
cd ..
bash run.sh
```

`run.sh` 当前会先编译自定义算子，然后执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 PROFILE_TARGET=both PROFILE_REPEAT=10 python test_attention.py
```

一次运行会同时 profile：

- `mix_v3`：自研 `torch.ops.npu.paged_attention_mix_v3`
- `atb_splitfuse`：ATB `torch_npu._npu_paged_attention_splitfuse`

profile 输出默认写到：

```text
profiler_attention/
```

## 3. 当前测试规模

`test_attention.py` 当前主测试用例是：

```python
test_paged_attention(2, 256, 512, 32, 4, 128, 128, [64, 192], [80, 432])
```

对应含义：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `num_batches` | 2 | batch 数 |
| `batch_seq_len` | 256 | 总 Q token 数 |
| `batch_kv_len` | 512 | 最大 KV 长度参考值 |
| `num_heads` | 32 | Q head 数 |
| `num_kv_heads` | 4 | KV head 数 |
| `head_size` | 128 | Q/K head dim |
| `page_size` | 128 | 每个 KV page 的 token 数 |
| `seq_lengths_host` | `[64, 192]` | 两个 batch 的 Q 长度 |
| `kv_lengths_host` | `[80, 432]` | 两个 batch 的 KV 长度 |

该规模下：

- GQA group size = `num_heads / num_kv_heads = 8`
- Q/K head dim = 128
- page size = 128
- batch0 使用 1 个 KV page，batch1 使用 4 个 KV page

## 4. 正确性验证

正确性通过时，输出里应出现：

```text
========== compare mix_v3 vs splitfuse ==========
mix_v3 vs splitfuse Pass!
```

这里的比较关系是：

- 自研输出：`run_mix_v3()`
- ATB 参考输出：`run_splitfuse()`
- 比较函数：`compare_outputs("mix_v3", out_mix_v3, out_splitfuse, ref_name="splitfuse")`

## 5. Profile 数据来源

本文的 profile 快照来自：

```text
profiler_attention_final47
```

具体 CSV：

```text
profiler_attention_final47/mix_v3_tb/huawei-arm-01_1232180_20260708070835129_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
profiler_attention_final47/atb_splitfuse_tb/huawei-arm-01_1232180_20260708070843513_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
```

统计方式：

- 两边都取 `PROFILE_REPEAT=10` 的 10 次 kernel 记录。
- 表格里的数值为 10 次平均值，单位为 `us`。
- ATB profiler 中会出现 `TransdataNdToNzKernel`，但该 kernel 不计入本文的可比链路。当前自研路径的 Q 输入已经在 Python 侧按 grouped-NZ 物理顺序准备好，因此对齐比较时不应把 ATB 内部的 Q `ND -> NZ` 转换时间算进去。
- 本文的 “ATB comparable chain” 只包含 3 个 kernel：`MulsF16Kernel + PagedAttentionDecoderNzMaskKernel + TransdataNzToNdKernel`。
- “ATB comparable chain” 是把每次 repeat 的这 3 个 kernel duration 相加后再平均；各部件耗时也是按这 3 个 kernel 的字段算术求和，仅用于观察可比链路成本。

## 6. Kernel 级别平均耗时

| 项目 | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 自研 `PagedAttentionMixV312` | 363.707 | 239.207 | 205.319 | 19.317 | 45.598 | 15.037 | 16.792 | 5.980 | 65.769% |
| ATB `TransdataNdToNzKernel` | 10.936 | 9.501 | 2.146 | 0.000 | 2.317 | 0.000 | 3.649 | 3.931 | 0.000% |
| ATB `MulsF16Kernel` | 5.771 | 4.949 | 0.946 | 0.000 | 0.591 | 0.000 | 3.255 | 2.392 | 0.000% |
| ATB `PagedAttentionDecoderNzMaskKernel` | 301.197 | 204.938 | 193.406 | 20.424 | 42.090 | 22.177 | 19.383 | 5.928 | 68.041% |
| ATB `TransdataNzToNdKernel` | 16.038 | 12.917 | 5.499 | 0.000 | 3.063 | 0.000 | 5.228 | 3.385 | 0.000% |

## 7. ATB 可比链路平均耗时

| 项目 | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATB comparable chain | 323.006 | 222.804 | 199.851 | 20.424 | 45.744 | 22.177 | 27.866 | 11.705 | 68.041% |

注意：

- `ATB comparable chain = MulsF16Kernel + PagedAttentionDecoderNzMaskKernel + TransdataNzToNdKernel`。
- `TransdataNdToNzKernel` 不计入可比链路。

## 8. 自研与 ATB 差值

自研单 kernel 相对 ATB attention core：

| 差值：自研 - ATB core | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PagedAttentionMixV312 - PagedAttentionDecoderNzMaskKernel` | +62.510 | +34.269 | +11.913 | -1.107 | +3.508 | -7.141 | -2.591 | +0.052 | -2.272% |

自研单 kernel 相对 ATB comparable chain：

| 差值：自研 - ATB comparable chain | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PagedAttentionMixV312 - ATB comparable chain` | +40.701 | +16.403 | +5.468 | -1.107 | -0.146 | -7.140 | -11.074 | -5.725 | -2.272% |
