# PagedAttentionMixV3 测试与 Profiler 文档

本文按 **2026-07-12 当前代码**记录自研 `PagedAttentionMixV3` 与 ATB `torch_npu._npu_paged_attention_splitfuse` 的：

- 环境准备和运行方式；
- 当前 11 组正确性回归；
- 单 Shape Profiler 复现方法；
- `profiler_attention_final67` 的 10 次平均数据；
- 自研 Kernel 与 ATB attention core / comparable chain 的对比口径。

> 本版主表和主性能结论固定来自 `profiler_attention_final67`，不与其他 Profiler 轮次混算。跨快照趋势会单独标注。结果只代表该代码快照、CANN/ATB 软件栈、Ascend 310P 设备和固定 Shape，不应直接外推到其他环境。本次只整理已有 CSV，没有重新运行 NPU Profiler。

## 1. 测试环境准备

### 1.1 拉取仓库

```bash
cd ~/var
git clone --recurse-submodules https://github.com/qingyiyi/Ascend_operator.git
cd Ascend_operator
```

### 1.2 Docker 示例

以下是现有测试环境使用过的容器配置。宿主机 `$HOME/var/Ascend_operator` 映射到容器 `/root/var/Ascend_operator`：

```bash
CROSSING_DOCKER_IMAGE=crossing_npu_benchmark:310p-aarch64-8.3.RC2-b92ca26
CONTAINER_NAME=dev_test

docker run -d --net=host --ipc=host \
  --name "${CONTAINER_NAME}" \
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
  -v "$HOME/var/Ascend_operator:/root/var/Ascend_operator" \
  "${CROSSING_DOCKER_IMAGE}" \
  tail -f /dev/null

docker exec -it "${CONTAINER_NAME}" bash
```

如果 CANN、ATB、设备节点或镜像版本不同，需要相应修改。

## 2. 当前 `run.sh` 的真实行为

执行：

```bash
cd ~/var/Ascend_operator/test_op
bash run.sh
```

脚本会先构建和安装 `PagedAttentionMixV3`，重建 PyTorch 扩展，再运行 `test_attention.py`。

会运行命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 PROFILE_TARGET=both PROFILE_REPEAT=10 python test_attention.py
```

## 3. 当前正确性回归

`test_op/test_attention.py::main()` 当前共有 11 个用例，均使用：

```text
Hq = 32
Hkv = 4
Dq = Dv = 128
page_size = 128
GQA group size = 8
```

| # | Batch | 总 Q token | `seq_lengths_host` | `kv_lengths_host` | 覆盖目标 |
|---:|---:|---:|---|---|---|
| 1 | 2 | 256 | `[128, 128]` | `[128, 128]` | 两个完整 128-row slice、单 KV tile |
| 2 | 2 | 256 | `[64, 192]` | `[80, 432]` | 主 Profile Shape、多 KV tile |
| 3 | 2 | 256 | `[63, 193]` | `[80, 432]` | 非 16 对齐 Q 尾块 |
| 4 | 2 | 256 | `[127, 129]` | `[160, 432]` | 127/128/1 边界 |
| 5 | 2 | 256 | `[129, 127]` | `[432, 160]` | 反向 Batch 顺序的边界覆盖 |
| 6 | 2 | 256 | `[1, 255]` | `[80, 432]` | 一行尾块与 128+127 |
| 7 | 2 | 256 | `[17, 239]` | `[80, 432]` | 小尾块与多 slice |
| 8 | 1 | 1 | `[1]` | `[1]` | Decode：单 Q、单 KV token |
| 9 | 1 | 1 | `[1]` | `[128]` | Decode：单 KV page |
| 10 | 1 | 1 | `[1]` | `[512]` | Decode：多 KV page |
| 11 | 2 | 2 | `[1, 1]` | `[80, 432]` | 多 Batch Decode |

### 3.1 Q 的当前布局

测试侧原始 Q 为 ND：

```text
[T, Hq, Dq]
```

传给自研算子前执行：

```python
query_head_major = query_nd.permute(1, 0, 2).contiguous()
query_nz = torch_npu.npu_format_cast(query_head_major, 29)
```

因此自研接口接收的是 **head-major Q**，逻辑 Shape 为：

```text
[Hq, T, Dq]
```

物理 token stride 为 `RoundUp(T, 16)`。长度 Tensor 仍保存逻辑 token 数，不保存 NZ padding 后长度。

### 3.2 通过判据

每个用例都会比较：

```text
自研：torch.ops.npu.paged_attention_mix_v3
参考：torch_npu._npu_paged_attention_splitfuse
```

通过时输出：

```text
========== compare mix_v3 vs splitfuse ==========
mix_v3 vs splitfuse Pass!
```

当前 `diff_tensor.py` 的判据为：

```text
max(abs(actual - reference))
------------------------------------ < 0.1
max(abs(reference)) + 1e-6
```

### 3.3 必须检查 11 条 Pass

`compare_outputs()` 遇到 NaN 或误差超阈值时只打印诊断，不会主动抛异常。因此 Python 进程退出码为 0 不代表全部正确。

推荐：

```bash
cd test_op
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=1 \
PROFILE_TARGET=none \
TEST_REPEAT=1 \
python test_attention.py | tee test_attention.log

PASS_COUNT="$(grep -c 'mix_v3 vs splitfuse Pass!' test_attention.log || true)"
test "${PASS_COUNT}" -eq 11 || {
  echo "Expected 11 passing cases, got ${PASS_COUNT}"
  exit 1
}
```

## 4. 单 Shape Profiler 复现

不建议直接对 `main()` 的 11 个用例全部采集，因为 trace 文件名固定为：

```text
mix_v3.json
atb_splitfuse.json
```

连续用例可能覆盖同名 trace。推荐直接调用固定 Shape：

```bash
cd ~/var/Ascend_operator/test_op
rm -rf profiler_attention

ASCEND_RT_VISIBLE_DEVICES=1 \
PROFILE_TARGET=both \
PROFILE_REPEAT=10 \
PROFILE_WARMUP=3 \
PROFILE_LEVEL=1 \
PROFILE_DIR=profiler_attention \
python - <<'PY'
from test_attention import test_paged_attention

test_paged_attention(
    2,
    256,
    512,
    32,
    4,
    128,
    128,
    [64, 192],
    [80, 432],
)
PY
```

固定 Shape 含义：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `num_batches` | 2 | Batch 数 |
| `batch_seq_len` | 256 | 总 Q token 数 |
| `batch_kv_len` | 512 | 测试输入的最大 KV 长度参考值 |
| `num_heads` | 32 | Q head 数 |
| `num_kv_heads` | 4 | KV head 数 |
| `head_size` | 128 | Q/K/V head dim |
| `page_size` | 128 | KV page token 数 |
| `seq_lengths_host` | `[64, 192]` | 两个请求的 Q 长度 |
| `kv_lengths_host` | `[80, 432]` | 两个请求的 KV 长度 |

该用例还对应：

```text
num_blocks = 5
block_table shape = [2, 5]
output shape = [256, 32, 128]
自研 Block Dim = 8
```

## 5. final67 数据来源

记录数量：

- 自研 `PagedAttentionMixV313`：10 条；
- ATB `TransdataNdToNzKernel`：10 条；
- ATB `MulsF16Kernel`：10 条；
- ATB `PagedAttentionDecoderNzMaskKernel`：10 条；
- ATB `TransdataNzToNdKernel`：10 条。

表中所有数据均为 10 次算术平均，单位为 `us`；`Cube Util` 单位为百分比。

本轮两个 Profile 的 `profiler_info.json` 均记录：

```text
CANN = 8.3.RC2
torch_npu = 2.1.0.post13
AI Core metric = ACL_AICORE_PIPE_UTILIZATION
```

## 6. 对比口径

ATB 调用中包含：

```text
TransdataNdToNzKernel
MulsF16Kernel
PagedAttentionDecoderNzMaskKernel
TransdataNzToNdKernel
```

主对比排除 `TransdataNdToNzKernel`，原因是：

- 自研算子接口本身要求 Q 已经是 head-major NZ；
- 测试代码在调用自研算子前已完成 Q 的 `permute + npu_format_cast(..., 29)`；
- 若把 ATB 内部 Q `ND -> NZ` 计入，而不计自研调用侧对应预处理，会造成接口边界不一致。

因此本文定义：

```text
ATB comparable chain
= MulsF16Kernel
+ PagedAttentionDecoderNzMaskKernel
+ TransdataNzToNdKernel
```

ATB comparable chain 的 `Duration` 是三个 Kernel 平均 Duration 的和。AI Core、Vector、MAC、Scalar、MTE1/2/3 也按三个 Kernel 的平均字段做算术求和，仅用于观察该链路的组成；`Cube Util` 采用 attention core 的值，不对百分比求和。

## 7. Kernel 级别 10 次平均

| Kernel | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 自研 `PagedAttentionMixV313` | 264.3550 | 221.1140 | 200.1891 | 20.4551 | 58.3758 | 24.6347 | 21.1559 | 6.8537 | 83.6430% |
| ATB `TransdataNdToNzKernel` | 11.0790 | 9.5560 | 2.1463 | 0.0000 | 2.3183 | 0.0000 | 3.7145 | 3.8688 | 0.0000% |
| ATB `MulsF16Kernel` | 5.7921 | 4.9260 | 0.9461 | 0.0000 | 0.5908 | 0.0000 | 3.2510 | 2.3835 | 0.0000% |
| ATB `PagedAttentionDecoderNzMaskKernel` | 301.3717 | 204.9550 | 193.4048 | 20.4251 | 41.9992 | 22.1796 | 19.3874 | 5.8998 | 68.0074% |
| ATB `TransdataNzToNdKernel` | 16.1370 | 12.9420 | 5.4988 | 0.0000 | 3.0660 | 0.0000 | 5.1964 | 3.3656 | 0.0000% |

## 8. ATB comparable chain

| 项目 | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATB comparable chain | 323.3008 | 222.8230 | 199.8497 | 20.4251 | 45.6560 | 22.1796 | 27.8348 | 11.6489 | 68.0074% |

如果把 ATB 的 Q `ND -> NZ` 也计入，ATB 四 Kernel 全链路平均 Duration 为：

```text
334.3798 us
```

该值仅作为补充，不作为本文主对比。

## 9. 自研与 ATB attention core

### 9.1 Duration

```text
自研：264.3550 us
ATB attention core：301.3717 us
差值：-37.0167 us
耗时降低：12.2827%
Speedup：1.1400×
```

### 9.2 各字段差值

| 自研 - ATB attention core | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 差值 | -37.0167 | +16.1590 | +6.7843 | +0.0300 | +16.3766 | +2.4551 | +1.7685 | +0.9539 | +15.6356 个百分点 |

注意：虽然自研 wall-clock Duration 更低，但部分 Pipe 指标更高。这些指标在 Kernel 内可以并行重叠，不能作为串行阶段直接相加，也不能仅凭单个 Pipe 时间判断最终 Duration。

## 10. 自研与 ATB comparable chain

### 10.1 Duration

```text
自研：264.3550 us
ATB comparable chain：323.3008 us
差值：-58.9458 us
耗时降低：18.2325%
Speedup：1.2230×
```

### 10.2 各字段差值

| 自研 - ATB comparable chain | Duration | AI Core | Vector | Cube/MAC | Scalar | MTE1 | MTE2 | MTE3 | Cube Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 差值 | -58.9458 | -1.7090 | +0.3394 | +0.0300 | +12.7198 | +2.4551 | -6.6789 | -4.7952 | +15.6356 个百分点 |
