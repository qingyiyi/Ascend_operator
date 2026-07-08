# PagedAttentionMixV3 开发文档

本文档基于 `PagedAttentionMixV3_写文档版` 代码整理，用于说明自研 `PagedAttentionMixV3` 的接口契约、tiling 逻辑、device 执行顺序、数据布局、同步关系和维护注意事项。文档不绑定具体测试规模，所有大小均以 tiling 参数和运行时输入为准。

## 1. 代码位置

写文档版源码位置：

```text
customadd/PagedAttentionMixV3
```

主要文件：

- `op_host/paged_attention_mix_v3.cpp`
- `op_host/paged_attention_mix_v3_tiling.h`
- `op_kernel/paged_attention_mix_v3.cpp`
- `torch_extension/csrc/host/paged_attention_mix_v3_ext.cpp`

## 2. 接口契约

host op 注册中，核心输入输出格式如下：

| 输入/输出 | 逻辑含义 | 格式约定 |
|---|---|---|
| `query` | Q token 和 Q heads | `FORMAT_FRACTAL_NZ` |
| `kCache` | paged KV cache 的 K | `FORMAT_FRACTAL_NZ` |
| `vCache` | paged KV cache 的 V | `FORMAT_FRACTAL_NZ` |
| `attentionMask` | attention mask | `FORMAT_FRACTAL_NZ` |
| `blockTable` | 逻辑 page 到物理 page 的映射 | `FORMAT_ND` |
| `kvLengths` | 每个 batch 的 KV 长度 | `FORMAT_ND`，tiling value-dependent |
| `seqLenPerRequest` | 每个 batch 的 Q 长度 | `FORMAT_ND`，tiling value-dependent |
| `queryRope` | 预留输入 | 当前 kernel 不使用 |
| `output` | attention 输出 | `FORMAT_ND` |

`query` 的重点是：它不再是普通 ND 物理布局。Python/调用侧需要传入已经转换为 NZ 物理格式的 Q，逻辑 shape 仍用于表达 `[numTokens, qHeadNum, qkHeadSize]`，但 kernel 按 NZ 物理顺序读取。

`InferShape` 以 `query` 作为输出 shape 基础，并根据 `value` hidden size 和 KV head 数修正输出最后一维，使输出逻辑为：

```text
[numTokens, qHeadNum, vHeadSize]
```

## 3. Host 侧 Tiling

`TilingFunc` 从输入 shape 和常量输入中提取运行参数：

- `numTokens`
- `qHeadNum`
- `qkHeadSize`
- `pageSize`
- `kvHeadNum`
- `vHeadSize`
- `maskColSize`
- `maskRowSize`
- `pageNumPerBatch`
- `gqaGroupSize`
- `usedCoreNum`
- `softmaxScale`

核心推导关系：

```text
kvHiddenSize = kHiddenBlocks * NZ_C0_SIZE
kvHeadNum    = kvHiddenSize / qkHeadSize
vHiddenSize  = vHiddenBlocks * NZ_C0_SIZE
vHeadSize    = vHiddenSize / kvHeadNum
gqaGroupSize = qHeadNum / kvHeadNum
```

host 侧会检查：

- Q head 数必须能被 KV head 数整除。
- Q/K/V/mask 的 head dim 或 C0 必须满足 NZ/cube 对齐要求。
- page size 必须满足 kernel 支持范围和对齐要求。
- `seqLenPerRequest` 的和必须覆盖 `numTokens`。
- 每个 batch 的 `kvLen` 必须不小于 `seqLen`。
- mask 的 row/col padding 必须覆盖本次请求会访问的区域。
- 每个 batch 使用的逻辑 page 数不能超过 `blockTable` 形状能表示的范围。

`qBlockRows` 由 GQA tile size 决定：

```text
gqaTileSize = min(gqaGroupSize, MAX_GQA_TILE_SIZE)
qBlockRows  = align_down(MAX_Q_ROWS_PER_GROUP / gqaTileSize, CUBE_BLOCK_SIZE)
```

总 task 数按 batch、Q block 和 KV head group 展开：

```text
totalTaskNum = sum_over_batch(ceil(seqLen / qBlockRows) * kvHeadNum)
```

`usedCoreNum` 取平台 AI Core 数和 `totalTaskNum` 的较小值。

## 4. Device 总体执行顺序

device 入口为：

```cpp
extern "C" __global__ __aicore__ void paged_attention_mix_v3(...)
```

入口内部创建 `PagedAttentionMixV3Kernel`，执行：

```text
Init()
Process()
```

`Init()` 负责绑定 GM tensor，并调用 `InitLocalBuffer()` 切分 L1、L0A、L0B、L0C 和 UB。

`Process()` 的主流程是：

```text
按 core 切分 task
定位起始 batch/qBlock/group
PrimeSingleBufferFlags
遍历 task
  遍历 GQA tile
    计算 qTokenStart 和 qHeadStart
    遍历 KV tile pair
      BMM1 Ping
      BMM1 Pong
      Softmax Ping
      Softmax Pong
      BMM2 Ping
      BMM2 Pong
      Update Ping
      Update Pong
    释放 Q L1 guard
    NormalizeAndWriteOutput
  推进 group/qBlock/batch
SyncEnd
```

该版本采用 ping/pong KV tile pair 结构。ping 和 pong 有独立的 K/V/mask/prob/score/PV buffer，但 Q L1 是共享缓冲。

## 5. Task 划分和运行时游标

每个 task 对应：

```text
一个 batch
一个 qBlock
一个 KV head group
```

device 侧通过 `taskIdStart` 找到起始 task 所属的 batch、Q block 和 group：

```text
qBatchOffset = 当前 batch 在全局 Q token 维的起点
qBlockStart  = 当前 Q block 在 batch 内的起点
groupIdx     = 当前 KV head group
```

每个 GQA tile 内进一步计算：

```text
qTokenStart = qBatchOffset + qBlockStart
qHeadStart  = groupIdx * gqaGroupSize + gqaTileStart
```

这两个值直接决定 Q NZ GM 中当前 tile 的读取起点。

## 6. Q 的 NZ 链路

### 6.1 Q 输入格式

写文档版中，host 侧将 `query` 注册为：

```cpp
Format({ge::FORMAT_FRACTAL_NZ})
```

因此 Q 的实际物理输入应是由原始逻辑 Q：

```text
[numTokens, qHeadNum, qkHeadSize]
```

转换得到的 NZ tensor。kernel 侧按如下物理顺序理解 Q：

```text
[token][kBlock][head][c0]
```

其中：

```text
kBlock = qkHeadSize / CUBE_BLOCK_SIZE
c0     = CUBE_BLOCK_SIZE
```

这和之前“手工拼成 grouped-NZ 大连续块”的方案不同。当前版本不依赖 `queryGroupNzSize/qBatchNzOffset/queryTileNzOffset`，而是直接用 token 起点和 head 起点在真实 Q NZ 中取数。

### 6.2 CopyQueryToL1

`CopyQueryToL1` 的职责是：从真实 Q NZ GM 中抽取当前 Q block、当前 GQA head tile，并写入 kernel 内部 folded-GQA NZ 布局的 `queryL1`。

关键参数：

```cpp
CopyQueryToL1(qTokenStart,
              qHeadStart,
              realQBlockRows,
              qkHeadSize,
              gqaTileSize)
```

函数内部计算：

```text
mSize        = qBlockRows * gqaTileSize
qkHeadBlocks = qkHeadSize / CUBE_BLOCK_SIZE
tokenStride  = qHeadNum * qkHeadBlocks * CUBE_BLOCK_SIZE
blockLen     = gqaTileSize * CUBE_BLOCK_SIZE * sizeof(query_dtype) / DATA_BLOCK_BYTES
srcStride    = (tokenStride - gqaTileSize * CUBE_BLOCK_SIZE) * sizeof(query_dtype) / DATA_BLOCK_BYTES
```

对每个 `kBlock` 执行一次 `DataCopy`：

```text
srcOffset = qTokenStart * tokenStride
          + kBlock * qHeadNum * CUBE_BLOCK_SIZE
          + qHeadStart * CUBE_BLOCK_SIZE

dstOffset = kBlock * mSize * CUBE_BLOCK_SIZE
```

`DataCopyParams` 的语义：

```text
blockCount = realQBlockRows
blockLen   = 当前 GQA tile 的 heads * c0
srcStride  = 跳过同一个 token 中不属于当前 GQA tile 的其余 Q head 和后续 kBlock 区域
dstStride  = 0
```

所以每个 burst 对应一个 Q token，在固定 `kBlock` 下连续搬当前 GQA tile 覆盖的 heads。循环所有 `kBlock` 后，`queryL1` 中形成：

```text
[kBlock][qRow * gqaTileSize + headInTile][c0]
```

这就是后续 `LoadQueryToL0A()` 需要的 folded-GQA NZ 布局。

### 6.3 LoadQueryToL0A

`LoadQueryToL0A` 从 `queryL1` 按 M block 装入 L0A。由于 L0A 作为 cube A 操作数使用，`LoadData` 会按硬件约定转换成 MMAD 需要的布局。

Q L1 是 ping/pong 共享缓冲。首次加载当前 Q tile 前，需要等待 ping 和 pong 两条路径都释放写保护；当前 GQA tile 的 KV 循环结束后，再统一释放 Q guard。

## 7. K/V Cache 和 Mask 布局

### 7.1 K/V Cache

K/V cache 均为 `FORMAT_FRACTAL_NZ`。kernel 通过 `blockTable` 将逻辑 page 映射到物理 page，再根据 `groupIdx`、`tileStartInPage` 和 head size 计算 GM 起点。

`CopyKeyToL1`：

- 从 K cache GM 搬当前 KV tile 的当前 KV group。
- 使用 head block 方向 multi-burst 搬运。
- 后续 `LoadKeyToL0B` 将 K 从 L1 装入 L0B。

`CopyValueToL1`：

- 在 BMM1 阶段把 V 从 GM 预取到 L1。
- V 的 L1->L0B 不在 BMM1 内完成，而是在 BMM2 阶段执行。

### 7.2 Mask

mask 为 `FORMAT_FRACTAL_NZ`。`CopyMaskToL1` 在 BMM1 阶段把当前 Q block 和 KV tile 对应的 mask 子块搬到 slot-local mask L1。

写文档版中，mask 的 L1->UB 在 Softmax 阶段执行：

```text
LoadMaskToUb(maskUb, maskL1Slot, realQBlockRows, pageSize)
```

mask 本身不在 GM 中按 GQA 维度展开。`AddMask` 通过 vector stride 在读取时完成 GQA 广播。

## 8. Local Buffer 切分

`InitLocalBuffer()` 根据 tiling 参数切分：

- `queryL1`：单份共享 Q L1
- `keyL1Ping/keyL1Pong`
- `maskL1Ping/maskL1Pong`
- `queryL0APing/queryL0APong`
- `keyL0BPing/keyL0BPong`
- `valueL0BPing/valueL0BPong`
- `scoreL0CPing/scoreL0CPong`
- `probL1Ping/probL1Pong`
- `valueL1Ping/valueL1Pong`
- UB 中的 score/prob/mask/online-softmax 状态/output/work 区

状态语义：

- `mOrgF16Ub/lOrgUb/outUb` 是跨 KV tile 的 online softmax 全局状态。
- `mPageF16Ub/lPageUb` 是 ping tile-local 状态。
- `mPageF16UbPong/lPageUbPong` 是 pong tile-local 状态。
- `scoreF16UbPing/scoreF16UbPong` 是 slot-local score half 缓冲。
- `probF16UbPing/probF16UbPong` 是 slot-local P half 缓冲。

## 9. BMM1 阶段

`Bmm1PingSingle` 和 `Bmm1PongSingle` 只是选择 ping/pong buffer，核心逻辑在 `Bmm1Single`。

`Bmm1Single` 执行顺序：

1. 如当前 KV tile 是当前 Q tile 的首个可见 tile，则执行 `CopyQueryToL1`。
2. 执行 `CopyMaskToL1`，预取当前 tile 的 mask 到 L1。
3. 执行 `LoadQueryToL0A`，将共享 Q L1 装入 slot-local L0A。
4. 执行 `CopyKeyToL1`，搬 K GM->L1。
5. 执行 `LoadKeyToL0B`，搬 K L1->L0B。
6. 执行 `CopyValueToL1`，预取 V GM->L1。
7. 执行 BMM1 MMAD：

```text
score = Q * K
```

BMM1 输出在 `scoreL0C`，其逻辑 shape 为：

```text
[qBlockRows * gqaTileSize, tileSize]
```

注意：该写文档版中 V 只在 BMM1 阶段预取到 L1，真正的 V L1->L0B 在 BMM2 阶段完成。

## 10. Softmax 阶段

`SoftmaxSingle` 对 BMM1 score 执行 scale、mask、局部 softmax 状态构造。

执行顺序：

1. `CopyScoreToUb`：将 `scoreL0C` 从 L0C 搬到 UB，并完成 float->half 转换。
2. `ScaleScoreHalf`：执行 `score *= softmaxScale`。
3. `LoadMaskToUb`：将 BMM1 阶段预取好的 mask 从 L1 搬到 UB。
4. `AddMask`：将 mask 加到 score 上，并在 vector 读取时完成 GQA 广播。
5. `MaskInvalidRows`：将 tail Q block 中无效 folded-M 行置为 softmax 最小值。
6. `BuildSoftmaxLocalState`：构造当前 KV tile 的 softmax 局部状态。

### 10.1 AddMask 的 GQA 广播

当 `gqaTileSize == MAX_GQA_TILE_SIZE` 时，`AddMask` 使用粗粒度 vector repeat 路径：

```text
src1BlkStride = 0
```

这表示同一个 Q row 的 mask block 会在一个 repeat 内复用到该 Q row 对应的多个 GQA head 上。mask 存储仍是 `[qRow, kvCol]`，不会提前物理扩展成 `[qRow, gqaHead, kvCol]`。

### 10.2 BuildSoftmaxLocalState

该函数完成：

```text
mPage = rowmax(score)
mNew  = first_tile ? mPage : max(mPage, mOrg)
score = score - mNew
P     = exp(score)
lPage = rowsum(P)
```

并维护 online softmax 所需的状态：

- 首个 KV tile：`mOrg = mNew`
- 后续 KV tile：`localDelta = mOld - mNew`，随后 `mOrg = mNew`

最后 `CastProbToL1` 将当前 tile 的 half `P` 写到 slot-local `probL1`。

## 11. BMM2 阶段

`Bmm2Single` 执行：

1. `Bmm2LoadValueToL0B`：将 BMM1 阶段已经预取到 L1 的 V 装入 L0B。
2. `Bmm2LoadProbToL0A`：将 Softmax 产生的 P 从 L1 装入 L0A。
3. `Bmm2Mmad`：执行：

```text
PV = P * V
```

BMM2 输出是当前 KV tile 的局部输出分子，存放在 `outL0C`。

## 12. Update 阶段

`UpdateSingle` 将当前 KV tile 的 `PV` 合并进跨 tile 的 online softmax 输出状态。

online softmax 合并公式：

```text
scale = exp(mOld - mNew)
lNew = lOld * scale + lPage
outNumeratorNew = outNumeratorOld * scale + PV
```

执行顺序：

1. `CopyPvChunkToUb`：发起 PV L0C->UB。
2. `UpdateOnlineState`：更新 `lOrg`，并在非首 tile 时按行缩放旧 `outUb`。
3. `WaitAndFoldPvChunk`：等待 PV 到达 UB，并写入或累加到 `outUb`。

如果 V head size 被拆成多个 chunk，只有首个 chunk 前执行 `UpdateOnlineState`。后续 chunk 只是同一个 PV 矩阵的不同列，不会再次更新 `m/l` 行状态。

## 13. Normalize 和写回

所有可见 KV tile 处理完成后，`NormalizeAndWriteOutput` 执行：

```text
output = outNumerator / lOrg
```

步骤：

1. `BroadcastRows`：将 `lOrg` 广播为按行可除的布局。
2. `DivRows`：对 `outUb` 执行按行除法。
3. `Cast`：将 float 输出转换成 half。
4. `DataCopy`：按 Q row 写回当前 GQA tile 对应的 output head 区间。

写回时，每个 burst 对应一个 Q row 中当前 GQA tile 覆盖的 head 区间，`dstStride` 跳过同一 Q row 中不属于当前 GQA tile 的其他 head。

## 14. Event Flag 和生命周期

该版本通过固定 event id 维护 ping/pong 生命周期：

- `Pingflag/Pongflag`：主 ping/pong slot 依赖。
- `PingflagPlus2/PongflagPlus2`：K/P/V 等 L0 相关依赖。
- `PingflagPlus4/PongflagPlus4`：K/V GM->L1 和 L1->L0B 的中间依赖。
- `PingflagPlus6/PongflagPlus6`：V L1 slot 的写保护。

生命周期重点：

- Q L1 是共享缓冲，首次 `CopyQueryToL1` 前等待 ping/pong 两侧的 `MTE1_MTE2` guard。
- 当前 GQA tile 的 KV tile 循环结束后，如果本轮消费过 Q，则统一释放 ping/pong 的 Q guard。
- K、V、mask、score、prob、PV 都按 ping/pong slot 隔离。
- `mOrg/lOrg/outUb` 是当前 Q block/GQA tile 的全局 online softmax 状态，不能和其他 Q block 混用。
- `mPage/lPage/prob/outL0C` 是 tile-local 状态，必须按同一个 ping/pong slot 配对消费。

维护 flag 时必须保证：

- 每个 `WaitFlag` 在所有分支上都有匹配的 `SetFlag`。
- 有 pong tile 和无 pong tile 的路径都能结束当前 slot 生命周期。
- tail Q block、最后一个可见 KV tile、无效 row 等边界路径不遗漏 token。

## 15. 与 ATB 对齐的当前策略

写文档版的主要对齐点：

- Q/K/V/mask 均按 NZ 物理格式输入，减少 kernel 内部 layout 转换。
- Q 从真实 Q NZ 中按 `qTokenStart/qHeadStart/kBlock` 抽取当前 GQA tile。
- K/V cache 通过 `blockTable` 做逻辑 page 到物理 page 映射。
- BMM1 使用 folded-GQA M 维，把一个 Q row 的多个 GQA heads 合并进 M 维。
- Softmax 使用 tile-local `mPage/lPage/P`，并维护跨 KV tile 的 `mOrg/lOrg/outUb`。
- BMM2 使用 `P * V` 得到局部输出分子。
- Update 使用 online softmax 公式合并历史 tile 和当前 tile。

