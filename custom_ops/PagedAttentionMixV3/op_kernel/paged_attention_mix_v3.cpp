#include "kernel_operator.h"

namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t VEC_SIZE_IN_BYTES = 8 * DATA_BLOCK_BYTES;
constexpr uint32_t FLOATS_PER_VECTOR_REPEAT = VEC_SIZE_IN_BYTES / sizeof(float);
constexpr uint32_t HALFS_PER_VECTOR_REPEAT = VEC_SIZE_IN_BYTES / sizeof(half);
constexpr uint32_t CUBE_BLOCK_SIZE = 16;
// Keep the persistent output numerator in the same C0/NZ order produced by
// BMM2: [vBlock][row][C0]. This avoids folding every KV tile into ND. The
// public output remains ND and is converted only once during the final write.
constexpr uint32_t MAX_Q_ROWS_PER_GROUP = 128;
constexpr uint32_t Q_BLOCK_ROWS = MAX_Q_ROWS_PER_GROUP;
constexpr uint32_t CUBE_MATRIX_SIZE = CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
constexpr uint32_t MAX_GQA_TILE_SIZE = 8;
constexpr uint32_t KV_TILE_SIZE = 128;
constexpr uint32_t MAX_V_HEAD_SIZE = 128;
// P6a reserves a bounded L1 arena for immutable KV tiles owned by the first
// batch/KV-head group handled by each AI Core. The tile count is derived from
// the runtime head sizes; no page-size or model-shape special case is used.
constexpr uint32_t KV_REUSE_CACHE_BYTES = 128 * 1024;
constexpr uint32_t MAX_KV_REUSE_CACHE_TILES = 8;
constexpr uint32_t UB_REGION_BYTES = 32 * 1024;
constexpr uint32_t UB_SCORE_PING_OFFSET = 0 * UB_REGION_BYTES;
constexpr uint32_t UB_SCORE_PONG_OFFSET = 1 * UB_REGION_BYTES;
constexpr uint32_t UB_SHARED_OFFSET = 2 * UB_REGION_BYTES;
constexpr uint32_t UB_ROW_STATE_OFFSET = 4 * UB_REGION_BYTES;
constexpr uint32_t UB_WORK_OFFSET = 5 * UB_REGION_BYTES;
// The largest vector workspace is 128 rows x 16 fp32 values = 8 KiB.
// Do not assume ATB's raw-buffer top-of-UB layout is safe under AscendC TPipe.
// On the current 310P build, outUb local row 112 starts at exactly 248 KiB and
// is the only corrupted row in every 128-row slice. Pack output directly after
// the actual workspace so the whole arena ends at 232 KiB.
constexpr uint32_t UB_WORK_BYTES = MAX_Q_ROWS_PER_GROUP * CUBE_BLOCK_SIZE * sizeof(float);
constexpr uint32_t UB_OUTPUT_OFFSET = UB_WORK_OFFSET + UB_WORK_BYTES;
constexpr uint32_t UB_TOTAL_BYTES = UB_OUTPUT_OFFSET +
                                    MAX_Q_ROWS_PER_GROUP * MAX_V_HEAD_SIZE * sizeof(float);
constexpr float SOFTMAX_INIT_MIN = -65504.0F;

static_assert(MAX_KV_REUSE_CACHE_TILES <= 32,
              "KV reuse validity mask supports at most 32 cached tiles");
static_assert(MAX_Q_ROWS_PER_GROUP * KV_TILE_SIZE * sizeof(half) <= UB_REGION_BYTES,
              "fp16 score/probability must fit in one 32 KiB UB slot");
static_assert(MAX_Q_ROWS_PER_GROUP * KV_TILE_SIZE * sizeof(float) <= 2 * UB_REGION_BYTES,
              "fp32 score/PV scratch must fit in the shared 64 KiB UB arena");
static_assert(MAX_Q_ROWS_PER_GROUP * MAX_V_HEAD_SIZE * sizeof(float) <= 2 * UB_REGION_BYTES,
              "persistent fp32 output must fit in the final 64 KiB UB arena");
static_assert(UB_WORK_OFFSET + UB_WORK_BYTES <= UB_OUTPUT_OFFSET,
              "vector workspace must not overlap persistent output");
static_assert(UB_TOTAL_BYTES == 232 * 1024,
              "packed UB arena must stay below the observed 248 KiB collision boundary");

constexpr auto Pingflag = EVENT_ID0;
constexpr auto Pongflag = EVENT_ID1;
constexpr auto PingflagPlus2 = EVENT_ID2;
constexpr auto PongflagPlus2 = EVENT_ID3;
constexpr auto PingflagPlus4 = EVENT_ID4;
constexpr auto PongflagPlus4 = EVENT_ID5;
constexpr auto PingflagPlus6 = EVENT_ID6;
constexpr auto PongflagPlus6 = EVENT_ID7;
// R2+R3 has one physical owner even though mask/prob/PV use different logical tensors.
// EVENT_ID7 is free for the V_MTE1/V_MTE2 event types used by this shared-arena guard.
constexpr auto SharedUbFlag = EVENT_ID7;
// outF16Ub aliases scoreF16UbPong. Protect the interval from the final output
// MTE3 read until the next Vector reuse of the pong score slot without draining
// every pipeline at the end of each task. EVENT_ID6 is free for MTE3_V.
constexpr auto OutputReuseFlag = EVENT_ID6;
}

class PagedAttentionMixV3Kernel {
public:
    __aicore__ inline PagedAttentionMixV3Kernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR attentionMask,
                                GM_ADDR blockTable, GM_ADDR kvLengths, GM_ADDR seqLenPerRequest,
                                GM_ADDR queryRope, GM_ADDR output,
                                const PagedAttentionMixV3TilingData *tilingData)
    {
        queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_QUERY *>(query));
        kCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_KCACHE *>(kCache));
        vCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VCACHE *>(vCache));
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_ATTENTIONMASK *>(attentionMask));
        blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        kvLengthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(kvLengths));
        seqLenGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seqLenPerRequest));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUTPUT *>(output));
        queryNzGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_QUERY *>(query));
        this->tilingData = tilingData;
        (void)queryRope;
        InitLocalBuffer();
    }

    __aicore__ inline void Process()
    {
        if (!ubLayoutValid) {
            return;
        }

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t usedCoreNum = tilingData->usedCoreNum == 0 ? 1U : tilingData->usedCoreNum;
        if (coreIdx >= usedCoreNum) {
            return;
        }

        const uint32_t qHeadNum = tilingData->qHeadNum;
        const uint32_t qkHeadSize = tilingData->qkHeadSize;
        const uint32_t vHeadSize = tilingData->vHeadSize;
        const uint32_t outputTokenStride = qHeadNum * vHeadSize;
        const uint32_t pageSize = tilingData->pageSize;
        const uint32_t pageNumPerBatch = tilingData->pageNumPerBatch;
        const uint32_t gqaGroupSize = tilingData->gqaGroupSize;
        const uint32_t kvTileSize = Min(pageSize, KV_TILE_SIZE);
        const uint32_t tilesPerPage = CeilDiv(pageSize, kvTileSize);
        const uint32_t keyPhysicalPageStride = pageSize * tilingData->kvHeadNum * qkHeadSize;
        const uint32_t valuePhysicalPageStride = pageSize * tilingData->kvHeadNum * vHeadSize;
        const uint32_t keyGroupStride = pageSize * qkHeadSize;
        const uint32_t valueGroupStride = pageSize * vHeadSize;

        uint32_t totalTaskNum = 0;
        uint64_t totalTaskWeight = 0;
        for (uint32_t batchIdx = 0; batchIdx < tilingData->batchSize; ++batchIdx) {
            const uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            const uint32_t kvLen = static_cast<uint32_t>(kvLengthsGm.GetValue(batchIdx));
            const uint32_t qSliceNum = CeilDiv(seqLen, qBlockRows);
            totalTaskNum += qSliceNum * qHeadNum;
            totalTaskWeight += GetPerHeadTaskWeight(seqLen,
                                                    kvLen,
                                                    pageSize,
                                                    kvTileSize,
                                                    tilesPerPage) * qHeadNum;
        }
        if (totalTaskNum == 0 || totalTaskWeight == 0) {
            return;
        }

        // P9A: keep each core's task range contiguous, but split by estimated
        // work instead of task count. A task pays one fixed-cost unit plus one
        // unit for every visible KV tile. This preserves the existing
        // batch->qHead->qSlice order and KV-reuse locality while avoiding the
        // large imbalance between short-context and long-context tasks.
        const uint64_t weightPerCore = totalTaskWeight / usedCoreNum;
        const uint64_t weightRemainder = totalTaskWeight % usedCoreNum;
        const uint64_t targetWeightStart =
            weightPerCore * coreIdx + weightRemainder * coreIdx / usedCoreNum;
        const uint32_t nextCoreIdx = coreIdx + 1;
        const uint64_t targetWeightEnd =
            weightPerCore * nextCoreIdx + weightRemainder * nextCoreIdx / usedCoreNum;
        uint32_t taskIdStart = FindWeightedTaskBoundary(targetWeightStart,
                                                        qHeadNum,
                                                        pageSize,
                                                        kvTileSize,
                                                        tilesPerPage,
                                                        totalTaskNum);
        uint32_t taskIdEnd = FindWeightedTaskBoundary(targetWeightEnd,
                                                      qHeadNum,
                                                      pageSize,
                                                      kvTileSize,
                                                      tilesPerPage,
                                                      totalTaskNum);
        if (taskIdStart >= taskIdEnd) {
            return;
        }

        uint32_t qBatchOffset = 0;
        uint32_t taskBase = 0;
        uint32_t batchIdx = 0;
        for (; batchIdx < tilingData->batchSize; ++batchIdx) {
            const uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            const uint32_t batchTaskNum = CeilDiv(seqLen, qBlockRows) * qHeadNum;
            if (taskIdStart < taskBase + batchTaskNum) {
                break;
            }
            taskBase += batchTaskNum;
            qBatchOffset += seqLen;
        }
        if (batchIdx >= tilingData->batchSize) {
            return;
        }

        uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
        uint32_t kvLen = static_cast<uint32_t>(kvLengthsGm.GetValue(batchIdx));
        uint32_t historyKvLen = kvLen - seqLen;
        uint32_t kvPageNum = CeilDiv(kvLen, pageSize);
        uint32_t taskInBatch = taskIdStart - taskBase;
        uint32_t qSliceNum = CeilDiv(seqLen, qBlockRows);
        uint32_t qHeadIdx = taskInBatch / qSliceNum;
        uint32_t qSliceIdx = taskInBatch % qSliceNum;
        uint32_t qBlockStart = qSliceIdx * qBlockRows;
        uint32_t groupIdx = qHeadIdx / gqaGroupSize;
        uint32_t qHeadInGroup = qHeadIdx - groupIdx * gqaGroupSize;
        uint32_t keyGroupBase = groupIdx * keyGroupStride;
        uint32_t valueGroupBase = groupIdx * valueGroupStride;
        uint32_t qHeadOutputOffset = qHeadIdx * vHeadSize;
        uint32_t blockTableBatchOffset = batchIdx * pageNumPerBatch;
        bool blockTableCacheActive = PrefetchBlockTable(blockTableBatchOffset, kvPageNum);

        /****************** PrimeSingleBufferFlags(); ******************/ 
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PingflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PongflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(PingflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(PongflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(PingflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(PongflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Pingflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Pongflag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus6);
        // The first shared-arena owner is always Softmax Ping's mask load (MTE1).
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
        // Initially the pong score/output slot is free. Each output write returns
        // this token only after MTE3 has finished reading outF16Ub.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OutputReuseFlag);
        bool outputReusePending = true;
        // Keep the cache immutable after choosing its owner. This avoids any
        // cache-overwrite synchronization in the hot path while still reusing
        // K/V across consecutive q-head/q-slice tasks of the same KV group.
        bool kvReuseOwnerValid = false;
        uint32_t kvReuseOwnerBatch = 0;
        uint32_t kvReuseOwnerGroup = 0;
        uint32_t kvReuseValidMask = 0;
        
        for (uint32_t taskId = taskIdStart; taskId < taskIdEnd; ++taskId) {
            const uint32_t realQBlockRows = Min(qBlockRows, seqLen - qBlockStart);
            {
                constexpr uint32_t curGqaTileSize = 1;
                const uint32_t mActual = realQBlockRows;
                const uint32_t roundM = (mActual + CUBE_BLOCK_SIZE - 1) & ~(CUBE_BLOCK_SIZE - 1);
                const uint32_t qTokenStart = qBatchOffset + qBlockStart;
                const uint32_t qHeadStart = qHeadIdx;
                // The first visible KV tile overwrites O/l/m, so no vector init is needed here.

                const uint32_t qHistoryStart = historyKvLen + qBlockStart;
                const uint32_t visibleKvEnd = qHistoryStart + realQBlockRows;
                const uint32_t totalKvTiles = kvPageNum * tilesPerPage;
                const uint32_t maskRowOffset = qTokenStart * CUBE_BLOCK_SIZE;
                const uint32_t outputOffset = qTokenStart * outputTokenStride + qHeadOutputOffset;
                uint32_t cursorPageIdx = 0;
                uint32_t cursorTileStartInPage = 0;
                bool queryGuardConsumed = false;

                if (!kvReuseOwnerValid && kvReuseCacheTileCount != 0) {
                    kvReuseOwnerValid = true;
                    kvReuseOwnerBatch = batchIdx;
                    kvReuseOwnerGroup = groupIdx;
                }
                const bool isKvReuseOwner = kvReuseOwnerValid &&
                                            batchIdx == kvReuseOwnerBatch &&
                                            groupIdx == kvReuseOwnerGroup;

                for (uint32_t tilePairStart = 0; tilePairStart < totalKvTiles; tilePairStart += 2) {
                    KvTileInfo pingTile;
                    if (!BuildKvTileInfoAtPosition(pingTile,
                                                   cursorPageIdx,
                                                   cursorTileStartInPage,
                                                   kvTileSize,
                                                   pageSize,
                                                   visibleKvEnd,
                                                   maskRowOffset)) {
                        break;
                    }
                    AdvanceKvTileCursor(cursorPageIdx,
                                        cursorTileStartInPage,
                                        kvTileSize,
                                        pageSize);

                    KvTileInfo pongTile;
                    const bool hasPongTile = tilePairStart + 1 < totalKvTiles &&
                                             BuildKvTileInfoAtPosition(pongTile,
                                                                       cursorPageIdx,
                                                                       cursorTileStartInPage,
                                                                       kvTileSize,
                                                                       pageSize,
                                                                       visibleKvEnd,
                                                                       maskRowOffset);
                    if (hasPongTile) {
                        AdvanceKvTileCursor(cursorPageIdx,
                                            cursorTileStartInPage,
                                            kvTileSize,
                                            pageSize);
                    }

                    const bool usePingKvReuse = isKvReuseOwner &&
                                                tilePairStart < kvReuseCacheTileCount;
                    const uint32_t pingKvReuseBit = usePingKvReuse ? (1U << tilePairStart) : 0U;
                    const bool fillPingKvReuse = usePingKvReuse &&
                                                 (kvReuseValidMask & pingKvReuseBit) == 0;
                    if (fillPingKvReuse) {
                        kvReuseValidMask |= pingKvReuseBit;
                    }
                    const uint32_t pongTileIdx = tilePairStart + 1;
                    const bool usePongKvReuse = hasPongTile && isKvReuseOwner &&
                                                pongTileIdx < kvReuseCacheTileCount;
                    const uint32_t pongKvReuseBit = usePongKvReuse ? (1U << pongTileIdx) : 0U;
                    const bool fillPongKvReuse = usePongKvReuse &&
                                                 (kvReuseValidMask & pongKvReuseBit) == 0;
                    if (fillPongKvReuse) {
                        kvReuseValidMask |= pongKvReuseBit;
                    }

                    /****** ATB: Bmm1 Ping Start ******/
                    const bool isFirstKvTile = tilePairStart == 0;
                    queryGuardConsumed = queryGuardConsumed || isFirstKvTile;
                    const bool loadPingKvFromGm = !usePingKvReuse || fillPingKvReuse;
                    uint32_t pingKeyOffset = 0;
                    uint32_t pingValueOffset = 0;
                    if (loadPingKvFromGm) {
                        const uint32_t physicalPageIdx = GetPhysicalPageIdx(
                            pingTile.pageIdxInSeq,
                            blockTableBatchOffset,
                            blockTableCacheActive);
                        const uint32_t tileOffset = pingTile.tileStartInPage * CUBE_BLOCK_SIZE;
                        pingKeyOffset = physicalPageIdx * keyPhysicalPageStride + keyGroupBase + tileOffset;
                        pingValueOffset = physicalPageIdx * valuePhysicalPageStride + valueGroupBase + tileOffset;
                    }
                    Bmm1PingSingle(isFirstKvTile,
                                   isFirstKvTile,
                                   hasPongTile,
                                   qTokenStart,
                                   qHeadStart,
                                   realQBlockRows,
                                   mActual,
                                   roundM,
                                   qkHeadSize,
                                   curGqaTileSize,
                                   pingTile.maskOffset,
                                   pingKeyOffset,
                                   pingValueOffset,
                                   pageSize,
                                   pingTile.tileSize,
                                   vHeadSize,
                                   usePingKvReuse,
                                   loadPingKvFromGm,
                                   tilePairStart);
                    if (hasPongTile) {
                        const bool loadPongKvFromGm = !usePongKvReuse || fillPongKvReuse;
                        uint32_t pongKeyOffset = 0;
                        uint32_t pongValueOffset = 0;
                        if (loadPongKvFromGm) {
                            const uint32_t physicalPageIdx = GetPhysicalPageIdx(
                                pongTile.pageIdxInSeq,
                                blockTableBatchOffset,
                                blockTableCacheActive);
                            const uint32_t tileOffset = pongTile.tileStartInPage * CUBE_BLOCK_SIZE;
                            pongKeyOffset = physicalPageIdx * keyPhysicalPageStride + keyGroupBase + tileOffset;
                            pongValueOffset = physicalPageIdx * valuePhysicalPageStride + valueGroupBase + tileOffset;
                        }
                        /****** ATB: Bmm1 Pong Starts ******/
                        Bmm1PongSingle(false,
                                       isFirstKvTile,
                                       false,
                                       qTokenStart,
                                       qHeadStart,
                                       realQBlockRows,
                                       mActual,
                                       roundM,
                                       qkHeadSize,
                                       curGqaTileSize,
                                       pongTile.maskOffset,
                                       pongKeyOffset,
                                       pongValueOffset,
                                       pageSize,
                                       pongTile.tileSize,
                                       vHeadSize,
                                       usePongKvReuse,
                                       loadPongKvFromGm,
                                       pongTileIdx);
                    }

                    /****** ATB: Softmax Ping Starts ******/
                    // R2+R3 owner sequence for one pair is:
                    // Softmax Ping -> Softmax Pong (optional) -> Update Ping -> Update Pong (optional).
                    // Mask is written by MTE1, while PV is written by MTE2; both are consumed by V.
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
                    SoftmaxPingSingle(qHistoryStart,
                                      pingTile.tilePageStart,
                                      realQBlockRows,
                                      mActual,
                                      roundM,
                                      curGqaTileSize,
                                      pingTile.tileSize,
                                      isFirstKvTile);
                    if (hasPongTile) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
                        /****** ATB: Softmax Pong Starts ******/
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
                        if (outputReusePending) {
                            // scoreF16UbPong aliases the previous task's output source.
                            // Delay only this first Vector reuse until MTE3 has read it.
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OutputReuseFlag);
                            outputReusePending = false;
                        }
                        SoftmaxPongSingle(qHistoryStart,
                                          pongTile.tilePageStart,
                                          realQBlockRows,
                                          mActual,
                                          roundM,
                                          curGqaTileSize,
                                          pongTile.tileSize,
                                          false);
                    }
                    // The next owner is always Update Ping, whose L0C->UB copy uses MTE2.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SharedUbFlag);

                    /****** ATB: Bmm2 Ping Starts ******/
                    Bmm2PingSingle(mActual,
                                   roundM,
                                   pingTile.tileSize,
                                   vHeadSize,
                                   usePingKvReuse,
                                   tilePairStart);     //* P*V
                    if (hasPongTile) {
                        /****** ATB: Bmm2 Pong Starts ******/
                        Bmm2PongSingle(mActual,
                                       roundM,
                                       pongTile.tileSize,
                                       vHeadSize,
                                       usePongKvReuse,
                                       pongTileIdx);     //* P*V
                    }

                    /****** ATB: Update Ping Starts ******/
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SharedUbFlag);
                    UpdatePingSingle(mActual,
                                     roundM,
                                     vHeadSize,
                                     isFirstKvTile);     //* 更新UB，同时累加到总out上
                    if (hasPongTile) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SharedUbFlag);
                        /****** ATB: Update Pong Starts ******/
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SharedUbFlag);
                        UpdatePongSingle(mActual,
                                         roundM,
                                         vHeadSize,
                                         false);     //* 更新UB，同时累加到总out上
                    }
                    // Every pair hands the arena back to the next Softmax Ping mask load.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
                    if (!hasPongTile) {
                        break;
                    }
                }
                if (queryGuardConsumed) {
                    // Match ATB's shared-Q lifecycle: the first Q GM->L1 load waits both ping/pong write guards,
                    // then ping/pong L1->L0A consumers release both guards before the next GQA tile reloads Q.
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Pingflag);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Pongflag);
                }

                if (outputReusePending) {
                    // A one-tile task never touched the pong score slot. Consume
                    // the previous output token before casting into the alias.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OutputReuseFlag);
                    outputReusePending = false;
                }
                NormalizeAndWriteOutput(outputOffset,         //* 此时得到了总l，因此处理l得到最终的输出
                                        mActual,
                                        roundM,
                                        qHeadNum,
                                        vHeadSize);
                outputReusePending = true;
            }

            if (taskId + 1 >= taskIdEnd) {
                // return;
                break;
            }
            ++qSliceIdx;
            if (qSliceIdx < qSliceNum) {
                qBlockStart += qBlockRows;
                continue;
            }
            qSliceIdx = 0;
            qBlockStart = 0;
            ++qHeadIdx;
            if (qHeadIdx < qHeadNum) {
                qHeadOutputOffset += vHeadSize;
                ++qHeadInGroup;
                if (qHeadInGroup == gqaGroupSize) {
                    qHeadInGroup = 0;
                    ++groupIdx;
                    keyGroupBase += keyGroupStride;
                    valueGroupBase += valueGroupStride;
                }
                continue;
            }

            qHeadIdx = 0;
            groupIdx = 0;
            qHeadInGroup = 0;
            keyGroupBase = 0;
            valueGroupBase = 0;
            qHeadOutputOffset = 0;
            qBatchOffset += seqLen;
            ++batchIdx;
            blockTableBatchOffset += pageNumPerBatch;
            if (batchIdx >= tilingData->batchSize) {
                // return;
                break;
            }
            seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            kvLen = static_cast<uint32_t>(kvLengthsGm.GetValue(batchIdx));
            historyKvLen = kvLen - seqLen;
            kvPageNum = CeilDiv(kvLen, pageSize);
            blockTableCacheActive = PrefetchBlockTable(blockTableBatchOffset, kvPageNum);
            qSliceNum = CeilDiv(seqLen, qBlockRows);
        }

        /****************** SyncEnd(): drain flags paired with the initial priming above. ******************/
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PingflagPlus6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(PongflagPlus6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(PingflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(PongflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(PingflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(PongflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Pongflag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PingflagPlus2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PongflagPlus2);
        // Consume the token returned by the final Update (or the initial token if there was no tile).
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(SharedUbFlag);
        if (outputReusePending) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OutputReuseFlag);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    struct KvTileInfo {
        uint32_t tileStartInPage;
        uint32_t tileSize;
        uint32_t tilePageStart;
        uint32_t maskOffset;
        uint32_t pageIdxInSeq;
    };

    __aicore__ inline uint32_t CeilDiv(uint32_t lhs, uint32_t rhs) const
    {
        return (lhs + rhs - 1) / rhs;
    }

    __aicore__ inline uint64_t GetSliceTaskWeight(uint32_t seqLen,
                                                    uint32_t kvLen,
                                                    uint32_t qSliceIdx,
                                                    uint32_t pageSize,
                                                    uint32_t kvTileSize,
                                                    uint32_t tilesPerPage) const
    {
        const uint32_t qBlockStart = qSliceIdx * qBlockRows;
        const uint32_t realQBlockRows = Min(qBlockRows, seqLen - qBlockStart);
        const uint32_t visibleKvEnd = kvLen - seqLen + qBlockStart + realQBlockRows;
        const uint32_t fullPageNum = visibleKvEnd / pageSize;
        const uint32_t tailInPage = visibleKvEnd - fullPageNum * pageSize;
        const uint32_t visibleKvTileCount =
            fullPageNum * tilesPerPage +
            (tailInPage == 0 ? 0U : CeilDiv(tailInPage, kvTileSize));
        // One unit represents fixed per-task work (Q load, normalize and ND
        // write); the remaining units track BMM/softmax/update work per tile.
        return static_cast<uint64_t>(visibleKvTileCount) + 1U;
    }

    __aicore__ inline uint64_t GetPerHeadTaskWeight(uint32_t seqLen,
                                                      uint32_t kvLen,
                                                      uint32_t pageSize,
                                                      uint32_t kvTileSize,
                                                      uint32_t tilesPerPage) const
    {
        const uint32_t qSliceNum = CeilDiv(seqLen, qBlockRows);
        uint64_t perHeadWeight = 0;
        for (uint32_t qSliceIdx = 0; qSliceIdx < qSliceNum; ++qSliceIdx) {
            perHeadWeight += GetSliceTaskWeight(seqLen,
                                                kvLen,
                                                qSliceIdx,
                                                pageSize,
                                                kvTileSize,
                                                tilesPerPage);
        }
        return perHeadWeight;
    }

    __aicore__ inline uint32_t FindWeightedTaskBoundary(uint64_t targetWeight,
                                                           uint32_t qHeadNum,
                                                           uint32_t pageSize,
                                                           uint32_t kvTileSize,
                                                           uint32_t tilesPerPage,
                                                           uint32_t totalTaskNum)
    {
        if (targetWeight == 0) {
            return 0;
        }

        uint64_t weightBeforeBatch = 0;
        uint32_t taskBase = 0;
        for (uint32_t batchIdx = 0; batchIdx < tilingData->batchSize; ++batchIdx) {
            const uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            const uint32_t kvLen = static_cast<uint32_t>(kvLengthsGm.GetValue(batchIdx));
            const uint32_t qSliceNum = CeilDiv(seqLen, qBlockRows);
            const uint64_t perHeadWeight = GetPerHeadTaskWeight(seqLen,
                                                               kvLen,
                                                               pageSize,
                                                               kvTileSize,
                                                               tilesPerPage);
            const uint64_t batchWeight = perHeadWeight * qHeadNum;
            const uint32_t batchTaskNum = qSliceNum * qHeadNum;
            if (targetWeight > weightBeforeBatch + batchWeight) {
                weightBeforeBatch += batchWeight;
                taskBase += batchTaskNum;
                continue;
            }

            const uint64_t targetInBatch = targetWeight - weightBeforeBatch;
            const uint32_t qHeadIdx = static_cast<uint32_t>(targetInBatch / perHeadWeight);
            if (qHeadIdx >= qHeadNum) {
                return taskBase + batchTaskNum;
            }

            const uint64_t targetInHead = targetInBatch - perHeadWeight * qHeadIdx;
            uint64_t weightBeforeSlice = 0;
            const uint32_t headTaskBase = taskBase + qHeadIdx * qSliceNum;
            for (uint32_t qSliceIdx = 0; qSliceIdx < qSliceNum; ++qSliceIdx) {
                const uint64_t sliceWeight = GetSliceTaskWeight(seqLen,
                                                               kvLen,
                                                               qSliceIdx,
                                                               pageSize,
                                                               kvTileSize,
                                                               tilesPerPage);
                const uint64_t weightAfterSlice = weightBeforeSlice + sliceWeight;
                if (targetInHead <= weightAfterSlice) {
                    const uint64_t distanceToPrevious = targetInHead - weightBeforeSlice;
                    const uint64_t distanceToNext = weightAfterSlice - targetInHead;
                    return headTaskBase + qSliceIdx +
                           (distanceToNext < distanceToPrevious ? 1U : 0U);
                }
                weightBeforeSlice = weightAfterSlice;
            }
            return headTaskBase + qSliceNum;
        }
        return totalTaskNum;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t RoundUp(uint32_t value, uint32_t alignment) const
    {
        return CeilDiv(value, alignment) * alignment;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline void AdvanceKvTileCursor(uint32_t &pageIdxInSeq,
                                                    uint32_t &tileStartInPage,
                                                    uint32_t kvTileSize,
                                                    uint32_t pageSize) const
    {
        tileStartInPage += kvTileSize;
        if (tileStartInPage >= pageSize) {
            ++pageIdxInSeq;
            tileStartInPage = 0;
        }
    }

    __aicore__ inline bool BuildKvTileInfoAtPosition(KvTileInfo &tileInfo,
                                                     uint32_t pageIdxInSeq,
                                                     uint32_t tileStartInPage,
                                                     uint32_t kvTileSize,
                                                     uint32_t pageSize,
                                                     uint32_t visibleKvEnd,
                                                     uint32_t maskRowOffset)
    {
        const uint32_t tilePageStart = pageIdxInSeq * pageSize + tileStartInPage;
        if (tilePageStart >= visibleKvEnd) {
            return false;
        }

        tileInfo.tileStartInPage = tileStartInPage;
        tileInfo.tileSize = Min(kvTileSize, pageSize - tileStartInPage);
        tileInfo.tilePageStart = tilePageStart;
        tileInfo.maskOffset =
            ((tilePageStart >> 4) * tilingData->maskRowSize << 4) + maskRowOffset;
        tileInfo.pageIdxInSeq = pageIdxInSeq;
        return true;
    }

    __aicore__ inline bool PrefetchBlockTable(uint32_t blockTableBatchOffset,
                                               uint32_t kvPageNum)
    {
        if (kvPageNum == 0 || kvPageNum > blockTableCacheCapacity) {
            return false;
        }

        constexpr uint32_t entriesPerDataBlock = DATA_BLOCK_BYTES / sizeof(int32_t);
        const uint32_t alignedEntryCount = kvPageNum / entriesPerDataBlock * entriesPerDataBlock;
        if (alignedEntryCount != 0) {
            // The previous batch may still have issued scalar reads from this UB
            // cache. Drain S before MTE2 overwrites it, then wait until the bulk
            // GM->UB copy is visible to scalar GetValue calls in the task loop.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(Pingflag);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(Pingflag);
            AscendC::DataCopy(blockTableUb,
                              blockTableGm[blockTableBatchOffset],
                              AscendC::DataCopyParams(
                                  1,
                                  static_cast<uint16_t>(alignedEntryCount / entriesPerDataBlock),
                                  0,
                                  0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(Pingflag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(Pingflag);
        }

        // Handle a sub-32-byte tail without reading beyond the last batch's
        // block-table allocation. This runs once per batch, not once per KV tile.
        for (uint32_t pageIdx = alignedEntryCount; pageIdx < kvPageNum; ++pageIdx) {
            blockTableUb.SetValue(
                pageIdx,
                blockTableGm.GetValue(blockTableBatchOffset + pageIdx));
        }
        return true;
    }

    __aicore__ inline uint32_t GetPhysicalPageIdx(uint32_t pageIdxInSeq,
                                                  uint32_t blockTableBatchOffset,
                                                  bool blockTableCacheActive)
    {
        if (blockTableCacheActive) {
            return static_cast<uint32_t>(blockTableUb.GetValue(pageIdxInSeq));
        }
        return static_cast<uint32_t>(
            blockTableGm.GetValue(blockTableBatchOffset + pageIdxInSeq));
    }

    __aicore__ inline void InitLocalBuffer()
    {   // 根据 tiling 参数，把这个 kernel 后面要用到的 L1/L0/UB 空间全部申请并切好
        qBlockRows = Q_BLOCK_ROWS;
        if (qBlockRows == 0 ||
            qBlockRows > MAX_Q_ROWS_PER_GROUP ||
            qBlockRows % CUBE_BLOCK_SIZE != 0) {
            return;
        }
        constexpr uint32_t gqaTileSize = 1;

        const uint32_t l1QueryBytes = qBlockRows * gqaTileSize * tilingData->qkHeadSize * sizeof(DTYPE_QUERY);
        pipe.InitBuffer(queryL1Buf, l1QueryBytes);
        queryL1 = queryL1Buf.Get<DTYPE_QUERY>();

        const uint32_t kvTileSize = Min(tilingData->pageSize, KV_TILE_SIZE);

        const uint32_t l1KeyElements = kvTileSize * tilingData->qkHeadSize;
        const uint32_t l1KeyBytes = l1KeyElements * sizeof(DTYPE_KCACHE);
        pipe.InitBuffer(keyL1Buf, 2 * l1KeyBytes);
        keyL1Ping = keyL1Buf.Get<DTYPE_KCACHE>();
        keyL1Pong = keyL1Ping[l1KeyElements];

        const uint32_t l1MaskElements = qBlockRows * kvTileSize;
        const uint32_t l1MaskBytes = l1MaskElements * sizeof(DTYPE_ATTENTIONMASK);
        pipe.InitBuffer(maskL1Buf, 2 * l1MaskBytes);
        maskL1Ping = maskL1Buf.Get<DTYPE_ATTENTIONMASK>();
        maskL1Pong = maskL1Ping[l1MaskElements];

        const uint32_t mSize = qBlockRows * gqaTileSize;
        const uint32_t scoreSize = mSize * kvTileSize;
        const uint32_t outSize = mSize * tilingData->vHeadSize;
        const uint32_t outF16Size = Max(outSize, scoreSize);
        const uint32_t l0aElements = mSize * Max(tilingData->qkHeadSize, kvTileSize);
        const uint32_t l0aBytes = l0aElements * sizeof(DTYPE_QUERY);
        const uint32_t l0bElements = kvTileSize * Max(tilingData->qkHeadSize, tilingData->vHeadSize);
        const uint32_t l0bBytes = l0bElements * sizeof(DTYPE_KCACHE);
        const uint32_t l0cElements = mSize * Max(kvTileSize, tilingData->vHeadSize);
        const uint32_t l0cBytes = l0cElements * sizeof(float);
        const uint32_t l1ProbElements = scoreSize;
        const uint32_t l1ProbBytes = l1ProbElements * sizeof(half);
        const uint32_t l1ValueElements = kvTileSize * tilingData->vHeadSize;
        const uint32_t l1ValueBytes = l1ValueElements * sizeof(DTYPE_VCACHE);
        const uint32_t kvReuseTileBytes = l1KeyBytes + l1ValueBytes;
        kvReuseCacheTileCount = Min(KV_REUSE_CACHE_BYTES / kvReuseTileBytes,
                                    MAX_KV_REUSE_CACHE_TILES);
        kvReuseKeyTileElements = l1KeyElements;
        kvReuseValueTileElements = l1ValueElements;
        const uint32_t rowSize = mSize;
        const uint32_t maskSize = qBlockRows * kvTileSize;
        const uint32_t workSize = Max(kvTileSize * CUBE_BLOCK_SIZE, mSize * CUBE_BLOCK_SIZE);
        const uint32_t scoreSlotBytes = scoreSize * sizeof(half);
        const uint32_t sharedScratchBytes = scoreSize * sizeof(float);
        const uint32_t outBytes = outSize * sizeof(float);
        const uint32_t outF16Bytes = outF16Size * sizeof(half);
        const uint32_t rowStateBytes = rowSize * (4 * sizeof(half) + 5 * sizeof(float));
        const uint32_t blockTableUbOffset =
            UB_ROW_STATE_OFFSET + RoundUp(rowStateBytes, DATA_BLOCK_BYTES);
        blockTableCacheCapacity = blockTableUbOffset <= UB_WORK_OFFSET
                                      ? (UB_WORK_OFFSET - blockTableUbOffset) / sizeof(int32_t)
                                      : 0;
        const uint32_t maskBytes = maskSize * sizeof(DTYPE_ATTENTIONMASK);
        const uint32_t workBytes = workSize * sizeof(float);

        // P1c established the fixed ATB-style arena and shared R2+R3 owner protocol.
        // P2 changes only the row/task semantics: one task owns one Q head and one
        // qBlockRows-token slice, so every active matrix row is a token row.
        ubLayoutValid = scoreSlotBytes <= UB_REGION_BYTES &&
                        outF16Bytes <= UB_REGION_BYTES &&
                        sharedScratchBytes <= 2 * UB_REGION_BYTES &&
                        maskBytes <= 2 * UB_REGION_BYTES &&
                        rowStateBytes <= UB_REGION_BYTES &&
                        workBytes <= UB_WORK_BYTES &&
                        outBytes <= 2 * UB_REGION_BYTES;

        pipe.InitBuffer(queryL0ABuf, 2 * l0aBytes);
        pipe.InitBuffer(keyL0BBuf, 2 * l0bBytes);
        pipe.InitBuffer(scoreL0CBuf, 2 * l0cBytes);
        pipe.InitBuffer(probL1Buf, 2 * l1ProbBytes);
        pipe.InitBuffer(valueL1Buf, 2 * l1ValueBytes);
        // Reserve a fixed, parameter-independent L1 arena. A tile may be too
        // large to cache, in which case kvReuseCacheTileCount is zero and the
        // normal ping/pong path remains active. InitBuffer must not receive a
        // runtime-computed zero-byte size.
        pipe.InitBuffer(kvReuseL1Buf, KV_REUSE_CACHE_BYTES);
        pipe.InitBuffer(ubBuf, UB_TOTAL_BYTES);

        queryL0APing = queryL0ABuf.Get<DTYPE_QUERY>();
        queryL0APong = queryL0APing[l0aElements];
        probL0APing = queryL0ABuf.Get<half>();
        probL0APong = probL0APing[l0aElements];
        keyL0BPing = keyL0BBuf.Get<DTYPE_KCACHE>();
        keyL0BPong = keyL0BPing[l0bElements];
        valueL0BPing = keyL0BBuf.Get<DTYPE_VCACHE>();
        valueL0BPong = valueL0BPing[l0bElements];
        scoreL0CPing = scoreL0CBuf.Get<float>();
        scoreL0CPong = scoreL0CPing[l0cElements];
        outL0CPing = scoreL0CBuf.Get<float>();
        outL0CPong = outL0CPing[l0cElements];
        ubBase = ubBuf.Get<uint8_t>();

        // R0/R1: ping and pong fp16 score slots. Probability overwrites score only
        // after BuildProbFromScore has consumed the complete score into fp32 scratch.
        scoreF16UbPing = ubBase[UB_SCORE_PING_OFFSET].ReinterpretCast<half>();
        probF16UbPing = scoreF16UbPing;
        scoreF16UbPong = ubBase[UB_SCORE_PONG_OFFSET].ReinterpretCast<half>();
        probF16UbPong = scoreF16UbPong;
        outF16Ub = scoreF16UbPong;

        // R2+R3: one physical shared arena. Softmax loads one mask here, overwrites it
        // with fp32 probability scratch, and Update later overwrites it with one PV chunk.
        probUb = ubBase[UB_SHARED_OFFSET].ReinterpretCast<float>();
        maskF16UbPing = ubBase[UB_SHARED_OFFSET].ReinterpretCast<DTYPE_ATTENTIONMASK>();
        maskF16UbPong = maskF16UbPing;

        // R4: persistent row state only.
        uint32_t rowOffset = UB_ROW_STATE_OFFSET;
        mOrgF16Ub = ubBase[rowOffset].ReinterpretCast<half>();
        rowOffset += rowSize * sizeof(half);
        mPageF16Ub = ubBase[rowOffset].ReinterpretCast<half>();
        rowOffset += rowSize * sizeof(half);
        mPageF16UbPong = ubBase[rowOffset].ReinterpretCast<half>();
        rowOffset += rowSize * sizeof(half);
        mNewF16Ub = ubBase[rowOffset].ReinterpretCast<half>();
        rowOffset += rowSize * sizeof(half);
        mScaleUb = ubBase[rowOffset].ReinterpretCast<float>();
        rowOffset += rowSize * sizeof(float);
        lOrgUb = ubBase[rowOffset].ReinterpretCast<float>();
        rowOffset += rowSize * sizeof(float);
        lPageUb = ubBase[rowOffset].ReinterpretCast<float>();
        rowOffset += rowSize * sizeof(float);
        lPageUbPong = ubBase[rowOffset].ReinterpretCast<float>();
        rowOffset += rowSize * sizeof(float);
        lNewUb = ubBase[rowOffset].ReinterpretCast<float>();
        // R4 has substantial slack after the row state. Stage the current
        // batch's tiny logical->physical page map here so every q-head/q-slice
        // task can resolve K/V pages from UB instead of issuing scalar GM loads.
        blockTableUb = ubBase[blockTableUbOffset].ReinterpretCast<int32_t>();

        // R5 low 8 KiB: vector workspace; the following packed 64 KiB:
        // cross-KV-tile fp32 output numerator. Keeping it below 232 KiB avoids
        // the observed 248 KiB address collision that corrupted local row 112.
        workUb = ubBase[UB_WORK_OFFSET].ReinterpretCast<float>();
        workF16Ub = ubBase[UB_WORK_OFFSET].ReinterpretCast<half>();
        outUb = ubBase[UB_OUTPUT_OFFSET].ReinterpretCast<float>();
        probL1Ping = probL1Buf.Get<half>();
        probL1Pong = probL1Ping[l1ProbElements];
        valueL1Ping = valueL1Buf.Get<DTYPE_VCACHE>();
        valueL1Pong = valueL1Ping[l1ValueElements];
        // Split the cache arena in bytes rather than in K/V elements. This is
        // correct even when the K and V cache element types have different sizes.
        AscendC::LocalTensor<uint8_t> kvReuseBase = kvReuseL1Buf.Get<uint8_t>();
        kvReuseKeyL1 = kvReuseBase.ReinterpretCast<DTYPE_KCACHE>();
        kvReuseValueL1 =
            kvReuseBase[kvReuseCacheTileCount * l1KeyBytes].ReinterpretCast<DTYPE_VCACHE>();
    }

    __aicore__ inline uint32_t NzMatrixElementCount(uint32_t rowCount, uint32_t colCount) const
    {
        return CeilDiv(rowCount, CUBE_BLOCK_SIZE) * colCount * CUBE_BLOCK_SIZE;
    }

    __aicore__ inline uint32_t QueryGroupNzElementCount(uint32_t seqLen, uint32_t qkHeadSize) const
    {
        const uint32_t gqaGroupSize = tilingData->gqaGroupSize;
        const uint32_t gqaTileLimit = Min(gqaGroupSize, MAX_GQA_TILE_SIZE);
        uint32_t groupSize = 0;
        for (uint32_t tileStart = 0; tileStart < gqaGroupSize; tileStart += gqaTileLimit) {
            const uint32_t tileSize = Min(gqaTileLimit, gqaGroupSize - tileStart);
            groupSize += NzMatrixElementCount(seqLen * tileSize, qkHeadSize);
        }

        return groupSize;
    }

    __aicore__ inline uint32_t QueryBatchNzElementCount(uint32_t seqLen, uint32_t qkHeadSize) const
    {
        return QueryGroupNzElementCount(seqLen, qkHeadSize) * tilingData->kvHeadNum;
    }

    //* query is passed as FORMAT_FRACTAL_NZ from head-major [head, token, dim] Q.
    //* GM physical order is [head][kBlock][token][c0]. In P3 gqaTileSize is one;
    //* roundM is the local physical row stride, while GM contributes only mActual rows.
    __aicore__ inline void CopyQueryToL1(uint32_t qTokenStart,
                                         uint32_t qHeadStart,
                                         uint32_t realQBlockRows,
                                         uint32_t roundM,
                                         uint32_t qkHeadSize,
                                         uint32_t gqaTileSize)
    {
        const uint32_t qkHeadBlocks = qkHeadSize / CUBE_BLOCK_SIZE;
        // torch_npu physical FRACTAL_NZ pads the token/M storage stride to C0.
        // Keep logical numTokens for bounds and token offsets, but use the padded
        // physical row count between K blocks and Q heads.  This is observable for
        // decode: logical [qHead, 1, headDim] has a 16-row physical head stride.
        const uint32_t numTokens = tilingData->numTokensPad;
        const uint32_t physicalTokenRows = RoundUp(numTokens, CUBE_BLOCK_SIZE);
        const uint32_t headStride = physicalTokenRows * qkHeadSize;
        const uint32_t kBlockStride = physicalTokenRows * CUBE_BLOCK_SIZE;
        const uint32_t blockLen = CUBE_BLOCK_SIZE * sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES;
        const uint32_t dstStride = (gqaTileSize - 1) * CUBE_BLOCK_SIZE *
                                   sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES;
        constexpr uint32_t DATA_COPY_PARAM_LIMIT = 65535;

        if (gqaTileSize == 0 ||
            qHeadStart + gqaTileSize > tilingData->qHeadNum ||
            qkHeadSize % CUBE_BLOCK_SIZE != 0 ||
            qTokenStart + realQBlockRows > numTokens ||
            realQBlockRows > DATA_COPY_PARAM_LIMIT ||
            blockLen > DATA_COPY_PARAM_LIMIT ||
            dstStride > DATA_COPY_PARAM_LIMIT) {
            return;
        }

        // Keep M=1 in the same NZ layout as every other tail.  The previous
        // packed-vector shortcut copied only the valid row from each K block and
        // then relied on a VECTOR->VECTOR L1->L0A interpretation.  That sequence
        // is not equivalent to the already-validated NZ->ZZ path in this AscendC
        // implementation when the whole invocation contains only one Q row.

        // P10A: a per-q-head task has one head in this path.  Physical NZ stores
        // consecutive K/C0 blocks with physicalTokenRows between them, while L1
        // stores the same blocks with roundM between them.  Express that as one
        // multi-burst GM->L1 copy instead of issuing one DataCopy per K block.
        // This matches ATB's prefill Q-load geometry and removes repeated MTE2
        // command setup without changing either the GM or L1 layout.
        const uint32_t srcKBlockStride = physicalTokenRows - realQBlockRows;
        const uint32_t dstKBlockStride = roundM - realQBlockRows;
        if (gqaTileSize == 1 &&
            qkHeadBlocks <= DATA_COPY_PARAM_LIMIT &&
            realQBlockRows <= DATA_COPY_PARAM_LIMIT &&
            srcKBlockStride <= DATA_COPY_PARAM_LIMIT &&
            dstKBlockStride <= DATA_COPY_PARAM_LIMIT) {
            const uint32_t srcOffset = qHeadStart * headStride + qTokenStart * CUBE_BLOCK_SIZE;
            AscendC::DataCopy(queryL1,
                              queryNzGm[srcOffset],
                              AscendC::DataCopyParams(static_cast<uint16_t>(qkHeadBlocks),
                                                      static_cast<uint16_t>(realQBlockRows),
                                                      static_cast<uint16_t>(srcKBlockStride),
                                                      static_cast<uint16_t>(dstKBlockStride)));
            return;
        }

        // Retain the general folded-head/large-stride geometry even though the
        // current per-q-head task mapping normally takes the coalesced path.
        for (uint32_t kBlock = 0; kBlock < qkHeadBlocks; ++kBlock) {
            for (uint32_t tileHead = 0; tileHead < gqaTileSize; ++tileHead) {
                const uint32_t qHeadId = qHeadStart + tileHead;
                const uint32_t srcOffset =
                    qHeadId * headStride + kBlock * kBlockStride + qTokenStart * CUBE_BLOCK_SIZE;
                const uint32_t dstOffset = kBlock * roundM * CUBE_BLOCK_SIZE + tileHead * CUBE_BLOCK_SIZE;
                AscendC::DataCopy(queryL1[dstOffset],
                                  queryNzGm[srcOffset],
                                  AscendC::DataCopyParams(static_cast<uint16_t>(realQBlockRows),
                                                          static_cast<uint16_t>(blockLen),
                                                          0,
                                                          static_cast<uint16_t>(dstStride)));
            }
        }
    }

    __aicore__ inline void CopyKeyToL1(AscendC::LocalTensor<DTYPE_KCACHE> dstKeyL1,
                                       uint32_t pageOffset,
                                       uint32_t fullPageSize,
                                       uint32_t tileSize,
                                       uint32_t qkHeadSize)
    {
        const uint16_t burstCount = static_cast<uint16_t>(qkHeadSize / CUBE_BLOCK_SIZE);
        const uint16_t srcGap = static_cast<uint16_t>(fullPageSize - tileSize);
        AscendC::DataCopy(dstKeyL1,
                          kCacheGm[pageOffset],
                          AscendC::DataCopyParams(burstCount,
                                                  static_cast<uint16_t>(tileSize),
                                                  srcGap,
                                                  0));
    }

    //* Mask GM is physical NZ: [1, mask_col_blocks, mask_rows_pad, 16].
    //* Copy all selected 16-column blocks with one multi-burst transfer and pack rows contiguously in L1.
    __aicore__ inline void CopyMaskToL1(AscendC::LocalTensor<DTYPE_ATTENTIONMASK> dstMaskL1,
                                        uint32_t maskOffset,
                                        uint32_t realQBlockRows,
                                        uint32_t pageSize)
    {
        const uint16_t maskBlockCount = static_cast<uint16_t>(pageSize / CUBE_BLOCK_SIZE);
        const uint16_t maskBlockLen =
            static_cast<uint16_t>(realQBlockRows * CUBE_BLOCK_SIZE * sizeof(DTYPE_ATTENTIONMASK) / DATA_BLOCK_BYTES);
        const uint16_t srcStride =
            static_cast<uint16_t>((tilingData->maskRowSize - realQBlockRows) * CUBE_BLOCK_SIZE *
                                  sizeof(DTYPE_ATTENTIONMASK) / DATA_BLOCK_BYTES);
        AscendC::DataCopy(dstMaskL1,
                          maskGm[maskOffset],
                          AscendC::DataCopyParams(maskBlockCount, maskBlockLen, srcStride, 0));
    }

    //* 将数据load到L0A 同时转成ZZ
    __aicore__ inline void LoadQueryToL0A(AscendC::LocalTensor<DTYPE_QUERY> dstQueryL0A,
                                          uint32_t mActual,
                                          uint32_t mSize,
                                          uint32_t qkHeadSize)
    {   //* 这里由于输出设定为A2了，所以会自动转换成ZZ https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00169.html
        (void)mActual;
        // M=1 deliberately uses the normal roundM=16 NZ->ZZ load as well.  MMAD
        // still receives m=1, so only row 0 contributes to the result.

        const uint32_t nzMBlockStride = CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
        for (uint32_t mOffset = 0; mOffset < mSize; mOffset += CUBE_BLOCK_SIZE) {    //? 这里是逐行拷贝到L0？
            AscendC::LoadData(dstQueryL0A[mOffset * qkHeadSize],
                              queryL1[(mOffset / CUBE_BLOCK_SIZE) * nzMBlockStride],
                              AscendC::LoadData2dParams(        //* 参数列表说明 https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0238.html#ZH-CN_TOPIC_0000002151312349__table8955841508
                                  0,
                                  static_cast<uint16_t>(qkHeadSize * sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES),    //迭代次数，每个迭代可以处理512B的数据
                                  static_cast<uint16_t>(mSize * sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES),         //相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B
                                  0,
                                  0,
                                  false,
                                  0));
        }
    }

    __aicore__ inline void LoadKeyToL0B(AscendC::LocalTensor<DTYPE_KCACHE> dstKeyL0B,
                                        AscendC::LocalTensor<DTYPE_KCACHE> srcKeyL1,
                                        uint32_t pageSize,
                                        uint32_t qkHeadSize)
    {   //* 这里由于输出设定为B2了，所以会自动转换成ZN
        AscendC::LoadData(dstKeyL0B,
                          srcKeyL1,
                          AscendC::LoadData2dParams(
                              0,
                              static_cast<uint16_t>(pageSize * qkHeadSize * sizeof(DTYPE_KCACHE) / 512),
                              1,
                              0,
                              0,
                              false,
                              0));
    }

    __aicore__ inline void Bmm1PingSingle(bool loadQueryFromGm,
                                          bool waitQueryReady,
                                          bool setPeerQueryReady,
                                          uint32_t qTokenStart,
                                          uint32_t qHeadStart,
                                          uint32_t realQBlockRows,
                                          uint32_t mActual,
                                          uint32_t roundM,
                                          uint32_t qkHeadSize,
                                          uint32_t gqaGroupSize,
                                          uint32_t maskOffset,
                                          uint32_t keyPageOffset,
                                          uint32_t valuePageOffset,
                                          uint32_t fullPageSize,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize,
                                          bool useKvReuse,
                                          bool loadKvFromGm,
                                          uint32_t kvReuseTileIdx)
    {
        /****** ATB: Bmm1 Ping / QK LOAD + Mask PRELOAD + V PRELOAD ******/
        Bmm1Single(loadQueryFromGm,
                   waitQueryReady,
                   setPeerQueryReady,
                   Pongflag,
                   qTokenStart,
                   qHeadStart,
                   realQBlockRows,
                   mActual,
                   roundM,
                   qkHeadSize,
                   gqaGroupSize,
                   maskOffset,
                   keyPageOffset,
                   valuePageOffset,
                   fullPageSize,
                   tileSize,
                   vHeadSize,
                   useKvReuse,
                   loadKvFromGm,
                   kvReuseTileIdx,
                   Pingflag,
                   PingflagPlus2,
                   PingflagPlus4,
                   PingflagPlus6,
                   keyL1Ping,
                   maskL1Ping,
                   valueL1Ping,
                   queryL0APing,
                   keyL0BPing,
                   scoreL0CPing);
    }

    __aicore__ inline void Bmm1PongSingle(bool loadQueryFromGm,
                                          bool waitQueryReady,
                                          bool setPeerQueryReady,
                                          uint32_t qTokenStart,
                                          uint32_t qHeadStart,
                                          uint32_t realQBlockRows,
                                          uint32_t mActual,
                                          uint32_t roundM,
                                          uint32_t qkHeadSize,
                                          uint32_t gqaGroupSize,
                                          uint32_t maskOffset,
                                          uint32_t keyPageOffset,
                                          uint32_t valuePageOffset,
                                          uint32_t fullPageSize,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize,
                                          bool useKvReuse,
                                          bool loadKvFromGm,
                                          uint32_t kvReuseTileIdx)
    {
        /****** ATB: Bmm1 Pong / QK LOAD + Mask PRELOAD + V PRELOAD ******/
        Bmm1Single(loadQueryFromGm,
                   waitQueryReady,
                   setPeerQueryReady,
                   Pingflag,
                   qTokenStart,
                   qHeadStart,
                   realQBlockRows,
                   mActual,
                   roundM,
                   qkHeadSize,
                   gqaGroupSize,
                   maskOffset,
                   keyPageOffset,
                   valuePageOffset,
                   fullPageSize,
                   tileSize,
                   vHeadSize,
                   useKvReuse,
                   loadKvFromGm,
                   kvReuseTileIdx,
                   Pongflag,
                   PongflagPlus2,
                   PongflagPlus4,
                   PongflagPlus6,
                   keyL1Pong,
                   maskL1Pong,
                   valueL1Pong,
                   queryL0APong,
                   keyL0BPong,
                   scoreL0CPong);
    }

    __aicore__ inline void Bmm1Single(bool loadQueryFromGm,
                                      bool waitQueryReady,
                                      bool setPeerQueryReady,
                                      decltype(Pingflag) peerQueryFlag,
                                      uint32_t qTokenStart,
                                      uint32_t qHeadStart,
                                      uint32_t realQBlockRows,
                                      uint32_t mActual,
                                      uint32_t roundM,
                                      uint32_t qkHeadSize,
                                      uint32_t gqaGroupSize,
                                      uint32_t maskOffset,
                                      uint32_t keyPageOffset,
                                      uint32_t valuePageOffset,
                                      uint32_t fullPageSize,
                                      uint32_t tileSize,
                                      uint32_t vHeadSize,
                                      bool useKvReuse,
                                      bool loadKvFromGm,
                                      uint32_t kvReuseTileIdx,
                                      decltype(Pingflag) mainFlag,
                                      decltype(Pingflag) plus2Flag,
                                      decltype(Pingflag) plus4Flag,
                                      decltype(Pingflag) plus6Flag,
                                      AscendC::LocalTensor<DTYPE_KCACHE> slotKeyL1,
                                      AscendC::LocalTensor<DTYPE_ATTENTIONMASK> slotMaskL1,
                                      AscendC::LocalTensor<DTYPE_VCACHE> slotValueL1,
                                      AscendC::LocalTensor<DTYPE_QUERY> slotQueryL0A,
                                      AscendC::LocalTensor<DTYPE_KCACHE> slotKeyL0B,
                                      AscendC::LocalTensor<float> slotScoreL0C)
    {
        // first time load Q GM ——> L1
        if (loadQueryFromGm) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(mainFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(peerQueryFlag);
            CopyQueryToL1(qTokenStart, qHeadStart, realQBlockRows, roundM, qkHeadSize, gqaGroupSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);
            if (setPeerQueryReady) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(peerQueryFlag);
            }
        }

        // Load Mask GM ——> L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(plus2Flag);
        CopyMaskToL1(slotMaskL1, maskOffset, realQBlockRows, tileSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(plus2Flag);

        // Load Q L1 ——> L0a
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mainFlag);
        if (waitQueryReady) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);
        }
        LoadQueryToL0A(slotQueryL0A, mActual, roundM, qkHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mainFlag);

        // Keep the normal ping/pong event schedule for both cache hits and
        // misses. This mirrors ATB's initKV protocol: only the GM->L1 copy is
        // conditional, while the ready/ownership tokens remain unchanged.
        AscendC::LocalTensor<DTYPE_KCACHE> keyL1Source = slotKeyL1;
        if (useKvReuse) {
            keyL1Source = kvReuseKeyL1[kvReuseTileIdx * kvReuseKeyTileElements];
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(plus4Flag);
        if (loadKvFromGm) {
            CopyKeyToL1(keyL1Source,
                        keyPageOffset,
                        fullPageSize,
                        tileSize,
                        qkHeadSize);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(plus2Flag);
        LoadKeyToL0B(slotKeyL0B, keyL1Source, tileSize, qkHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(plus4Flag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);

        // Preserve the original BMM1 V preload. Cache hits skip only GM->L1;
        // Ping/Pong still publish their own plus4 ready token for BMM2, so the
        // transfer remains overlapped instead of becoming a BMM2 dependency.
        AscendC::LocalTensor<DTYPE_VCACHE> valueL1Source = slotValueL1;
        if (useKvReuse) {
            valueL1Source = kvReuseValueL1[kvReuseTileIdx * kvReuseValueTileElements];
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(plus6Flag);
        if (loadKvFromGm) {
            CopyValueToL1(valueL1Source,
                          valuePageOffset,
                          fullPageSize,
                          tileSize,
                          vHeadSize);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(plus4Flag);

        // MMAD
        // 接下来MMAD要使用q和k，因此确保MTE1的两个都同时完成，同时解开V对于M的保护
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(mainFlag);

        // Q/P use the normal roundM NZ->ZZ layout even when only one row is
        // valid.  Avoid selecting the hardware M=1 packed-vector interpretation
        // for that layout: execute one padded 16-row cube and discard rows 1..15
        // in the later softmax/output stages.
        const uint32_t cubeM = mActual == 1 ? CUBE_BLOCK_SIZE : mActual;
        AscendC::MmadParams qkMmadParams;
        qkMmadParams.m = cubeM;
        qkMmadParams.n = tileSize;
        qkMmadParams.k = qkHeadSize;
        qkMmadParams.cmatrixInitVal = true;
        qkMmadParams.cmatrixSource = false;
        qkMmadParams.unitFlag = 0;
        AscendC::Mmad(slotScoreL0C, slotQueryL0A, slotKeyL0B, qkMmadParams);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(plus2Flag);
        AscendC::SetFlag<AscendC::HardEvent::M_V>(mainFlag);
    }

    __aicore__ inline void CopyScoreToUb(AscendC::LocalTensor<half> scoreUb,
                                         AscendC::LocalTensor<float> srcScoreL0C,
                                         uint32_t mSize,
                                         uint32_t pageSize)
    {
        AscendC::DataCopyParams l0cToUbParams;
        l0cToUbParams.blockCount = 1;
        l0cToUbParams.blockLen = static_cast<uint16_t>(mSize * pageSize / CUBE_MATRIX_SIZE);
        l0cToUbParams.srcStride = 0;
        l0cToUbParams.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;

        // Match ATB l0c_to_ub: copy the contiguous L0C fractal blocks to UB and cast fp32->fp16.
        AscendC::DataCopy(scoreUb,
                          srcScoreL0C,
                          l0cToUbParams,
                          enhancedParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Softmax/update state contract:
    // score/prob/localDelta/localLPage are tile-local and selected by the current ping/pong slot.
    // mOrg/lOrg/outUb are global online-softmax state across all KV tiles of the current Q block.
    // After softmax, localDelta is m_page for the first tile and dm = m_old - m_new for later tiles.
    // Update must consume the localDelta/localLPage from the same slot that produced probL1/outL0C.
    __aicore__ inline void SoftmaxPingSingle(uint32_t qRowKvStart,
                                             uint32_t pageStart,
                                             uint32_t realQBlockRows,
                                             uint32_t mActual,
                                             uint32_t roundM,
                                             uint32_t gqaGroupSize,
                                             uint32_t pageSize,
                                             bool isFirstKvTile)
    {
        SoftmaxSingle(qRowKvStart,
                      pageStart,
                      realQBlockRows,
                      mActual,
                      roundM,
                      gqaGroupSize,
                      pageSize,
                      isFirstKvTile,
                      scoreL0CPing,
                      scoreF16UbPing,
                      maskL1Ping,
                      maskF16UbPing,
                      probF16UbPing,
                      probL1Ping,
                      mPageF16Ub,
                      lPageUb,
                      Pingflag,
                      PingflagPlus2);
    }

    __aicore__ inline void SoftmaxPongSingle(uint32_t qRowKvStart,
                                             uint32_t pageStart,
                                             uint32_t realQBlockRows,
                                             uint32_t mActual,
                                             uint32_t roundM,
                                             uint32_t gqaGroupSize,
                                             uint32_t pageSize,
                                             bool isFirstKvTile)
    {
        SoftmaxSingle(qRowKvStart,
                      pageStart,
                      realQBlockRows,
                      mActual,
                      roundM,
                      gqaGroupSize,
                      pageSize,
                      isFirstKvTile,
                      scoreL0CPong,
                      scoreF16UbPong,
                      maskL1Pong,
                      maskF16UbPong,
                      probF16UbPong,
                      probL1Pong,
                      mPageF16UbPong,
                      lPageUbPong,
                      Pongflag,
                      PongflagPlus2);
    }

    __aicore__ inline void SoftmaxSingle(uint32_t qRowKvStart,
                                         uint32_t pageStart,
                                         uint32_t realQBlockRows,
                                         uint32_t mActual,
                                         uint32_t roundM,
                                         uint32_t gqaGroupSize,
                                         uint32_t pageSize,
                                         bool isFirstKvTile,
                                         AscendC::LocalTensor<float> scoreL0CSlot,
                                         AscendC::LocalTensor<half> scoreUb,
                                         AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskL1Slot,
                                         AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskUb,
                                         AscendC::LocalTensor<half> probHalfUb,
                                         AscendC::LocalTensor<half> probL1Slot,
                                         AscendC::LocalTensor<half> localDeltaF16,
                                         AscendC::LocalTensor<float> localLPage,
                                         decltype(Pingflag) mainFlag,
                                         decltype(Pingflag) plus2Flag)
    {
        // Load score loc ——> UB
        AscendC::WaitFlag<AscendC::HardEvent::M_V>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mainFlag);
        CopyScoreToUb(scoreUb, scoreL0CSlot, roundM, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(mainFlag);

        // score * scale
        ScaleScoreHalf(scoreUb, roundM, pageSize);       //* half score 对齐 ATB 默认 NZ 路径

        // Load Mask L1 ——> UB
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(plus2Flag);
        LoadMaskToUb(maskUb, maskL1Slot, realQBlockRows, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(plus2Flag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_V>(mainFlag);

        // score + mask
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_V>(mainFlag);
        AddMask(scoreUb, maskUb, realQBlockRows, roundM, gqaGroupSize, pageSize);    //* Softmax 阶段从 L1->UB 后再加到 score 上
        // MaskFullyInvisiblePageRows(scoreUb, qRowKvStart, pageStart, realQBlockRows, gqaGroupSize, pageSize);
        MaskInvalidRows(scoreUb, mActual, roundM, pageSize);     //* 将无效的行全部设置为softmax最小值

        // Build the current tile's local softmax state. The final l/out update happens after BMM2.
        BuildSoftmaxLocalState(scoreUb,
                               probHalfUb,
                               probL1Slot,
                               mActual,
                               roundM,
                               pageSize,
                               isFirstKvTile,
                               localDeltaF16,
                               localLPage,
                               mainFlag);
    }

    __aicore__ inline void ScaleScoreHalf(AscendC::LocalTensor<half> scoreUb,
                                          uint32_t mSize,
                                          uint32_t pageSize)
    {
        const uint8_t repeat = static_cast<uint8_t>(mSize * pageSize / HALFS_PER_VECTOR_REPEAT);
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::Muls<half, false>(scoreUb,
                                   scoreUb,
                                   (half)tilingData->softmaxScale,
                                   (uint64_t)0,
                                   repeat,
                                   AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddMask(AscendC::LocalTensor<half> scoreUb,
                                   AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskUb,
                                   uint32_t realQBlockRows,
                                   uint32_t roundM,
                                   uint32_t gqaGroupSize,
                                   uint32_t pageSize)
    {
        if (gqaGroupSize == 1) {
            // Per-Q-head layout: score is [nBlock][roundM][C0], while mask is
            // [nBlock][realQBlockRows][C0].  The old generic fallback emitted
            // one 16-element Add for every row and N block.  Keep the same
            // layout semantics, but add each contiguous matrix region at once.
            if (realQBlockRows == roundM) {
                AscendC::Add(scoreUb,
                             scoreUb,
                             maskUb,
                             roundM * pageSize);
            } else {
                const uint32_t validBlockElements = realQBlockRows * CUBE_BLOCK_SIZE;
                for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
                    const uint32_t blockIdx = nOffset / CUBE_BLOCK_SIZE;
                    const uint32_t scoreOffset = blockIdx * roundM * CUBE_BLOCK_SIZE;
                    const uint32_t maskOffset = blockIdx * realQBlockRows * CUBE_BLOCK_SIZE;
                    AscendC::Add(scoreUb[scoreOffset],
                                 scoreUb[scoreOffset],
                                 maskUb[maskOffset],
                                 validBlockElements);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        if (gqaGroupSize == MAX_GQA_TILE_SIZE) {
            AscendC::BinaryRepeatParams gqa8AddParams(1, 1, 0, 8, 8, 1);
            const uint8_t repeat = static_cast<uint8_t>(realQBlockRows);
            for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
                const uint32_t blockIdx = nOffset / CUBE_BLOCK_SIZE;
                const uint32_t scoreOffset = blockIdx * roundM * CUBE_BLOCK_SIZE;
                const uint32_t maskOffset = blockIdx * realQBlockRows * CUBE_BLOCK_SIZE;
                // GQA=8: one vector repeat maps one q row to its 8 query heads.
                AscendC::Add<half, false>(scoreUb[scoreOffset],
                                          scoreUb[scoreOffset],
                                          maskUb[maskOffset],
                                          (uint64_t)0,
                                          repeat,
                                          gqa8AddParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::BinaryRepeatParams addParams;
        addParams.dstBlkStride = 1;
        addParams.src0BlkStride = 1;
        addParams.src1BlkStride = 1;
        addParams.dstRepStride = 0;
        addParams.src0RepStride = 0;
        addParams.src1RepStride = 0;

        for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
            const uint32_t blockOffset = (nOffset / CUBE_BLOCK_SIZE) * roundM * CUBE_BLOCK_SIZE;
            for (uint32_t qRow = 0; qRow < realQBlockRows; ++qRow) {
                const uint32_t maskRowOffset = (nOffset / CUBE_BLOCK_SIZE) * realQBlockRows * CUBE_BLOCK_SIZE +
                                               qRow * CUBE_BLOCK_SIZE;
                for (uint32_t head = 0; head < gqaGroupSize; ++head) {
                    const uint32_t mOffset = qRow * gqaGroupSize + head;
                    AscendC::Add(scoreUb[blockOffset + mOffset * CUBE_BLOCK_SIZE],
                                 scoreUb[blockOffset + mOffset * CUBE_BLOCK_SIZE],
                                 maskUb[maskRowOffset],
                                 CUBE_BLOCK_SIZE,
                                 1,
                                 addParams);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadMaskToUb(AscendC::LocalTensor<DTYPE_ATTENTIONMASK> dstMaskUb,
                                        AscendC::LocalTensor<DTYPE_ATTENTIONMASK> srcMaskL1,
                                        uint32_t realQBlockRows,
                                        uint32_t pageSize)
    {
        const uint16_t maskBlockLen =
            static_cast<uint16_t>(pageSize * realQBlockRows * sizeof(DTYPE_ATTENTIONMASK) / DATA_BLOCK_BYTES);
        //* mask 已在 CopyMaskToL1 中以 NZ/C0-block 预取到 L1；这里保持同一布局搬到 UB。
        AscendC::DataCopy(dstMaskUb,
                          srcMaskL1,
                          AscendC::DataCopyParams(1, maskBlockLen, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MaskFullyInvisiblePageRows(AscendC::LocalTensor<half> scoreUb,
                                                      uint32_t qRowKvStart,
                                                      uint32_t pageStart,
                                                      uint32_t realQBlockRows,
                                                      uint32_t gqaGroupSize,
                                                      uint32_t pageSize)
    {
        uint32_t invisibleQRows = 0;
        for (; invisibleQRows < realQBlockRows; ++invisibleQRows) {
            const uint32_t rowVisibleKvEnd = qRowKvStart + invisibleQRows + 1;
            if (pageStart < rowVisibleKvEnd) {
                break;
            }
        }
        if (invisibleQRows == 0) {
            return;
        }

        const uint32_t mSize = qBlockRows * gqaGroupSize;
        const uint32_t invisibleElementsPerC0Block = invisibleQRows * gqaGroupSize * CUBE_BLOCK_SIZE;
        // Fully invisible q rows form a contiguous prefix in the folded GQA-M dimension.
        for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
            const uint32_t blockOffset = (nOffset / CUBE_BLOCK_SIZE) * mSize * CUBE_BLOCK_SIZE;
            AscendC::Duplicate(scoreUb[blockOffset],
                               (half)SOFTMAX_INIT_MIN,
                               invisibleElementsPerC0Block);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MaskInvalidRows(AscendC::LocalTensor<half> scoreUb,
                                           uint32_t realMSize,
                                           uint32_t mSize,
                                           uint32_t pageSize)
    {
        if (realMSize >= mSize) {
            return;
        }
        const uint32_t invalidElementsPerC0Block = (mSize - realMSize) * CUBE_BLOCK_SIZE;
        for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
            const uint32_t blockOffset = (nOffset / CUBE_BLOCK_SIZE) * mSize * CUBE_BLOCK_SIZE;
            AscendC::Duplicate(scoreUb[blockOffset + realMSize * CUBE_BLOCK_SIZE],
                               (half)SOFTMAX_INIT_MIN,
                               invalidElementsPerC0Block);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildSoftmaxLocalState(AscendC::LocalTensor<half> scoreUb,
                                                  AscendC::LocalTensor<half> probHalfUb,
                                                  AscendC::LocalTensor<half> probL1Slot,
                                                  uint32_t mActual,
                                                  uint32_t mSize,
                                                  uint32_t pageSize,
                                                  bool isFirstKvTile,
                                                  AscendC::LocalTensor<half> localDeltaF16,
                                                  AscendC::LocalTensor<float> localLPage,
                                                  decltype(Pingflag) mainFlag)
    {
        // Build current tile local state: P, l_page, and either m_page or dm for the later update.
        ReduceC0RowsMaxHalf(localDeltaF16, scoreUb, mSize, pageSize);       //* 找到每行最大值

        if (isFirstKvTile) {
            CopyVectorHalf(mNewF16Ub, localDeltaF16, mSize);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Max(mNewF16Ub, localDeltaF16, mOrgF16Ub, mSize);
            AscendC::PipeBarrier<PIPE_V>();
        }

        ExpandToC0BlockHalf(workF16Ub, mNewF16Ub, mSize);

        SubC0RowsHalf(scoreUb, scoreUb, workF16Ub, mSize, pageSize);

        BuildProbFromScore(scoreUb, probHalfUb, mSize, pageSize);         //* 求得 e^{score - m_new}

        ReduceC0RowsSum(localLPage, probUb, mSize, pageSize);      //* 即 \sum e^{score - m_new}

        if (isFirstKvTile) {
            CopyVectorHalf(mOrgF16Ub, mNewF16Ub, mSize);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // Reuse the local rowmax buffer as dm = m_old - m_new for the later output update.
            AscendC::Sub(localDeltaF16, mOrgF16Ub, mNewF16Ub, mSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(mOrgF16Ub, mNewF16Ub, (half)0.0F, mSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(mainFlag);
        CastProbToL1(probL1Slot, probHalfUb, mActual, mSize, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
    }

    __aicore__ inline void UpdateOnlineState(uint32_t mActual,
                                             uint32_t roundM,
                                             uint32_t vHeadSize,
                                             bool isFirstKvTile,
                                             AscendC::LocalTensor<half> localDeltaF16,
                                             AscendC::LocalTensor<float> localLPage)
    {
        if (isFirstKvTile) {
            CopyVectorFloat(lOrgUb, localLPage, mActual);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(mScaleUb, localDeltaF16, AscendC::RoundMode::CAST_NONE, mActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(mScaleUb, mScaleUb, mActual);            //*  mscale = e^{m_old - m_new}
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(lOrgUb, mScaleUb, lOrgUb, mActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(lNewUb, lOrgUb, localLPage, mActual);        //* l_new = l_old * exp(m_old - m_new) + l_page
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(lOrgUb, lNewUb, 0.0F, mActual);
            AscendC::PipeBarrier<PIPE_V>();

            BroadcastRows(workUb, mScaleUb, roundM);
            MulNzRows(outUb, outUb, workUb, mActual, roundM, vHeadSize);       //* 更新旧输出分子
        }
    }

    __aicore__ inline void BuildProbFromScore(AscendC::LocalTensor<half> scoreUb,
                                              AscendC::LocalTensor<half> probHalfUb,
                                              uint32_t mSize,
                                              uint32_t pageSize)
    {
        const uint32_t pSize = mSize * pageSize;
        const uint32_t halfPartSize = pSize / 2;
        const uint8_t halfToFloatRepeat = static_cast<uint8_t>(halfPartSize / FLOATS_PER_VECTOR_REPEAT);
        const uint8_t floatToHalfRepeat = static_cast<uint8_t>(halfPartSize / FLOATS_PER_VECTOR_REPEAT);

        for (uint32_t idx = 0; idx < 2; ++idx) {
            const uint32_t offset = idx * halfPartSize;
            AscendC::Cast<float, half, false>(probUb[offset],
                                              scoreUb[offset],
                                              AscendC::RoundMode::CAST_NONE,
                                              (uint64_t)0,
                                              halfToFloatRepeat,
                                              AscendC::UnaryRepeatParams(1, 1, 8, 4));
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t idx = 0; idx < 2; ++idx) {
            const uint32_t offset = idx * halfPartSize;
            AscendC::Exp<float, false>(probUb[offset],
                                       probUb[offset],
                                       (uint64_t)0,
                                       halfToFloatRepeat,
                                       AscendC::UnaryRepeatParams(1, 1, 8, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t idx = 0; idx < 2; ++idx) {
            const uint32_t offset = idx * halfPartSize;
            AscendC::Cast<half, float, false>(probHalfUb[offset],
                                              probUb[offset],
                                              AscendC::RoundMode::CAST_NONE,
                                              (uint64_t)0,
                                              floatToHalfRepeat,
                                              AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastRows(AscendC::LocalTensor<float> dst,
                                         AscendC::LocalTensor<float> src,
                                         uint32_t rowCount)
    {
        constexpr uint32_t broadcastBlockSize = DATA_BLOCK_BYTES / sizeof(float);
        const uint8_t repeat = static_cast<uint8_t>(rowCount / broadcastBlockSize);
        AscendC::Brcb(dst, src, repeat, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyVectorHalf(AscendC::LocalTensor<half> dst,
                                          AscendC::LocalTensor<half> src,
                                          uint32_t count)
    {
        AscendC::Adds(dst, src, (half)0.0F, count);
    }

    __aicore__ inline void CopyVectorFloat(AscendC::LocalTensor<float> dst,
                                           AscendC::LocalTensor<float> src,
                                           uint32_t count)
    {
        AscendC::Adds(dst, src, 0.0F, count);
    }

    __aicore__ inline void CastFloatToHalfChunked(AscendC::LocalTensor<half> dst,
                                                  AscendC::LocalTensor<float> src,
                                                  uint32_t count)
    {
        // ATB converts a full 128 x 128 output as two equal 128-repeat chunks.
        // Keep the same bound here instead of issuing one 255-repeat conversion
        // followed by a 1-repeat tail at the end of the UB arena.
        constexpr uint32_t MAX_VECTOR_REPEAT_TIMES = 128;
        uint32_t offset = 0;
        while (offset + FLOATS_PER_VECTOR_REPEAT <= count) {
            const uint32_t fullRepeat = Min(MAX_VECTOR_REPEAT_TIMES,
                                            (count - offset) / FLOATS_PER_VECTOR_REPEAT);
            AscendC::Cast<half, float, false>(dst[offset],
                                              src[offset],
                                              AscendC::RoundMode::CAST_NONE,
                                              (uint64_t)0,
                                              static_cast<uint8_t>(fullRepeat),
                                              AscendC::UnaryRepeatParams(1, 1, 4, 8));
            offset += fullRepeat * FLOATS_PER_VECTOR_REPEAT;
        }
        if (offset < count) {
            AscendC::Cast(dst[offset],
                          src[offset],
                          AscendC::RoundMode::CAST_NONE,
                          count - offset);
        }
    }

    __aicore__ inline uint8_t RepeatForC0Rows(uint32_t rowCount) const
    {
        return static_cast<uint8_t>(rowCount * CUBE_BLOCK_SIZE / HALFS_PER_VECTOR_REPEAT);
    }

    __aicore__ inline uint8_t RepeatForC0RowsFloat(uint32_t rowCount) const
    {
        return static_cast<uint8_t>(rowCount * CUBE_BLOCK_SIZE / FLOATS_PER_VECTOR_REPEAT);
    }

    __aicore__ inline void MaxVectorHalf(AscendC::LocalTensor<half> dst,
                                         AscendC::LocalTensor<half> src0,
                                         AscendC::LocalTensor<half> src1,
                                         uint8_t repeat,
                                         uint8_t dstRepeatStride,
                                         uint8_t src0RepeatStride,
                                         uint8_t src1RepeatStride)
    {
        AscendC::BinaryRepeatParams params(1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        AscendC::Max<half, false>(dst, src0, src1, (uint64_t)0, repeat, params);
    }

    __aicore__ inline void AddVectorFloat(AscendC::LocalTensor<float> dst,
                                          AscendC::LocalTensor<float> src0,
                                          AscendC::LocalTensor<float> src1,
                                          uint8_t repeat,
                                          uint8_t dstRepeatStride,
                                          uint8_t src0RepeatStride,
                                          uint8_t src1RepeatStride)
    {
        AscendC::BinaryRepeatParams params(1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        AscendC::Add<float, false>(dst, src0, src1, (uint64_t)0, repeat, params);
    }

    __aicore__ inline void SubVectorHalf(AscendC::LocalTensor<half> dst,
                                         AscendC::LocalTensor<half> src0,
                                         AscendC::LocalTensor<half> src1,
                                         uint8_t repeat)
    {
        AscendC::BinaryRepeatParams params(1, 1, 1, 8, 8, 8);
        AscendC::Sub<half, false>(dst, src0, src1, (uint64_t)0, repeat, params);
    }

    __aicore__ inline void RepeatReduceSumFloat(AscendC::LocalTensor<float> dst,
                                                AscendC::LocalTensor<float> src,
                                                uint8_t repeat,
                                                uint16_t dstRepeatStride,
                                                uint16_t srcBlockStride,
                                                uint16_t srcRepeatStride)
    {
        AscendC::RepeatReduceSum<float, false>(
            dst, src, repeat, 0, 0, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    __aicore__ inline void ExpandToC0BlockHalf(AscendC::LocalTensor<half> dst,
                                               AscendC::LocalTensor<half> src,
                                               uint32_t rowCount)
    {
        const uint8_t repeat = static_cast<uint8_t>(rowCount / CUBE_BLOCK_SIZE);
        for (uint32_t vaddsIdx = 0; vaddsIdx < 2; ++vaddsIdx) {
            AscendC::Adds<half, false>(dst[vaddsIdx * 8 * CUBE_BLOCK_SIZE],
                                       src,
                                       (half)0.0F,
                                       (uint64_t)0,
                                       repeat,
                                       AscendC::UnaryRepeatParams(1, 0, 16, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowBlock = 0; rowBlock < rowCount / CUBE_BLOCK_SIZE; ++rowBlock) {
            AscendC::Transpose(dst[rowBlock * CUBE_MATRIX_SIZE], dst[rowBlock * CUBE_MATRIX_SIZE]);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceC0RowsMaxHalf(AscendC::LocalTensor<half> dst,
                                               AscendC::LocalTensor<half> src,
                                               uint32_t rowCount,
                                               uint32_t colCount)
    {
        const uint8_t repeat = RepeatForC0Rows(rowCount);
        const uint32_t colBlockCount = colCount / CUBE_BLOCK_SIZE;
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        if (colBlockCount > 1) {
            MaxVectorHalf(workF16Ub,
                          src,
                          src[rowCount * CUBE_BLOCK_SIZE],
                          repeat,
                          8,
                          8,
                          8);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            CopyVectorHalf(workF16Ub, src, rowCount * CUBE_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t colBlock = 2; colBlock < colBlockCount; ++colBlock) {
            MaxVectorHalf(workF16Ub,
                          workF16Ub,
                          src[colBlock * rowCount * CUBE_BLOCK_SIZE],
                          repeat,
                          8,
                          8,
                          8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::BlockReduceMax<half, false>(dst, workF16Ub, repeat, 0, 1, 1, 8);
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SubC0RowsHalf(AscendC::LocalTensor<half> dst,
                                         AscendC::LocalTensor<half> src0,
                                         AscendC::LocalTensor<half> rowBlock,
                                         uint32_t rowCount,
                                         uint32_t colCount)
    {
        const uint8_t repeat = RepeatForC0Rows(rowCount);
        for (uint32_t colBlock = 0; colBlock < colCount / CUBE_BLOCK_SIZE; ++colBlock) {
            SubVectorHalf(dst[colBlock * rowCount * CUBE_BLOCK_SIZE],
                          src0[colBlock * rowCount * CUBE_BLOCK_SIZE],
                          rowBlock,
                          repeat);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceC0RowsSum(AscendC::LocalTensor<float> dst,
                                           AscendC::LocalTensor<float> src,
                                           uint32_t rowCount,
                                           uint32_t colCount)
    {
        const uint8_t repeat = RepeatForC0RowsFloat(rowCount);
        if (colCount / CUBE_BLOCK_SIZE > 1) {
            AddVectorFloat(workUb,
                           src,
                           src[rowCount * CUBE_BLOCK_SIZE],
                           repeat,
                           8,
                           8,
                           8);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            CopyVectorFloat(workUb, src, rowCount * CUBE_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t colBlock = 2; colBlock < colCount / CUBE_BLOCK_SIZE; ++colBlock) {
            AddVectorFloat(workUb,
                           workUb,
                           src[colBlock * rowCount * CUBE_BLOCK_SIZE],
                           repeat,
                           8,
                           8,
                           8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
        constexpr uint32_t MAX_VECTOR_REPEAT_TIMES = 255;
        for (uint32_t rowStart = 0; rowStart < rowCount; rowStart += MAX_VECTOR_REPEAT_TIMES) {
            const uint8_t rowRepeat = static_cast<uint8_t>(Min(MAX_VECTOR_REPEAT_TIMES, rowCount - rowStart));
            RepeatReduceSumFloat(dst[rowStart], workUb[rowStart * CUBE_BLOCK_SIZE], rowRepeat, 1, 1, 2);
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DumpC0RowsHalfToWork(AscendC::LocalTensor<half> src,
                                                uint32_t rowStart,
                                                uint32_t dumpId,
                                                uint32_t colCount)
    {
        DumpC0RowsHalfToWork(src, rowStart, dumpId, colCount, Min(tilingData->gqaGroupSize, MAX_GQA_TILE_SIZE));
    }

    __aicore__ inline void DumpC0RowsHalfToWork(AscendC::LocalTensor<half> src,
                                                uint32_t rowStart,
                                                uint32_t dumpId,
                                                uint32_t colCount,
                                                uint32_t gqaTileSize)
    {
        const uint32_t rowCount = qBlockRows * gqaTileSize;
        if (rowStart >= rowCount) {
            return;
        }
        const uint32_t dumpRows = Min(CUBE_BLOCK_SIZE, rowCount - rowStart);
        for (uint32_t colBlock = 0; colBlock < colCount / CUBE_BLOCK_SIZE; ++colBlock) {
            const uint32_t srcBlockOffset = colBlock * rowCount * CUBE_BLOCK_SIZE;
            const uint32_t dstColOffset = colBlock * CUBE_BLOCK_SIZE;
            for (uint32_t row = 0; row < dumpRows; ++row) {
                CopyVectorHalf(workF16Ub[row * colCount + dstColOffset],
                               src[srcBlockOffset + (rowStart + row) * CUBE_BLOCK_SIZE],
                               CUBE_BLOCK_SIZE);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DumpC0Rows(AscendC::LocalTensor<float> src,
                                      uint32_t rowStart,
                                      uint32_t dumpId,
                                      uint32_t colCount)
    {
        DumpC0Rows(src, rowStart, dumpId, colCount, Min(tilingData->gqaGroupSize, MAX_GQA_TILE_SIZE));
    }

    __aicore__ inline void DumpC0Rows(AscendC::LocalTensor<float> src,
                                      uint32_t rowStart,
                                      uint32_t dumpId,
                                      uint32_t colCount,
                                      uint32_t gqaTileSize)
    {
        const uint32_t rowCount = qBlockRows * gqaTileSize;
        if (rowStart >= rowCount) {
            return;
        }
        const uint32_t dumpRows = Min(CUBE_BLOCK_SIZE, rowCount - rowStart);
        for (uint32_t colBlock = 0; colBlock < colCount / CUBE_BLOCK_SIZE; ++colBlock) {
            const uint32_t srcBlockOffset = colBlock * rowCount * CUBE_BLOCK_SIZE;
            const uint32_t dstColOffset = colBlock * CUBE_BLOCK_SIZE;
            for (uint32_t row = 0; row < dumpRows; ++row) {
                CopyVectorFloat(workUb[row * colCount + dstColOffset],
                                src[srcBlockOffset + (rowStart + row) * CUBE_BLOCK_SIZE],
                                CUBE_BLOCK_SIZE);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulNzRows(AscendC::LocalTensor<float> dst,
                                     AscendC::LocalTensor<float> src0,
                                     AscendC::LocalTensor<float> rowScaleC0,
                                     uint32_t rowCount,
                                     uint32_t roundM,
                                     uint32_t colCount)
    {
        // outUb is [nBlock][roundM][C0]. BroadcastRows stores one fp32
        // data block (8 copies) per row; src1BlkStride=0 broadcasts it over
        // both fp32 blocks of C0 while src1RepStride advances to the next row.
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        params.dstRepStride = CUBE_BLOCK_SIZE * sizeof(float) / DATA_BLOCK_BYTES;
        params.src0RepStride = params.dstRepStride;
        params.src1RepStride = 1;

        for (uint32_t colBlock = 0; colBlock < colCount / CUBE_BLOCK_SIZE; ++colBlock) {
            const uint32_t blockOffset = colBlock * roundM * CUBE_BLOCK_SIZE;
            AscendC::Mul(dst[blockOffset],
                         src0[blockOffset],
                         rowScaleC0,
                         CUBE_BLOCK_SIZE,
                         static_cast<uint8_t>(rowCount),
                         params);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Bmm2LoadValueToL0B(AscendC::LocalTensor<DTYPE_VCACHE> dstValueL0B,
                                              AscendC::LocalTensor<DTYPE_VCACHE> srcValueL1,
                                              uint32_t pageSize,
                                              uint32_t vHeadSize,
                                              decltype(Pingflag) plus2Flag,
                                              decltype(Pingflag) plus6Flag,
                                              decltype(Pingflag) plus4Flag)
    {
        // Load V L1 ——> L0b
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(plus4Flag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(plus2Flag);
        LoadValueToL0B(dstValueL0B, srcValueL1, pageSize, vHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(plus6Flag);
    }

    __aicore__ inline void Bmm2LoadProbToL0A(AscendC::LocalTensor<half> dstProbL0A,
                                             AscendC::LocalTensor<half> srcProbL1,
                                             uint32_t mActual,
                                             uint32_t mSize,
                                             uint32_t pageSize,
                                             decltype(Pingflag) mainFlag)
    {
        // Load P L1 ——> L0a
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mainFlag);
        LoadProbToL0A(dstProbL0A, srcProbL1, mActual, mSize, pageSize);        //* load的同时转成ZZ
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(mainFlag);
    }

    __aicore__ inline void Bmm2Mmad(AscendC::LocalTensor<float> dstOutL0C,
                                    AscendC::LocalTensor<half> srcProbL0A,
                                    AscendC::LocalTensor<DTYPE_VCACHE> srcValueL0B,
                                    uint32_t mActual,
                                    uint32_t pageSize,
                                    uint32_t vHeadSize,
                                    decltype(Pingflag) mainFlag,
                                    decltype(Pingflag) plus2Flag)
    {
        // MMAD
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(mainFlag);

        // Match BMM1's padded-cube handling for the normal roundM P layout.
        // Only row 0 is folded into online state and written when mActual == 1.
        const uint32_t cubeM = mActual == 1 ? CUBE_BLOCK_SIZE : mActual;
        AscendC::MmadParams pvMmadParams;
        pvMmadParams.m = cubeM;
        pvMmadParams.n = vHeadSize;
        pvMmadParams.k = pageSize;
        pvMmadParams.cmatrixInitVal = true;
        pvMmadParams.cmatrixSource = false;
        pvMmadParams.unitFlag = 0;
        AscendC::Mmad(dstOutL0C, srcProbL0A, srcValueL0B, pvMmadParams);       //* 计算P * V

        AscendC::SetFlag<AscendC::HardEvent::M_V>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(plus2Flag);
    }

    __aicore__ inline void Bmm2PingSingle(uint32_t mActual,
                                          uint32_t roundM,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize,
                                          bool useKvReuse,
                                          uint32_t kvReuseTileIdx)
    {
        /****** ATB: Bmm2 Ping / V L1->L0B + P L1->L0A + P*V MMAD ******/
        Bmm2Single(mActual,
                   roundM,
                   tileSize,
                   vHeadSize,
                   useKvReuse,
                   kvReuseTileIdx,
                   Pingflag,
                   PingflagPlus2,
                   PingflagPlus6,
                   PingflagPlus4,
                   valueL0BPing,
                   valueL1Ping,
                   probL0APing,
                   probL1Ping,
                   outL0CPing);
    }

    __aicore__ inline void Bmm2PongSingle(uint32_t mActual,
                                          uint32_t roundM,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize,
                                          bool useKvReuse,
                                          uint32_t kvReuseTileIdx)
    {
        /****** ATB: Bmm2 Pong / V L1->L0B + P L1->L0A + P*V MMAD ******/
        Bmm2Single(mActual,
                   roundM,
                   tileSize,
                   vHeadSize,
                   useKvReuse,
                   kvReuseTileIdx,
                   Pongflag,
                   PongflagPlus2,
                   PongflagPlus6,
                   PongflagPlus4,
                   valueL0BPong,
                   valueL1Pong,
                   probL0APong,
                   probL1Pong,
                   outL0CPong);
    }

    __aicore__ inline void Bmm2Single(uint32_t mActual,
                                      uint32_t roundM,
                                      uint32_t tileSize,
                                      uint32_t vHeadSize,
                                      bool useKvReuse,
                                      uint32_t kvReuseTileIdx,
                                      decltype(Pingflag) mainFlag,
                                      decltype(Pingflag) plus2Flag,
                                      decltype(Pingflag) plus6Flag,
                                      decltype(Pingflag) plus4Flag,
                                      AscendC::LocalTensor<DTYPE_VCACHE> slotValueL0B,
                                      AscendC::LocalTensor<DTYPE_VCACHE> slotValueL1,
                                      AscendC::LocalTensor<half> slotProbL0A,
                                      AscendC::LocalTensor<half> slotProbL1,
                                      AscendC::LocalTensor<float> slotOutL0C)
    {
        AscendC::LocalTensor<DTYPE_VCACHE> valueL1Source = slotValueL1;
        if (useKvReuse) {
            valueL1Source = kvReuseValueL1[kvReuseTileIdx * kvReuseValueTileElements];
        }
        Bmm2LoadValueToL0B(slotValueL0B,
                           valueL1Source,
                           tileSize,
                           vHeadSize,
                           plus2Flag,
                           plus6Flag,
                           plus4Flag);
        Bmm2LoadProbToL0A(slotProbL0A, slotProbL1, mActual, roundM, tileSize, mainFlag);
        Bmm2Mmad(slotOutL0C, slotProbL0A, slotValueL0B,
                 mActual, tileSize, vHeadSize, mainFlag, plus2Flag);
    }

    __aicore__ inline void UpdatePingSingle(uint32_t mActual,
                                            uint32_t roundM,
                                            uint32_t vHeadSize,
                                            bool isFirstKvTile)
    {
        /****** ATB: Update Ping / L0C->UB + online state + fold PV ******/
        UpdateSingle(mActual, roundM, vHeadSize, isFirstKvTile, mPageF16Ub, lPageUb, outL0CPing, Pingflag);
    }

    __aicore__ inline void UpdatePongSingle(uint32_t mActual,
                                            uint32_t roundM,
                                            uint32_t vHeadSize,
                                            bool isFirstKvTile)
    {
        /****** ATB: Update Pong / L0C->UB + online state + fold PV ******/
        UpdateSingle(mActual, roundM, vHeadSize, isFirstKvTile, mPageF16UbPong, lPageUbPong, outL0CPong, Pongflag);
    }

    __aicore__ inline void UpdateSingle(uint32_t mActual,
                                        uint32_t roundM,
                                        uint32_t vHeadSize,
                                        bool isFirstKvTile,
                                        AscendC::LocalTensor<half> localDeltaF16,
                                        AscendC::LocalTensor<float> localLPage,
                                        AscendC::LocalTensor<float> slotOutL0C,
                                        decltype(Pingflag) mainFlag)
    {
        // load result L0c ——> UB  out+=result
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_V>(mainFlag);
        const uint32_t pvScratchCols = Min(vHeadSize, Min(tilingData->pageSize, KV_TILE_SIZE));
        const uint32_t firstChunkCols = Min(pvScratchCols, vHeadSize);

        CopyPvChunkToUb(probUb, slotOutL0C, roundM, 0, firstChunkCols, mainFlag);
        UpdateOnlineState(mActual, roundM, vHeadSize, isFirstKvTile, localDeltaF16, localLPage);
        WaitAndFoldPvChunk(probUb, mActual, roundM, vHeadSize, 0, firstChunkCols, isFirstKvTile, mainFlag);

        for (uint32_t chunkStart = firstChunkCols; chunkStart < vHeadSize; chunkStart += pvScratchCols) {
            const uint32_t chunkCols = Min(pvScratchCols, vHeadSize - chunkStart);
            CopyPvChunkToUb(probUb, slotOutL0C, roundM, chunkStart, chunkCols, mainFlag);
            WaitAndFoldPvChunk(probUb, mActual, roundM, vHeadSize, chunkStart, chunkCols, isFirstKvTile, mainFlag);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(mainFlag);
    }

    __aicore__ inline void CastProbToL1(AscendC::LocalTensor<half> dstProbL1,
                                        AscendC::LocalTensor<half> probHalfUb,
                                        uint32_t mActual,
                                        uint32_t mSize,
                                        uint32_t pageSize)
    {
        (void)mActual;
        // Preserve the full roundM NZ probability tile for M=1 instead of
        // switching to a separately packed vector representation.

        AscendC::DataCopy(dstProbL1,
                          probHalfUb,
                          AscendC::DataCopyParams(
                              1,
                              static_cast<uint16_t>(mSize * pageSize * sizeof(half) / DATA_BLOCK_BYTES),
                              0,
                              0));
    }

    __aicore__ inline void LoadProbToL0A(AscendC::LocalTensor<half> dstProbL0A,
                                         AscendC::LocalTensor<half> srcProbL1,
                                         uint32_t mActual,
                                         uint32_t mSize,
                                         uint32_t pageSize)
    {
        (void)mActual;
        // Use the same roundM NZ->ZZ path for the one-row probability matrix.

        for (uint32_t mOffset = 0; mOffset < mSize; mOffset += CUBE_BLOCK_SIZE) {
            AscendC::LoadData(dstProbL0A[mOffset * pageSize],
                              srcProbL1[mOffset * CUBE_BLOCK_SIZE],
                              AscendC::LoadData2dParams(
                                  0,
                                  static_cast<uint16_t>(pageSize * sizeof(half) / DATA_BLOCK_BYTES),
                                  static_cast<uint16_t>(mSize * sizeof(half) / DATA_BLOCK_BYTES),
                                  0,
                                  0,
                                  false,
                                  0));
        }
    }

    __aicore__ inline void CopyValueToL1(AscendC::LocalTensor<DTYPE_VCACHE> dstValueL1,
                                         uint32_t pageOffset,
                                         uint32_t fullPageSize,
                                         uint32_t tileSize,
                                         uint32_t vHeadSize)
    {
        const uint16_t burstCount = static_cast<uint16_t>(vHeadSize / CUBE_BLOCK_SIZE);
        const uint16_t srcGap = static_cast<uint16_t>(fullPageSize - tileSize);
        AscendC::DataCopy(dstValueL1,
                          vCacheGm[pageOffset],
                          AscendC::DataCopyParams(burstCount,
                                                  static_cast<uint16_t>(tileSize),
                                                  srcGap,
                                                  0));
    }

    __aicore__ inline void LoadValueToL0B(AscendC::LocalTensor<DTYPE_VCACHE> dstValueL0B,
                                          AscendC::LocalTensor<DTYPE_VCACHE> srcValueL1,
                                          uint32_t pageSize,
                                          uint32_t vHeadSize)
    {
        for (uint32_t kOffset = 0; kOffset < pageSize; kOffset += CUBE_BLOCK_SIZE) {
            AscendC::LoadData(dstValueL0B[kOffset * vHeadSize],
                              srcValueL1[kOffset * CUBE_BLOCK_SIZE],
                              AscendC::LoadData2dParams(                                            //每次处理512B数据
                                  0,                                                                //startIndex
                                  static_cast<uint16_t>(vHeadSize / CUBE_BLOCK_SIZE),               //repeatTimes
                                  static_cast<uint16_t>(pageSize * DATA_BLOCK_BYTES / 512),         //srcStride
                                  0,
                                  0,                                                                //dstGap
                                  true,                                                             //ifTranspose
                                  0));                                                              //addrMode
        }
    }

    __aicore__ inline void CopyPvChunkToUb(AscendC::LocalTensor<float> pvUb,
                                           AscendC::LocalTensor<float> srcOutL0C,
                                           uint32_t roundM,
                                           uint32_t chunkStart,
                                           uint32_t chunkCols,
                                           decltype(Pingflag) mainFlag)
    {
        AscendC::DataCopyParams l0cToUbParams;
        l0cToUbParams.blockCount = 1;
        l0cToUbParams.blockLen = static_cast<uint16_t>(roundM * chunkCols / CUBE_MATRIX_SIZE);
        l0cToUbParams.srcStride = 0;
        l0cToUbParams.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(pvUb,
                          srcOutL0C[chunkStart * roundM],
                          l0cToUbParams,
                          enhancedParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mainFlag);
    }

    __aicore__ inline void WaitAndFoldPvChunk(AscendC::LocalTensor<float> pvUb,
                                              uint32_t mActual,
                                              uint32_t roundM,
                                              uint32_t vHeadSize,
                                              uint32_t chunkStart,
                                              uint32_t chunkCols,
                                              bool isFirstKvTile,
                                              decltype(Pingflag) mainFlag)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mainFlag);

        // BMM2's L0C->UB result and the persistent accumulator now share the
        // same [nBlock][row][C0] order. Accumulate each C0 block directly; no
        // per-row NZ->ND fold is needed on every KV tile.
        for (uint32_t nOffset = 0; nOffset < chunkCols; nOffset += CUBE_BLOCK_SIZE) {
            const uint32_t dstOffset = (chunkStart + nOffset) * roundM;
            const uint32_t srcOffset = nOffset * roundM;
            const uint32_t count = mActual * CUBE_BLOCK_SIZE;
            if (isFirstKvTile) {
                CopyVectorFloat(outUb[dstOffset], pvUb[srcOffset], count);
            } else {
                AscendC::Add(outUb[dstOffset], outUb[dstOffset], pvUb[srcOffset], count);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeAndWriteOutput(uint32_t outputOffset,
                                                   uint32_t mActual,
                                                   uint32_t roundM,
                                                   uint32_t qHeadNum,
                                                   uint32_t vHeadSize)
    {
        // Keep ATB's fp16 final normalization while preserving the internal
        // [vBlock][row][C0] layout. Each block is contiguous, so cast/div are
        // dense vector operations. Only the final MTE3 copies scatter C0 blocks
        // into the public ND [token, qHead, vDim] tensor.
        CastFloatToHalfChunked(mNewF16Ub, lOrgUb, roundM);
        AscendC::PipeBarrier<PIPE_V>();
        ExpandToC0BlockHalf(workF16Ub, mNewF16Ub, roundM);

        for (uint32_t colStart = 0; colStart < vHeadSize; colStart += CUBE_BLOCK_SIZE) {
            const uint32_t blockOffset = colStart * roundM;
            const uint32_t blockElements = mActual * CUBE_BLOCK_SIZE;
            CastFloatToHalfChunked(outF16Ub[blockOffset], outUb[blockOffset], blockElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(outF16Ub[blockOffset],
                         outF16Ub[blockOffset],
                         workF16Ub,
                         blockElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Pingflag);
        const uint16_t dstStride = static_cast<uint16_t>(
            (qHeadNum * vHeadSize - CUBE_BLOCK_SIZE) *
            sizeof(DTYPE_OUTPUT) / DATA_BLOCK_BYTES);
        for (uint32_t colStart = 0; colStart < vHeadSize; colStart += CUBE_BLOCK_SIZE) {
            AscendC::DataCopy(outputGm[outputOffset + colStart],
                              outF16Ub[colStart * roundM],
                              AscendC::DataCopyParams(
                                  static_cast<uint16_t>(mActual),
                                  1,
                                  0,
                                  dstStride));
        }
        // Return ownership of outF16Ub/scoreF16UbPong only after all C0 writes
        // have consumed the source. The next task waits at its first reuse.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OutputReuseFlag);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> queryL1Buf;
    AscendC::TBuf<AscendC::TPosition::A1> keyL1Buf;
    AscendC::TBuf<AscendC::TPosition::A1> maskL1Buf;
    AscendC::TBuf<AscendC::TPosition::A2> queryL0ABuf;
    AscendC::TBuf<AscendC::TPosition::B2> keyL0BBuf;
    AscendC::TBuf<AscendC::TPosition::CO1> scoreL0CBuf;
    AscendC::TBuf<AscendC::TPosition::A1> probL1Buf;
    AscendC::TBuf<AscendC::TPosition::A1> valueL1Buf;
    AscendC::TBuf<AscendC::TPosition::A1> kvReuseL1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
    bool ubLayoutValid = false;
    AscendC::LocalTensor<DTYPE_QUERY> queryL1;
    AscendC::LocalTensor<DTYPE_KCACHE> keyL1Ping;
    AscendC::LocalTensor<DTYPE_KCACHE> keyL1Pong;
    AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskL1Ping;
    AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskL1Pong;
    AscendC::LocalTensor<DTYPE_QUERY> queryL0APing;
    AscendC::LocalTensor<DTYPE_QUERY> queryL0APong;
    AscendC::LocalTensor<half> probL0APing;
    AscendC::LocalTensor<half> probL0APong;
    AscendC::LocalTensor<DTYPE_KCACHE> keyL0BPing;
    AscendC::LocalTensor<DTYPE_KCACHE> keyL0BPong;
    AscendC::LocalTensor<DTYPE_VCACHE> valueL0BPing;
    AscendC::LocalTensor<DTYPE_VCACHE> valueL0BPong;
    AscendC::LocalTensor<float> scoreL0CPing;
    AscendC::LocalTensor<float> scoreL0CPong;
    AscendC::LocalTensor<float> outL0CPing;
    AscendC::LocalTensor<float> outL0CPong;
    AscendC::LocalTensor<uint8_t> ubBase;
    AscendC::LocalTensor<half> scoreF16UbPing;
    AscendC::LocalTensor<half> scoreF16UbPong;
    AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskF16UbPing;
    AscendC::LocalTensor<DTYPE_ATTENTIONMASK> maskF16UbPong;
    AscendC::LocalTensor<half> mOrgF16Ub;
    AscendC::LocalTensor<half> mPageF16Ub;
    AscendC::LocalTensor<half> mPageF16UbPong;
    AscendC::LocalTensor<half> mNewF16Ub;
    AscendC::LocalTensor<float> mScaleUb;
    AscendC::LocalTensor<float> probUb;
    AscendC::LocalTensor<float> lOrgUb;
    AscendC::LocalTensor<float> lPageUb;
    AscendC::LocalTensor<float> lPageUbPong;
    AscendC::LocalTensor<float> lNewUb;
    AscendC::LocalTensor<float> outUb;
    AscendC::LocalTensor<float> workUb;
    AscendC::LocalTensor<half> workF16Ub;
    AscendC::LocalTensor<half> probF16UbPing;
    AscendC::LocalTensor<half> probF16UbPong;
    AscendC::LocalTensor<half> outF16Ub;
    AscendC::LocalTensor<half> probL1Ping;
    AscendC::LocalTensor<half> probL1Pong;
    AscendC::LocalTensor<DTYPE_VCACHE> valueL1Ping;
    AscendC::LocalTensor<DTYPE_VCACHE> valueL1Pong;
    AscendC::LocalTensor<DTYPE_KCACHE> kvReuseKeyL1;
    AscendC::LocalTensor<DTYPE_VCACHE> kvReuseValueL1;
    AscendC::LocalTensor<int32_t> blockTableUb;

    AscendC::GlobalTensor<DTYPE_QUERY> queryGm;
    AscendC::GlobalTensor<DTYPE_QUERY> queryNzGm;
    AscendC::GlobalTensor<DTYPE_KCACHE> kCacheGm;
    AscendC::GlobalTensor<DTYPE_VCACHE> vCacheGm;
    AscendC::GlobalTensor<DTYPE_ATTENTIONMASK> maskGm;
    AscendC::GlobalTensor<int32_t> blockTableGm;
    AscendC::GlobalTensor<int64_t> kvLengthsGm;
    AscendC::GlobalTensor<int64_t> seqLenGm;
    AscendC::GlobalTensor<DTYPE_OUTPUT> outputGm;
    const PagedAttentionMixV3TilingData *tilingData;
    uint32_t qBlockRows;
    uint32_t kvReuseCacheTileCount = 0;
    uint32_t kvReuseKeyTileElements = 0;
    uint32_t kvReuseValueTileElements = 0;
    uint32_t blockTableCacheCapacity = 0;
};


extern "C" __global__ __aicore__ void paged_attention_mix_v3(GM_ADDR query, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR attentionMask, GM_ADDR blockTable, GM_ADDR kvLengths, GM_ADDR seqLenPerRequest, GM_ADDR queryRope, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    PagedAttentionMixV3Kernel op;
    op.Init(query, kCache, vCache, attentionMask, blockTable, kvLengths, seqLenPerRequest, queryRope, output,
            &tiling_data);
    op.Process();
    (void)workspace;
}
