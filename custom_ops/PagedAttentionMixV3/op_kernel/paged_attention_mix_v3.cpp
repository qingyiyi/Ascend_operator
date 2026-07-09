#include "kernel_operator.h"

namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t VEC_SIZE_IN_BYTES = 8 * DATA_BLOCK_BYTES;
constexpr uint32_t FLOATS_PER_VECTOR_REPEAT = VEC_SIZE_IN_BYTES / sizeof(float);
constexpr uint32_t HALFS_PER_VECTOR_REPEAT = VEC_SIZE_IN_BYTES / sizeof(half);
constexpr uint32_t CUBE_BLOCK_SIZE = 16;
constexpr uint32_t CUBE_MATRIX_SIZE = CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
constexpr uint32_t MAX_Q_ROWS_PER_GROUP = 128;
constexpr uint32_t MAX_GQA_TILE_SIZE = 8;
constexpr uint32_t KV_TILE_SIZE = 128;
constexpr float SOFTMAX_INIT_MIN = -65504.0F;

constexpr auto Pingflag = EVENT_ID0;
constexpr auto Pongflag = EVENT_ID1;
constexpr auto PingflagPlus2 = EVENT_ID2;
constexpr auto PongflagPlus2 = EVENT_ID3;
constexpr auto PingflagPlus4 = EVENT_ID4;
constexpr auto PongflagPlus4 = EVENT_ID5;
constexpr auto PingflagPlus6 = EVENT_ID6;
constexpr auto PongflagPlus6 = EVENT_ID7;
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
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t usedCoreNum = tilingData->usedCoreNum == 0 ? 1 : tilingData->usedCoreNum;
        if (coreIdx >= usedCoreNum) {
            return;
        }

        const uint32_t qHeadNum = tilingData->qHeadNum;
        const uint32_t qkHeadSize = tilingData->qkHeadSize;
        const uint32_t kvHeadNum = tilingData->kvHeadNum;
        const uint32_t vHeadSize = tilingData->vHeadSize;
        const uint32_t pageSize = tilingData->pageSize;
        const uint32_t pageNumPerBatch = tilingData->pageNumPerBatch;
        const uint32_t gqaGroupSize = tilingData->gqaGroupSize;
        const uint32_t gqaTileLimit = Min(gqaGroupSize, MAX_GQA_TILE_SIZE);

        uint32_t totalTaskNum = 0;
        for (uint32_t batchIdx = 0; batchIdx < tilingData->batchSize; ++batchIdx) {
            const uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            totalTaskNum += CeilDiv(seqLen, qBlockRows) * kvHeadNum;
        }
        if (totalTaskNum == 0) {
            return;
        }

        // task划分
        const uint32_t taskNumPerCore = totalTaskNum / usedCoreNum;
        const uint32_t tailTaskNum = totalTaskNum % usedCoreNum;
        uint32_t taskIdStart = coreIdx * taskNumPerCore + Min(coreIdx, tailTaskNum);
        uint32_t taskIdEnd = taskIdStart + taskNumPerCore + (coreIdx < tailTaskNum ? 1U : 0U);
        if (taskIdStart >= taskIdEnd) {
            return;
        }

        uint32_t qBatchOffset = 0;
        uint32_t taskBase = 0;
        uint32_t batchIdx = 0;
        for (; batchIdx < tilingData->batchSize; ++batchIdx) {
            const uint32_t seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            const uint32_t batchTaskNum = CeilDiv(seqLen, qBlockRows) * kvHeadNum;
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
        uint32_t qBlockStart = (taskInBatch / kvHeadNum) * qBlockRows;
        uint32_t groupIdx = taskInBatch % kvHeadNum;

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
        
        for (uint32_t taskId = taskIdStart; taskId < taskIdEnd; ++taskId) {
            const uint32_t realQBlockRows = Min(qBlockRows, seqLen - qBlockStart);
            for (uint32_t gqaTileStart = 0; gqaTileStart < gqaGroupSize; gqaTileStart += gqaTileLimit) {
                const uint32_t curGqaTileSize = Min(gqaTileLimit, gqaGroupSize - gqaTileStart);
                const uint32_t qTokenStart = qBatchOffset + qBlockStart;
                const uint32_t qHeadStart = groupIdx * gqaGroupSize + gqaTileStart;
                // The first visible KV tile overwrites O/l/m, so no vector init is needed here.

                const uint32_t visibleKvEnd = historyKvLen + qBlockStart + realQBlockRows;
                const uint32_t kvTileSize = Min(pageSize, KV_TILE_SIZE);
                const uint32_t tilesPerPage = CeilDiv(pageSize, kvTileSize);
                const uint32_t totalKvTiles = kvPageNum * tilesPerPage;
                bool queryGuardConsumed = false;
                for (uint32_t tilePairStart = 0; tilePairStart < totalKvTiles; tilePairStart += 2) {
                    KvTileInfo pingTile;
                    if (!BuildKvTileInfo(pingTile,
                                         tilePairStart,
                                         tilesPerPage,
                                         kvTileSize,
                                         pageSize,
                                         visibleKvEnd,
                                         batchIdx,
                                         pageNumPerBatch,
                                         qBatchOffset,
                                         qBlockStart)) {
                        break;
                    }

                    KvTileInfo pongTile;
                    const bool hasPongTile = BuildKvTileInfo(pongTile,
                                                             tilePairStart + 1,
                                                             tilesPerPage,
                                                             kvTileSize,
                                                             pageSize,
                                                             visibleKvEnd,
                                                             batchIdx,
                                                             pageNumPerBatch,
                                                             qBatchOffset,
                                                             qBlockStart);

                    /****** ATB: Bmm1 Ping Start ******/
                    queryGuardConsumed = queryGuardConsumed || pingTile.isFirstKvTile;
                    Bmm1PingSingle(pingTile.isFirstKvTile,
                                   pingTile.isFirstKvTile,
                                   hasPongTile,
                                   qTokenStart,
                                   qHeadStart,
                                   realQBlockRows,
                                   qkHeadSize,
                                   curGqaTileSize,
                                   pingTile.maskOffset,
                                   pingTile.physicalPageIdx,
                                   groupIdx,
                                   pageSize,
                                   pingTile.tileStartInPage,
                                   pingTile.tileSize,
                                   vHeadSize);
                    if (hasPongTile) {
                        /****** ATB: Bmm1 Pong Starts ******/
                        Bmm1PongSingle(false,
                                       pingTile.isFirstKvTile,
                                       false,
                                       qTokenStart,
                                       qHeadStart,
                                       realQBlockRows,
                                       qkHeadSize,
                                       curGqaTileSize,
                                       pongTile.maskOffset,
                                       pongTile.physicalPageIdx,
                                       groupIdx,
                                       pageSize,
                                       pongTile.tileStartInPage,
                                       pongTile.tileSize,
                                       vHeadSize);
                    }

                    /****** ATB: Softmax Ping Starts ******/
                    SoftmaxPingSingle(historyKvLen + qBlockStart,
                                      pingTile.tilePageStart,
                                      realQBlockRows,
                                      curGqaTileSize,
                                      pingTile.tileSize,
                                      pingTile.isFirstKvTile);
                    if (hasPongTile) {
                        /****** ATB: Softmax Pong Starts ******/
                        SoftmaxPongSingle(historyKvLen + qBlockStart,
                                          pongTile.tilePageStart,
                                          realQBlockRows,
                                          curGqaTileSize,
                                          pongTile.tileSize,
                                          pongTile.isFirstKvTile);
                    }

                    /****** ATB: Bmm2 Ping Starts ******/
                    Bmm2PingSingle(curGqaTileSize,
                                   pingTile.tileSize,
                                   vHeadSize);     //* P*V
                    if (hasPongTile) {
                        /****** ATB: Bmm2 Pong Starts ******/
                        Bmm2PongSingle(curGqaTileSize,
                                       pongTile.tileSize,
                                       vHeadSize);     //* P*V
                    }

                    /****** ATB: Update Ping Starts ******/
                    UpdatePingSingle(curGqaTileSize,
                                     vHeadSize,
                                     pingTile.isFirstKvTile);     //* 更新UB，同时累加到总out上
                    if (hasPongTile) {
                        /****** ATB: Update Pong Starts ******/
                        UpdatePongSingle(curGqaTileSize,
                                         vHeadSize,
                                         pongTile.isFirstKvTile);     //* 更新UB，同时累加到总out上
                    }
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

                NormalizeAndWriteOutput((qBatchOffset + qBlockStart) * qHeadNum * vHeadSize +         //* 此时得到了总l，因此处理l得到最终的输出
                                            (groupIdx * gqaGroupSize + gqaTileStart) * vHeadSize,
                                        realQBlockRows,
                                        curGqaTileSize,
                                        qHeadNum,
                                        vHeadSize);
            }

            if (taskId + 1 >= taskIdEnd) {
                // return;
                break;
            }
            ++groupIdx;
            if (groupIdx < kvHeadNum) {
                continue;
            }
            groupIdx = 0;
            qBlockStart += qBlockRows;
            if (qBlockStart < seqLen) {
                continue;
            }
            qBatchOffset += seqLen;
            ++batchIdx;
            if (batchIdx >= tilingData->batchSize) {
                // return;
                break;
            }
            seqLen = static_cast<uint32_t>(seqLenGm.GetValue(batchIdx));
            kvLen = static_cast<uint32_t>(kvLengthsGm.GetValue(batchIdx));
            historyKvLen = kvLen - seqLen;
            kvPageNum = CeilDiv(kvLen, pageSize);
            qBlockStart = 0;
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
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    struct KvTileInfo {
        uint32_t pageIdxInSeq;
        uint32_t tileStartInPage;
        uint32_t tileSize;
        uint32_t tilePageStart;
        uint32_t maskOffset;
        int32_t physicalPageIdx;
        bool isFirstKvTile;
    };

    __aicore__ inline uint32_t CeilDiv(uint32_t lhs, uint32_t rhs) const
    {
        return (lhs + rhs - 1) / rhs;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline bool BuildKvTileInfo(KvTileInfo &tileInfo,
                                           uint32_t tileLinearIdx,
                                           uint32_t tilesPerPage,
                                           uint32_t kvTileSize,
                                           uint32_t pageSize,
                                           uint32_t visibleKvEnd,
                                           uint32_t batchIdx,
                                           uint32_t pageNumPerBatch,
                                           uint32_t qBatchOffset,
                                           uint32_t qBlockStart)
    {
        const uint32_t pageIdxInSeq = tileLinearIdx / tilesPerPage;
        const uint32_t tileIdxInPage = tileLinearIdx - pageIdxInSeq * tilesPerPage;
        const uint32_t tileStartInPage = tileIdxInPage * kvTileSize;
        const uint32_t tilePageStart = pageIdxInSeq * pageSize + tileStartInPage;
        if (tilePageStart >= visibleKvEnd) {
            return false;
        }

        const uint32_t maskColBlock = tilePageStart / CUBE_BLOCK_SIZE;
        const uint32_t maskRowStart = qBatchOffset + qBlockStart;

        tileInfo.pageIdxInSeq = pageIdxInSeq;
        tileInfo.tileStartInPage = tileStartInPage;
        tileInfo.tileSize = Min(kvTileSize, pageSize - tileStartInPage);
        tileInfo.tilePageStart = tilePageStart;
        tileInfo.maskOffset = maskColBlock * tilingData->maskRowSize * CUBE_BLOCK_SIZE + maskRowStart * CUBE_BLOCK_SIZE;
        tileInfo.physicalPageIdx = blockTableGm.GetValue(batchIdx * pageNumPerBatch + pageIdxInSeq);
        tileInfo.isFirstKvTile = pageIdxInSeq == 0 && tileStartInPage == 0;
        return true;
    }

    __aicore__ inline void InitLocalBuffer()
    {   // 根据 tiling 参数，把这个 kernel 后面要用到的 L1/L0/UB 空间全部申请并切好
        const uint32_t gqaTileSize = Min(tilingData->gqaGroupSize, MAX_GQA_TILE_SIZE);
        qBlockRows = MAX_Q_ROWS_PER_GROUP / gqaTileSize;
        qBlockRows = qBlockRows / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
        if (qBlockRows == 0) {
            qBlockRows = CUBE_BLOCK_SIZE;
        }

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
        const uint32_t rowSize = mSize;
        const uint32_t maskSize = qBlockRows * kvTileSize;
        const uint32_t workSize = Max(kvTileSize * CUBE_BLOCK_SIZE, mSize * CUBE_BLOCK_SIZE) + 8192;
        const uint32_t ubBytes = (scoreSize + outSize + rowSize * 5 + workSize) * sizeof(float) +
                            (scoreSize * 2 + outF16Size + rowSize * 4) * sizeof(half) +
                            maskSize * 2 * sizeof(DTYPE_ATTENTIONMASK);
        uint32_t ubOffset = 0;

        pipe.InitBuffer(queryL0ABuf, 2 * l0aBytes);
        pipe.InitBuffer(keyL0BBuf, 2 * l0bBytes);
        pipe.InitBuffer(scoreL0CBuf, 2 * l0cBytes);
        pipe.InitBuffer(probL1Buf, 2 * l1ProbBytes);
        pipe.InitBuffer(valueL1Buf, 2 * l1ValueBytes);
        pipe.InitBuffer(ubBuf, ubBytes);

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
        scoreF16UbPing = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += scoreSize * sizeof(half);
        probUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += scoreSize * sizeof(float);
        probF16UbPing = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += scoreSize * sizeof(half);
        outUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += outSize * sizeof(float);
        outF16Ub = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += outF16Size * sizeof(half);
        // Reuse the final output cast buffer as pong score/prob storage. BuildProbFromScore
        // first consumes the whole half score into probUb, then overwrites the same half window with prob.
        scoreF16UbPong = outF16Ub;
        probF16UbPong = outF16Ub;
        mOrgF16Ub = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += rowSize * sizeof(half);
        mPageF16Ub = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += rowSize * sizeof(half);
        mPageF16UbPong = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += rowSize * sizeof(half);
        mNewF16Ub = ubBase[ubOffset].ReinterpretCast<half>();
        ubOffset += rowSize * sizeof(half);
        mScaleUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += rowSize * sizeof(float);
        lOrgUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += rowSize * sizeof(float);
        lPageUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += rowSize * sizeof(float);
        lPageUbPong = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += rowSize * sizeof(float);
        lNewUb = ubBase[ubOffset].ReinterpretCast<float>();
        ubOffset += rowSize * sizeof(float);
        maskF16UbPing = ubBase[ubOffset].ReinterpretCast<DTYPE_ATTENTIONMASK>();
        ubOffset += maskSize * sizeof(DTYPE_ATTENTIONMASK);
        maskF16UbPong = ubBase[ubOffset].ReinterpretCast<DTYPE_ATTENTIONMASK>();
        ubOffset += maskSize * sizeof(DTYPE_ATTENTIONMASK);
        workUb = ubBase[ubOffset].ReinterpretCast<float>();
        workF16Ub = ubBase[ubOffset].ReinterpretCast<half>();
        probL1Ping = probL1Buf.Get<half>();
        probL1Pong = probL1Ping[l1ProbElements];
        valueL1Ping = valueL1Buf.Get<DTYPE_VCACHE>();
        valueL1Pong = valueL1Ping[l1ValueElements];
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
    //* GM physical order is [head][kBlock][token][c0]. This copy keeps the local folded-GQA
    //* NZ order used by BMM1: rows=(q_rows * gqa_tile), cols=head_dim.
    __aicore__ inline void CopyQueryToL1(uint32_t qTokenStart,
                                         uint32_t qHeadStart,
                                         uint32_t realQBlockRows,
                                         uint32_t qkHeadSize,
                                         uint32_t gqaTileSize)
    {
        const uint32_t mSize = qBlockRows * gqaTileSize;
        const uint32_t qkHeadBlocks = qkHeadSize / CUBE_BLOCK_SIZE;
        const uint32_t numTokensPad = tilingData->numTokensPad;
        const uint32_t headStride = numTokensPad * qkHeadSize;
        const uint32_t kBlockStride = numTokensPad * CUBE_BLOCK_SIZE;
        const uint32_t blockLen = CUBE_BLOCK_SIZE * sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES;
        const uint32_t dstStride = (gqaTileSize - 1) * CUBE_BLOCK_SIZE *
                                   sizeof(DTYPE_QUERY) / DATA_BLOCK_BYTES;
        constexpr uint32_t DATA_COPY_PARAM_LIMIT = 65535;

        if (gqaTileSize == 0 ||
            qHeadStart + gqaTileSize > tilingData->qHeadNum ||
            qkHeadSize % CUBE_BLOCK_SIZE != 0 ||
            qTokenStart + realQBlockRows > numTokensPad ||
            realQBlockRows > DATA_COPY_PARAM_LIMIT ||
            blockLen > DATA_COPY_PARAM_LIMIT ||
            dstStride > DATA_COPY_PARAM_LIMIT) {
            return;
        }

        for (uint32_t kBlock = 0; kBlock < qkHeadBlocks; ++kBlock) {
            for (uint32_t tileHead = 0; tileHead < gqaTileSize; ++tileHead) {
                const uint32_t qHeadId = qHeadStart + tileHead;
                const uint32_t srcOffset =
                    qHeadId * headStride + kBlock * kBlockStride + qTokenStart * CUBE_BLOCK_SIZE;
                const uint32_t dstOffset = kBlock * mSize * CUBE_BLOCK_SIZE + tileHead * CUBE_BLOCK_SIZE;
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
                                       int32_t physicalPageIdx,
                                       uint32_t groupIdx,
                                       uint32_t fullPageSize,
                                       uint32_t tileStartInPage,
                                       uint32_t tileSize,
                                       uint32_t qkHeadSize)
    {
        const uint32_t hiddenOffset = groupIdx * qkHeadSize;
        const uint32_t pageOffset =
            static_cast<uint32_t>(physicalPageIdx) * fullPageSize * tilingData->kvHeadNum * qkHeadSize +
            hiddenOffset * fullPageSize + tileStartInPage * CUBE_BLOCK_SIZE;
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
                                          uint32_t mSize,
                                          uint32_t qkHeadSize)
    {   //* 这里由于输出设定为A2了，所以会自动转换成ZZ https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00169.html
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
                                          uint32_t qkHeadSize,
                                          uint32_t gqaGroupSize,
                                          uint32_t maskOffset,
                                          int32_t physicalPageIdx,
                                          uint32_t groupIdx,
                                          uint32_t fullPageSize,
                                          uint32_t tileStartInPage,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize)
    {
        /****** ATB: Bmm1 Ping / QK LOAD + Mask PRELOAD + V PRELOAD ******/
        Bmm1Single(loadQueryFromGm,
                   waitQueryReady,
                   setPeerQueryReady,
                   Pongflag,
                   qTokenStart,
                   qHeadStart,
                   realQBlockRows,
                   qkHeadSize,
                   gqaGroupSize,
                   maskOffset,
                   physicalPageIdx,
                   groupIdx,
                   fullPageSize,
                   tileStartInPage,
                   tileSize,
                   vHeadSize,
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
                                          uint32_t qkHeadSize,
                                          uint32_t gqaGroupSize,
                                          uint32_t maskOffset,
                                          int32_t physicalPageIdx,
                                          uint32_t groupIdx,
                                          uint32_t fullPageSize,
                                          uint32_t tileStartInPage,
                                          uint32_t tileSize,
                                          uint32_t vHeadSize)
    {
        /****** ATB: Bmm1 Pong / QK LOAD + Mask PRELOAD + V PRELOAD ******/
        Bmm1Single(loadQueryFromGm,
                   waitQueryReady,
                   setPeerQueryReady,
                   Pingflag,
                   qTokenStart,
                   qHeadStart,
                   realQBlockRows,
                   qkHeadSize,
                   gqaGroupSize,
                   maskOffset,
                   physicalPageIdx,
                   groupIdx,
                   fullPageSize,
                   tileStartInPage,
                   tileSize,
                   vHeadSize,
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
                                      uint32_t qkHeadSize,
                                      uint32_t gqaGroupSize,
                                      uint32_t maskOffset,
                                      int32_t physicalPageIdx,
                                      uint32_t groupIdx,
                                      uint32_t fullPageSize,
                                      uint32_t tileStartInPage,
                                      uint32_t tileSize,
                                      uint32_t vHeadSize,
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
        const uint32_t mSize = qBlockRows * gqaGroupSize;

        // first time load Q GM ——> L1
        if (loadQueryFromGm) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(mainFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(peerQueryFlag);
            CopyQueryToL1(qTokenStart, qHeadStart, realQBlockRows, qkHeadSize, gqaGroupSize);
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
        LoadQueryToL0A(slotQueryL0A, mSize, qkHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mainFlag);

        // Load K GM ——> L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(plus4Flag);
        CopyKeyToL1(slotKeyL1,
                    physicalPageIdx,
                    groupIdx,
                    fullPageSize,
                    tileStartInPage,
                    tileSize,
                    qkHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);

        // Load K GM ——> L0b  (分别解开前一个硬件和后一个硬件的保护，同时阻塞前一个硬件和后一个硬件，以防止冲突)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(plus2Flag);
        LoadKeyToL0B(slotKeyL0B, slotKeyL1, tileSize, qkHeadSize);        //* 由于 K(NZ) = K^T(ZN) 所以这里不需要转置
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(plus4Flag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);

        // Load V GM ——> L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(plus6Flag);
        CopyValueToL1(slotValueL1,
                      physicalPageIdx,
                      groupIdx,
                      fullPageSize,
                      tileStartInPage,
                      tileSize,
                      vHeadSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(plus4Flag);

        // MMAD
        // 接下来MMAD要使用q和k，因此确保MTE1的两个都同时完成，同时解开V对于M的保护
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(mainFlag);

        AscendC::MmadParams qkMmadParams;
        qkMmadParams.m = mSize;
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
                                             uint32_t gqaGroupSize,
                                             uint32_t pageSize,
                                             bool isFirstKvTile)
    {
        SoftmaxSingle(qRowKvStart,
                      pageStart,
                      realQBlockRows,
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
                                             uint32_t gqaGroupSize,
                                             uint32_t pageSize,
                                             bool isFirstKvTile)
    {
        SoftmaxSingle(qRowKvStart,
                      pageStart,
                      realQBlockRows,
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
        const uint32_t mSize = qBlockRows * gqaGroupSize;
        const uint32_t realMSize = realQBlockRows * gqaGroupSize;

        // Load score loc ——> UB
        AscendC::WaitFlag<AscendC::HardEvent::M_V>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mainFlag);
        CopyScoreToUb(scoreUb, scoreL0CSlot, mSize, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(mainFlag);

        // score * scale
        ScaleScoreHalf(scoreUb, mSize, pageSize);       //* half score 对齐 ATB 默认 NZ 路径

        // Load Mask L1 ——> UB
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(plus2Flag);
        LoadMaskToUb(maskUb, maskL1Slot, realQBlockRows, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(plus2Flag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_V>(mainFlag);

        // score + mask
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_V>(mainFlag);
        AddMask(scoreUb, maskUb, realQBlockRows, gqaGroupSize, pageSize);    //* Softmax 阶段从 L1->UB 后再加到 score 上
        // MaskFullyInvisiblePageRows(scoreUb, qRowKvStart, pageStart, realQBlockRows, gqaGroupSize, pageSize);
        MaskInvalidRows(scoreUb, realMSize, mSize, pageSize);     //* 将无效的行全部设置为softmax最小值

        // Build the current tile's local softmax state. The final l/out update happens after BMM2.
        BuildSoftmaxLocalState(scoreUb,
                               probHalfUb,
                               probL1Slot,
                               mSize,
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
                                   uint32_t gqaGroupSize,
                                   uint32_t pageSize)
    {
        if (gqaGroupSize == MAX_GQA_TILE_SIZE) {
            AscendC::BinaryRepeatParams gqa8AddParams(1, 1, 0, 8, 8, 1);
            const uint8_t repeat = static_cast<uint8_t>(realQBlockRows);
            const uint32_t mSize = qBlockRows * gqaGroupSize;
            for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
                const uint32_t blockIdx = nOffset / CUBE_BLOCK_SIZE;
                const uint32_t scoreOffset = blockIdx * mSize * CUBE_BLOCK_SIZE;
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

        const uint32_t mSize = qBlockRows * gqaGroupSize;     //? 错误旧版路径，写的太零碎，每次只add 16tile长度
        for (uint32_t nOffset = 0; nOffset < pageSize; nOffset += CUBE_BLOCK_SIZE) {
            const uint32_t blockOffset = (nOffset / CUBE_BLOCK_SIZE) * mSize * CUBE_BLOCK_SIZE;
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
        CastProbToL1(probL1Slot, probHalfUb, mSize, pageSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
    }

    __aicore__ inline void UpdateOnlineState(uint32_t mSize,
                                             uint32_t vHeadSize,
                                             bool isFirstKvTile,
                                             AscendC::LocalTensor<half> localDeltaF16,
                                             AscendC::LocalTensor<float> localLPage)
    {
        if (isFirstKvTile) {
            CopyVectorFloat(lOrgUb, localLPage, mSize);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(mScaleUb, localDeltaF16, AscendC::RoundMode::CAST_NONE, mSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(mScaleUb, mScaleUb, mSize);            //*  mscale = e^{m_old - m_new}
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(lOrgUb, mScaleUb, lOrgUb, mSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(lNewUb, lOrgUb, localLPage, mSize);        //* l_new = l_old * exp(m_old - m_new) + l_page
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(lOrgUb, lNewUb, 0.0F, mSize);
            AscendC::PipeBarrier<PIPE_V>();

            BroadcastRows(workUb, mScaleUb, mSize);
            MulRows(outUb, outUb, workUb, mSize, vHeadSize);       //* 更新旧输出分子
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

    __aicore__ inline void MulRows(AscendC::LocalTensor<float> dst,
                                   AscendC::LocalTensor<float> src0,
                                   AscendC::LocalTensor<float> src1,
                                   uint32_t rowCount,
                                   uint32_t colCount)
    {
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        params.dstRepStride = static_cast<uint16_t>(colCount * sizeof(float) / DATA_BLOCK_BYTES);
        params.src0RepStride = static_cast<uint16_t>(colCount * sizeof(float) / DATA_BLOCK_BYTES);
        params.src1RepStride = 1;

        const uint32_t simdSize = VEC_SIZE_IN_BYTES / sizeof(float);
        uint32_t colOffset = 0;
        for (; colOffset + simdSize <= colCount; colOffset += simdSize) {
            AscendC::Mul(dst[colOffset], src0[colOffset], src1, simdSize, rowCount, params);
        }
        if (colOffset < colCount) {
            AscendC::Mul(dst[colOffset], src0[colOffset], src1, colCount - colOffset, rowCount, params);
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
                                             uint32_t mSize,
                                             uint32_t pageSize,
                                             decltype(Pingflag) mainFlag)
    {
        // Load P L1 ——> L0a
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mainFlag);
        LoadProbToL0A(dstProbL0A, srcProbL1, mSize, pageSize);        //* load的同时转成ZZ
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(mainFlag);
    }

    __aicore__ inline void Bmm2Mmad(AscendC::LocalTensor<float> dstOutL0C,
                                    AscendC::LocalTensor<half> srcProbL0A,
                                    AscendC::LocalTensor<DTYPE_VCACHE> srcValueL0B,
                                    uint32_t mSize,
                                    uint32_t pageSize,
                                    uint32_t vHeadSize,
                                    decltype(Pingflag) mainFlag,
                                    decltype(Pingflag) plus2Flag)
    {
        // MMAD
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(plus2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(mainFlag);

        AscendC::MmadParams pvMmadParams;
        pvMmadParams.m = mSize;
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

    __aicore__ inline void Bmm2PingSingle(uint32_t gqaGroupSize,
                                          uint32_t pageSize,
                                          uint32_t vHeadSize)
    {
        /****** ATB: Bmm2 Ping / V L1->L0B + P L1->L0A + P*V MMAD ******/
        Bmm2Single(gqaGroupSize,
                   pageSize,
                   vHeadSize,
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

    __aicore__ inline void Bmm2PongSingle(uint32_t gqaGroupSize,
                                          uint32_t pageSize,
                                          uint32_t vHeadSize)
    {
        /****** ATB: Bmm2 Pong / V L1->L0B + P L1->L0A + P*V MMAD ******/
        Bmm2Single(gqaGroupSize,
                   pageSize,
                   vHeadSize,
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

    __aicore__ inline void Bmm2Single(uint32_t gqaGroupSize,
                                      uint32_t pageSize,
                                      uint32_t vHeadSize,
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
        const uint32_t mSize = qBlockRows * gqaGroupSize;
        Bmm2LoadValueToL0B(slotValueL0B, slotValueL1, pageSize, vHeadSize, plus2Flag, plus6Flag, plus4Flag);
        Bmm2LoadProbToL0A(slotProbL0A, slotProbL1, mSize, pageSize, mainFlag);
        Bmm2Mmad(slotOutL0C, slotProbL0A, slotValueL0B, mSize, pageSize, vHeadSize, mainFlag, plus2Flag);
    }

    __aicore__ inline void UpdatePingSingle(uint32_t gqaGroupSize,
                                            uint32_t vHeadSize,
                                            bool isFirstKvTile)
    {
        /****** ATB: Update Ping / L0C->UB + online state + fold PV ******/
        UpdateSingle(gqaGroupSize, vHeadSize, isFirstKvTile, mPageF16Ub, lPageUb, outL0CPing, Pingflag);
    }

    __aicore__ inline void UpdatePongSingle(uint32_t gqaGroupSize,
                                            uint32_t vHeadSize,
                                            bool isFirstKvTile)
    {
        /****** ATB: Update Pong / L0C->UB + online state + fold PV ******/
        UpdateSingle(gqaGroupSize, vHeadSize, isFirstKvTile, mPageF16UbPong, lPageUbPong, outL0CPong, Pongflag);
    }

    __aicore__ inline void UpdateSingle(uint32_t gqaGroupSize,
                                        uint32_t vHeadSize,
                                        bool isFirstKvTile,
                                        AscendC::LocalTensor<half> localDeltaF16,
                                        AscendC::LocalTensor<float> localLPage,
                                        AscendC::LocalTensor<float> slotOutL0C,
                                        decltype(Pingflag) mainFlag)
    {
        // load result L0c ——> UB  out+=result
        const uint32_t mSize = qBlockRows * gqaGroupSize;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_V>(mainFlag);
        const uint32_t pvScratchCols = Min(vHeadSize, Min(tilingData->pageSize, KV_TILE_SIZE));
        const uint32_t firstChunkCols = Min(pvScratchCols, vHeadSize);

        CopyPvChunkToUb(probUb, slotOutL0C, mSize, 0, firstChunkCols, mainFlag);
        UpdateOnlineState(mSize, vHeadSize, isFirstKvTile, localDeltaF16, localLPage);
        WaitAndFoldPvChunk(probUb, mSize, vHeadSize, 0, firstChunkCols, isFirstKvTile, mainFlag);

        for (uint32_t chunkStart = firstChunkCols; chunkStart < vHeadSize; chunkStart += pvScratchCols) {
            const uint32_t chunkCols = Min(pvScratchCols, vHeadSize - chunkStart);
            CopyPvChunkToUb(probUb, slotOutL0C, mSize, chunkStart, chunkCols, mainFlag);
            WaitAndFoldPvChunk(probUb, mSize, vHeadSize, chunkStart, chunkCols, isFirstKvTile, mainFlag);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(mainFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_M>(mainFlag);
    }

    __aicore__ inline void CastProbToL1(AscendC::LocalTensor<half> dstProbL1,
                                        AscendC::LocalTensor<half> probHalfUb,
                                        uint32_t mSize,
                                        uint32_t pageSize)
    {
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
                                         uint32_t mSize,
                                         uint32_t pageSize)
    {
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
                                         int32_t physicalPageIdx,
                                         uint32_t groupIdx,
                                         uint32_t fullPageSize,
                                         uint32_t tileStartInPage,
                                         uint32_t tileSize,
                                         uint32_t vHeadSize)
    {
        const uint32_t hiddenOffset = groupIdx * vHeadSize;
        const uint32_t vPageOffset =
            static_cast<uint32_t>(physicalPageIdx) * fullPageSize * tilingData->kvHeadNum * vHeadSize +
            hiddenOffset * fullPageSize +
            tileStartInPage * CUBE_BLOCK_SIZE;
        const uint16_t burstCount = static_cast<uint16_t>(vHeadSize / CUBE_BLOCK_SIZE);
        const uint16_t srcGap = static_cast<uint16_t>(fullPageSize - tileSize);
        AscendC::DataCopy(dstValueL1,
                          vCacheGm[vPageOffset],
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
                                           uint32_t mSize,
                                           uint32_t chunkStart,
                                           uint32_t chunkCols,
                                           decltype(Pingflag) mainFlag)
    {
        AscendC::DataCopyParams l0cToUbParams;
        l0cToUbParams.blockCount = 1;
        l0cToUbParams.blockLen = static_cast<uint16_t>(mSize * chunkCols / CUBE_MATRIX_SIZE);
        l0cToUbParams.srcStride = 0;
        l0cToUbParams.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(pvUb,
                          srcOutL0C[chunkStart * mSize],
                          l0cToUbParams,
                          enhancedParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mainFlag);
    }

    __aicore__ inline void WaitAndFoldPvChunk(AscendC::LocalTensor<float> pvUb,
                                              uint32_t mSize,
                                              uint32_t vHeadSize,
                                              uint32_t chunkStart,
                                              uint32_t chunkCols,
                                              bool isFirstKvTile,
                                              decltype(Pingflag) mainFlag)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mainFlag);
        if (isFirstKvTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
            for (uint32_t nOffset = 0; nOffset < chunkCols; nOffset += CUBE_BLOCK_SIZE) {
                CopyPvChunkToOut(outUb[chunkStart + nOffset], pvUb[nOffset * mSize], mSize, vHeadSize);
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }

        AscendC::BinaryRepeatParams addParams;
        addParams.dstBlkStride = 1;
        addParams.src0BlkStride = 1;
        addParams.src1BlkStride = 1;
        addParams.dstRepStride = static_cast<uint16_t>(vHeadSize * sizeof(float) / DATA_BLOCK_BYTES);
        addParams.src0RepStride = static_cast<uint16_t>(vHeadSize * sizeof(float) / DATA_BLOCK_BYTES);
        addParams.src1RepStride = static_cast<uint16_t>(CUBE_BLOCK_SIZE * sizeof(float) / DATA_BLOCK_BYTES);
        for (uint32_t nOffset = 0; nOffset < chunkCols; nOffset += CUBE_BLOCK_SIZE) {
            AscendC::Add(outUb[chunkStart + nOffset],
                         outUb[chunkStart + nOffset],
                         pvUb[nOffset * mSize],
                         CUBE_BLOCK_SIZE,
                         mSize,
                         addParams);    //* 写回UB，同时累加到总out上
        }
    }

    __aicore__ inline void CopyPvChunkToOut(AscendC::LocalTensor<float> dst,
                                            AscendC::LocalTensor<float> src,
                                            uint32_t mSize,
                                            uint32_t vHeadSize)
    {
        AscendC::Adds<float, false>(dst,
                                    src,
                                    0.0F,
                                    (uint64_t)0,
                                    static_cast<uint8_t>(mSize),
                                    AscendC::UnaryRepeatParams(
                                        1,
                                        1,
                                        static_cast<uint16_t>(vHeadSize * sizeof(float) / DATA_BLOCK_BYTES),
                                        static_cast<uint16_t>(CUBE_BLOCK_SIZE * sizeof(float) / DATA_BLOCK_BYTES)));
    }

    __aicore__ inline void NormalizeAndWriteOutput(uint32_t outputOffset,
                                                   uint32_t realQBlockRows,
                                                   uint32_t gqaGroupSize,
                                                   uint32_t qHeadNum,
                                                   uint32_t vHeadSize)
    {
        const uint32_t mSize = qBlockRows * gqaGroupSize;

        BroadcastRows(workUb, lOrgUb, mSize);   

        DivRows(outUb, outUb, workUb, mSize, vHeadSize);       //* output = outUB / l0rgUB

        AscendC::Cast(outF16Ub, outUb, AscendC::RoundMode::CAST_NONE, mSize * vHeadSize);   
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Pingflag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Pingflag);
        AscendC::DataCopy(outputGm[outputOffset],
                          outF16Ub,
                          AscendC::DataCopyParams(
                              static_cast<uint16_t>(realQBlockRows),
                              static_cast<uint16_t>(gqaGroupSize * vHeadSize * sizeof(DTYPE_OUTPUT) / DATA_BLOCK_BYTES),
                              0,
                              static_cast<uint16_t>((qHeadNum - gqaGroupSize) * vHeadSize *
                                                    sizeof(DTYPE_OUTPUT) / DATA_BLOCK_BYTES)));
        AscendC::PipeBarrier<PIPE_ALL>();  //! 让当前 AICore 上所有 pipe 的前序操作都完成/可见后，再继续执行后面的操作   //!以硬件视角
    }

    __aicore__ inline void DivRows(AscendC::LocalTensor<float> dst,
                                   AscendC::LocalTensor<float> src0,
                                   AscendC::LocalTensor<float> src1,
                                   uint32_t rowCount,
                                   uint32_t colCount)
    {
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        const uint32_t simdSize = VEC_SIZE_IN_BYTES / sizeof(float);
        const uint32_t broadcastBlockSize = DATA_BLOCK_BYTES / sizeof(float);
        if (colCount % broadcastBlockSize != 0) {
            params.dstRepStride = 0;
            params.src0RepStride = 0;
            params.src1RepStride = 0;
            for (uint32_t row = 0; row < rowCount; ++row) {
                const uint32_t rowOffset = row * colCount;
                const uint32_t divisorOffset = row * broadcastBlockSize;
                uint32_t colOffset = 0;
                for (; colOffset + simdSize <= colCount; colOffset += simdSize) {
                    AscendC::Div(dst[rowOffset + colOffset],
                                 src0[rowOffset + colOffset],
                                 src1[divisorOffset],
                                 simdSize,
                                 1,
                                 params);
                }
                if (colOffset < colCount) {
                    AscendC::Div(dst[rowOffset + colOffset],
                                 src0[rowOffset + colOffset],
                                 src1[divisorOffset],
                                 colCount - colOffset,
                                 1,
                                 params);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        const uint16_t rowStride = static_cast<uint16_t>(colCount / broadcastBlockSize);
        params.dstRepStride = rowStride;
        params.src0RepStride = rowStride;
        params.src1RepStride = 1;

        constexpr uint32_t MAX_VECTOR_REPEAT_TIMES = 255;
        for (uint32_t rowStart = 0; rowStart < rowCount; rowStart += MAX_VECTOR_REPEAT_TIMES) {
            const uint8_t repeat = static_cast<uint8_t>(Min(MAX_VECTOR_REPEAT_TIMES, rowCount - rowStart));
            const uint32_t rowOffset = rowStart * colCount;
            const uint32_t divisorOffset = rowStart * broadcastBlockSize;
            uint32_t colOffset = 0;
            for (; colOffset + simdSize <= colCount; colOffset += simdSize) {
                AscendC::Div(dst[rowOffset + colOffset],
                             src0[rowOffset + colOffset],
                             src1[divisorOffset],
                             simdSize,
                             repeat,
                             params);
            }
            if (colOffset < colCount) {
                AscendC::Div(dst[rowOffset + colOffset],
                             src0[rowOffset + colOffset],
                             src1[divisorOffset],
                             colCount - colOffset,
                             repeat,
                             params);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
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
};


extern "C" __global__ __aicore__ void paged_attention_mix_v3(GM_ADDR query, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR attentionMask, GM_ADDR blockTable, GM_ADDR kvLengths, GM_ADDR seqLenPerRequest, GM_ADDR queryRope, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    PagedAttentionMixV3Kernel op;
    op.Init(query, kCache, vCache, attentionMask, blockTable, kvLengths, seqLenPerRequest, queryRope, output,
            &tiling_data);
    op.Process();
    (void)workspace;
}
