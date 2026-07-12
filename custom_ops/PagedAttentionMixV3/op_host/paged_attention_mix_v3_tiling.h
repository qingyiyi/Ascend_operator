#ifndef PAGED_ATTENTION_MIX_V3_TILING_H_
#define PAGED_ATTENTION_MIX_V3_TILING_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t PAGED_ATTENTION_MIX_V3_MAX_CORE_NUM = 32;

BEGIN_TILING_DATA_DEF(PagedAttentionMixV3TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, numTokensPad);
  TILING_DATA_FIELD_DEF(uint32_t, qHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, qkHeadSize);
  TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, vHeadSize);
  TILING_DATA_FIELD_DEF(uint32_t, pageSize);
  TILING_DATA_FIELD_DEF(uint32_t, pageNumPerBatch);
  TILING_DATA_FIELD_DEF(float, softmaxScale);
  TILING_DATA_FIELD_DEF(uint32_t, maskColSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskRowSize);
  TILING_DATA_FIELD_DEF(uint32_t, gqaGroupSize);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, taskStartPerCore);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, taskEndPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PagedAttentionMixV3, PagedAttentionMixV3TilingData)
}

#endif // PAGED_ATTENTION_MIX_V3_TILING_H_
