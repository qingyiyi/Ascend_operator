#ifndef PAGED_ATTENTION_MIX_V3_TILING_H_
#define PAGED_ATTENTION_MIX_V3_TILING_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PagedAttentionMixV3TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
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
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PagedAttentionMixV3, PagedAttentionMixV3TilingData)
}

#endif // PAGED_ATTENTION_MIX_V3_TILING_H_
