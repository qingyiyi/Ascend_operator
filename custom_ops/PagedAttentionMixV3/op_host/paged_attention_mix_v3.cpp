#include "paged_attention_mix_v3_tiling.h"
#include "register/op_def_registry.h"

#include <cstdint>
#include <limits>
#include <cstdio>
#include <algorithm>

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t CUBE_BLOCK_SIZE = 16;
constexpr uint32_t MAX_Q_ROWS_PER_GROUP = 128;
constexpr uint32_t MAX_GQA_TILE_SIZE = 8;
constexpr uint32_t MAX_GQA_GROUP_SIZE = 16;
constexpr uint32_t NZ_C0_SIZE = 16;

template <typename T>
bool MissingConstTensor(const gert::Tensor *tensor)
{
    return tensor == nullptr || tensor->GetData<T>() == nullptr;
}

uint32_t GetDim(const gert::TilingContext *context, size_t inputIdx, size_t dimIdx)
{
    return static_cast<uint32_t>(context->GetInputShape(inputIdx)->GetOriginShape().GetDim(dimIdx));
}

void SetWorkspace(gert::TilingContext *context)
{
    static auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = ascendcPlatform->GetLibApiWorkSpaceSize();
    }
}

uint32_t CalcQBlockRows(uint32_t gqaGroupSize)
{
    const uint32_t gqaTileSize = std::min(gqaGroupSize, MAX_GQA_TILE_SIZE);
    uint32_t qBlockRows = MAX_Q_ROWS_PER_GROUP / gqaTileSize;
    qBlockRows = qBlockRows / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
    return qBlockRows == 0 ? CUBE_BLOCK_SIZE : qBlockRows;
}

} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  PagedAttentionMixV3TilingData tiling;

  const auto *attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float softmaxScale = *(attrs->GetFloat(0));
  //* 从context获取各个维度参数
  const uint32_t numTokens = GetDim(context, 0, 0);
  const uint32_t qHeadNum = GetDim(context, 0, 1);
  const uint32_t qkHeadSize = GetDim(context, 0, 2);
  const uint32_t kHiddenBlocks = GetDim(context, 1, 1);
  const uint32_t pageSize = GetDim(context, 1, 2);
  const uint32_t kNzC0 = GetDim(context, 1, 3);
  const uint32_t vHiddenBlocks = GetDim(context, 2, 1);
  const uint32_t vPageSize = GetDim(context, 2, 2);
  const uint32_t vNzC0 = GetDim(context, 2, 3);
  const uint32_t maskColBlocks = GetDim(context, 3, 1);
  const uint32_t maskRowSize = GetDim(context, 3, 2);
  const uint32_t maskNzC0 = GetDim(context, 3, 3);
  const uint32_t maskColSize = maskColBlocks * NZ_C0_SIZE;
  const uint32_t pageNumPerBatch = GetDim(context, 4, 1);
  const uint32_t kvHiddenSize = kHiddenBlocks * kNzC0;
  const uint32_t kvHeadNum = qkHeadSize == 0 ? 0 : kvHiddenSize / qkHeadSize;
  const uint32_t vHiddenSize = vHiddenBlocks * vNzC0;
  const uint32_t vHeadSize = kvHeadNum == 0 ? 0 : vHiddenSize / kvHeadNum;

  /************** for debug ****************/
  /*
  std::printf(
      "[MixV3 tiling] numTokens=%u qHeadNum=%u qkHeadSize=%u "
      "pageSize=%u kvHeadNum=%u kvHiddenSize=%u "
      "vPageSize=%u vHiddenSize=%u vHeadSize=%u "
      "maskColSize=%u pageNumPerBatch=%u softmaxScale=%f\n",
      numTokens,
      qHeadNum,
      qkHeadSize,
      pageSize,
      kvHeadNum,
      kvHiddenSize,
      vPageSize,
      vHiddenSize,
      vHeadSize,
      maskColSize,
      pageNumPerBatch,
      softmaxScale
  );
  std::fflush(stdout);
  //*/

  // std::printf("HOST test test test !!!\n");
  // std::fflush(stdout);
  /************** for debug ****************/

  if (numTokens == 0 || maskColSize == 0 || maskRowSize == 0 || pageNumPerBatch == 0) {
    return ge::GRAPH_FAILED;
  }
  if (qHeadNum == 0 || kvHeadNum == 0 || qHeadNum % kvHeadNum != 0) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t gqaGroupSize = qHeadNum / kvHeadNum;
  if (kNzC0 != NZ_C0_SIZE || vNzC0 != NZ_C0_SIZE || kvHiddenSize % qkHeadSize != 0 ||
      vHiddenSize % kvHeadNum != 0 || pageSize != vPageSize) {
    return ge::GRAPH_FAILED;
  }
  if (gqaGroupSize > MAX_GQA_GROUP_SIZE) {
    return ge::GRAPH_FAILED;
  }
  if (pageSize == 0 || pageSize > 128 || pageSize % 16 != 0) {
    return ge::GRAPH_FAILED;
  }
  if (qkHeadSize == 0 || vHeadSize == 0 || qkHeadSize % 16 != 0 || vHeadSize % 16 != 0) {
    return ge::GRAPH_FAILED;
  }
  if (qkHeadSize > 256 || vHeadSize > 128) {
    return ge::GRAPH_FAILED;
  }
  if (maskNzC0 != NZ_C0_SIZE || maskRowSize % 16 != 0) {
    return ge::GRAPH_FAILED;
  }

  const auto *kvLengths = context->GetInputTensor(5);
  const auto *seqLenPerRequest = context->GetInputTensor(6);

  if (MissingConstTensor<int64_t>(kvLengths) || MissingConstTensor<int64_t>(seqLenPerRequest)) {
    context->SetBlockDim(1);
    tiling.set_batchSize(0);
    tiling.set_qHeadNum(qHeadNum);
    tiling.set_qkHeadSize(qkHeadSize);
    tiling.set_kvHeadNum(kvHeadNum);
    tiling.set_vHeadSize(vHeadSize);
    tiling.set_pageSize(pageSize);
    tiling.set_pageNumPerBatch(pageNumPerBatch);
    tiling.set_softmaxScale(softmaxScale);
    tiling.set_maskColSize(maskColSize);
    tiling.set_maskRowSize(maskRowSize);
    tiling.set_gqaGroupSize(gqaGroupSize);
    tiling.set_usedCoreNum(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    SetWorkspace(context);
    return ge::GRAPH_SUCCESS;
  }

  uint32_t batchSize = 0;
  uint32_t totalSeqLen = 0;
  const int64_t *seqLens = seqLenPerRequest->GetData<int64_t>();
  const int64_t seqLenCount = seqLenPerRequest->GetShapeSize();
  
  /************** for debug ****************/
  /*
  std::printf("[MixV3 tiling] seqLenCount=%ld", seqLenCount);
  for (int64_t i = 0; i < seqLenCount && i < 8; ++i) {
      std::printf(" seq[%ld]=%ld", i, seqLens[i]);
  }
  std::printf("\n");
  std::fflush(stdout);
  //*/
  /************** for debug ****************/
  
  for (int64_t i = 0; i < seqLenCount; ++i)     //* 依次遍历每个batch的seq len 检查合法性
  {
    if (seqLens[i] <= 0 || seqLens[i] > std::numeric_limits<uint32_t>::max()) {
      return ge::GRAPH_FAILED;
    }
    const uint32_t curSeqLen = static_cast<uint32_t>(seqLens[i]);
    if (curSeqLen > numTokens - totalSeqLen) {
      return ge::GRAPH_FAILED;
    }
    totalSeqLen += curSeqLen;
    ++batchSize;
    if (totalSeqLen == numTokens) 
    {
      break;
    }
  }
  if (batchSize == 0 || totalSeqLen != numTokens) 
  {
    return ge::GRAPH_FAILED;
  }
  if (totalSeqLen > maskRowSize) {
    return ge::GRAPH_FAILED;
  }

  const int64_t *kvLens = kvLengths->GetData<int64_t>();
  const int64_t kvLenCount = kvLengths->GetShapeSize();

  /************** for debug ****************/
  /*
  std::printf("[MixV3 tiling] kvLenCount=%ld", kvLenCount);
  for (int64_t i = 0; i < kvLenCount && i < 8; ++i) {
      std::printf(" kv[%ld]=%ld", i, kvLens[i]);
  }
  std::printf("\n");
  std::fflush(stdout);
  //*/
  /************** for debug ****************/

  if (kvLenCount < static_cast<int64_t>(batchSize)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t qBlockRows = CalcQBlockRows(gqaGroupSize);
  uint64_t totalTaskNum = 0;
  for (uint32_t i = 0; i < batchSize; ++i) {
    if (kvLens[i] <= 0 || kvLens[i] > std::numeric_limits<uint32_t>::max()) {
      return ge::GRAPH_FAILED;
    }
    const uint32_t curKvLen = static_cast<uint32_t>(kvLens[i]);
    const uint32_t curSeqLen = static_cast<uint32_t>(seqLens[i]);
    const uint32_t curKvPageNum = (curKvLen + pageSize - 1) / pageSize;
    if (curKvLen < curSeqLen || curKvPageNum * pageSize > maskColSize) {
      return ge::GRAPH_FAILED;
    }
    if (curKvPageNum > pageNumPerBatch) {
      return ge::GRAPH_FAILED;
    }
    totalTaskNum += static_cast<uint64_t>((curSeqLen + qBlockRows - 1) / qBlockRows) * kvHeadNum;
  }

  static auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const uint32_t coreNumAic = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
  const uint32_t maxCoreNum = coreNumAic == 0 ? 1 : coreNumAic;
  const uint32_t usedCoreNum =
      totalTaskNum == 0 ? 1 : static_cast<uint32_t>(totalTaskNum < maxCoreNum ? totalTaskNum : maxCoreNum);

  //* 将待传数据放在tiling中
  tiling.set_batchSize(batchSize);
  tiling.set_qHeadNum(qHeadNum);
  tiling.set_qkHeadSize(qkHeadSize);
  tiling.set_kvHeadNum(kvHeadNum);
  tiling.set_vHeadSize(vHeadSize);
  tiling.set_pageSize(pageSize);
  tiling.set_pageNumPerBatch(pageNumPerBatch);
  tiling.set_softmaxScale(softmaxScale);
  tiling.set_maskColSize(maskColSize);
  tiling.set_maskRowSize(maskRowSize);
  tiling.set_gqaGroupSize(gqaGroupSize);
  tiling.set_usedCoreNum(usedCoreNum);

  context->SetBlockDim(usedCoreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  SetWorkspace(context);

  return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{   // 推导输出 tensor 的形状
    // const gert::Shape* x1_shape = context->GetInputShape(0);
    // gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    const gert::Shape* queryShape = context->GetInputShape(0);
    const gert::Shape* keyShape = context->GetInputShape(1);
    const gert::Shape* valueShape = context->GetInputShape(2);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (queryShape == nullptr || outputShape == nullptr) {
      return GRAPH_FAILED;
    }
    *outputShape = *queryShape;
    if (keyShape != nullptr && valueShape != nullptr && queryShape->GetDimNum() >= 3 &&
        keyShape->GetDimNum() >= 4 && valueShape->GetDimNum() >= 4) {
      const int64_t qHeadSize = queryShape->GetDim(2);
      const int64_t kHiddenSize = keyShape->GetDim(1) * keyShape->GetDim(3);
      const int64_t kvHeadNum = qHeadSize == 0 ? 0 : kHiddenSize / qHeadSize;
      const int64_t vHiddenSize = valueShape->GetDim(1) * valueShape->GetDim(3);
      if (kvHeadNum > 0) {
        outputShape->SetDim(2, vHiddenSize / kvHeadNum);
      }
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class PagedAttentionMixV3 : public OpDef {
public:
    explicit PagedAttentionMixV3(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("kCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("vCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("attentionMask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("blockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kvLengths")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL, DependScope::TILING)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqLenPerRequest")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL, DependScope::TILING)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queryRope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("softmaxScale").Float();
        this->Attr("alibiMask").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(PagedAttentionMixV3);
}
