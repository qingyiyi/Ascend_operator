#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace paged_attention_mix_v3_ext {
namespace {

void CheckInputs(const at::Tensor &query,
                 const at::Tensor &kCache,
                 const at::Tensor &vCache,
                 const at::Tensor &attentionMask,
                 const at::Tensor &blockTable,
                 const at::Tensor &kvLengths,
                 const at::Tensor &seqLenPerRequest)
{
    TORCH_CHECK(query.dim() == 3, "query must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(kCache.dim() == 4, "k_cache must be NZ [num_blocks, hidden_blocks, page_size, 16]");
    TORCH_CHECK(vCache.dim() == 4, "v_cache must be NZ [num_blocks, hidden_blocks, page_size, 16]");
    TORCH_CHECK(attentionMask.dim() == 4, "attention_mask must be NZ [1, mask_col_blocks, mask_rows_pad, 16]");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be [batch, pages_per_batch]");
    TORCH_CHECK(kvLengths.dim() == 1, "kv_lengths must be a 1-D int64 NPU tensor");
    TORCH_CHECK(seqLenPerRequest.dim() == 1, "seq_len_per_request must be a 1-D int64 NPU tensor");

    TORCH_CHECK(query.scalar_type() == at::kHalf, "query dtype must be float16");
    TORCH_CHECK(kCache.scalar_type() == at::kHalf, "k_cache dtype must be float16");
    TORCH_CHECK(vCache.scalar_type() == at::kHalf, "v_cache dtype must be float16");
    TORCH_CHECK(attentionMask.scalar_type() == at::kHalf, "attention_mask dtype must be float16");
    TORCH_CHECK(blockTable.scalar_type() == at::kInt, "block_table dtype must be int32");
    TORCH_CHECK(kvLengths.scalar_type() == at::kLong, "kv_lengths dtype must be int64");
    TORCH_CHECK(seqLenPerRequest.scalar_type() == at::kLong, "seq_len_per_request dtype must be int64");

    TORCH_CHECK(query.size(1) > 0, "query num_heads must be positive");
    TORCH_CHECK(query.size(2) > 0, "query head_size must be positive");
    TORCH_CHECK(kCache.size(1) > 0, "k_cache hidden_blocks must be positive");
    TORCH_CHECK(kCache.size(2) > 0, "k_cache page_size must be positive");
    TORCH_CHECK(kCache.size(3) == 16, "k_cache last NZ dim must be 16");
    TORCH_CHECK(vCache.size(1) > 0, "v_cache hidden_blocks must be positive");
    TORCH_CHECK(vCache.size(3) == 16, "v_cache last NZ dim must be 16");
    TORCH_CHECK(attentionMask.size(0) == 1, "attention_mask NZ batch dim must be 1");
    TORCH_CHECK(attentionMask.size(1) > 0, "attention_mask NZ col blocks must be positive");
    TORCH_CHECK(attentionMask.size(2) > 0, "attention_mask NZ row pad must be positive");
    TORCH_CHECK(attentionMask.size(3) == 16, "attention_mask last NZ dim must be 16");
    TORCH_CHECK(attentionMask.size(2) % 16 == 0, "attention_mask NZ row pad must be aligned to 16");
    TORCH_CHECK(attentionMask.size(2) >= query.size(0),
                "attention_mask NZ row pad must cover query num_tokens");
    TORCH_CHECK((kCache.size(1) * kCache.size(3)) % query.size(2) == 0,
                "k_cache hidden size must be divisible by query head_size");
    const auto numKvHeads = (kCache.size(1) * kCache.size(3)) / query.size(2);
    TORCH_CHECK(numKvHeads > 0, "k_cache derived num_kv_heads must be positive");
    TORCH_CHECK((vCache.size(1) * vCache.size(3)) % numKvHeads == 0,
                "v_cache hidden size must be divisible by num_kv_heads");
    TORCH_CHECK(kCache.size(0) == vCache.size(0), "k_cache and v_cache must have the same num_blocks");
    TORCH_CHECK(kCache.size(2) == vCache.size(2), "k_cache and v_cache must have the same page_size");
    TORCH_CHECK(query.size(1) % numKvHeads == 0, "query num_heads must be divisible by num_kv_heads");
}

} // namespace

at::Tensor RunPagedAttentionMixV3(const at::Tensor &query,
                                  const at::Tensor &kCache,
                                  const at::Tensor &vCache,
                                  const at::Tensor &attentionMask,
                                  const at::Tensor &blockTable,
                                  const at::Tensor &kvLengths,
                                  const at::Tensor &seqLenPerRequest,
                                  const c10::optional<at::Tensor> &queryRope,
                                  double softmaxScale,
                                  bool alibiMask)
{
    CheckInputs(query, kCache, vCache, attentionMask, blockTable, kvLengths, seqLenPerRequest);

    const auto numKvHeads = (kCache.size(1) * kCache.size(3)) / query.size(2);
    const auto vHeadSize = (vCache.size(1) * vCache.size(3)) / numKvHeads;
    auto output = at::empty({query.size(0), query.size(1), vHeadSize}, query.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("PagedAttentionMixV3")
        .Input(query)
        .Input(kCache)
        .Input(vCache)
        .Input(attentionMask)
        .Input(blockTable)
        .Input(kvLengths)
        .Input(seqLenPerRequest);

    if (queryRope.has_value() && queryRope.value().defined()) {
        cmd.Input(queryRope.value());
    } else {
        cmd.Input();
    }

    cmd.Output(output)
       .Attr("softmaxScale", static_cast<float>(softmaxScale))
       .Attr("alibiMask", alibiMask)
       .Run();

    return output;
}

} // namespace paged_attention_mix_v3_ext

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("paged_attention_mix_v3(Tensor query, Tensor k_cache, Tensor v_cache, "
          "Tensor attention_mask, Tensor block_table, Tensor kv_lengths, "
          "Tensor seq_len_per_request, Tensor? query_rope, float softmax_scale, "
          "bool alibi_mask) -> Tensor");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("paged_attention_mix_v3", TORCH_FN(paged_attention_mix_v3_ext::RunPagedAttentionMixV3));
}
} // namespace
