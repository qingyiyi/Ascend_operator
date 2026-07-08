#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)
remove_definitions(-DOpSpace=Mki)
file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/stubs/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/platform/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling_runner/*.cpp
)

set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
set(METADEF_DIR ${PROJECT_SOURCE_DIR}/3rdparty/metadef)
set(CANN_OPS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cann-ops-adv/src/transformer)
set(TBE_TILING_DIR ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling)
set(CANNDEV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops)
set(OPS_NN_DIR ${PROJECT_SOURCE_DIR}/3rdparty/ops-nn)
set(OPS_BASE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/opbase)
set(OPS_MATH_DIR ${PROJECT_SOURCE_DIR}/3rdparty/ops-math)
set(GE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/air)
set(DEVKIT_DIR ${PROJECT_SOURCE_DIR}/3rdparty/asc-devkit)

# 迁移后算子源文件
set(OPS_SOURCE
    ${OPS_NN_DIR}/index/scatter_elements_v2/op_host/scatter_elements_v2_tiling.cpp
    ${OPS_NN_DIR}/norm/layer_norm_v4/op_host/layer_norm_v4_tiling.cpp
    ${OPS_NN_DIR}/norm/layer_norm_v4/op_host/layer_norm_v4_tiling_base.cpp
    ${OPS_MATH_DIR}/common/src/op_host/tiling_util.cpp
    ${OPS_BASE_DIR}/src/op_common/atvoss/elewise/elewise_tiling.cpp
    ${OPS_BASE_DIR}/src/op_common/log/log.cpp
)

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(ASDOPS_METADEF_FILES
        ${METADEF_DIR}/base/any_value.cc
        ${METADEF_DIR}/base/asc/tilingdata_base_impl.cc
        ${METADEF_DIR}/base/type/ascend_string_impl.cc
        ${METADEF_DIR}/base/utils/aligned_ptr.cc
        ${METADEF_DIR}/base/utils/type_utils_impl.cc
        ${METADEF_DIR}/base/runtime/compute_node_info.cc
        ${METADEF_DIR}/base/runtime/runtime_attrs.cc
        ${METADEF_DIR}/base/runtime/tiling_data.cc
        ${METADEF_DIR}/base/registry/op_impl_registry.cc
        ${GE_DIR}/graph_metadef/exe_graph/lowering/getcdim.cc
        ${GE_DIR}/graph_metadef/exe_graph/lowering/shape_utils.cc
        ${GE_DIR}/graph_metadef/graph/attr/ge_attr_define.cc
        ${GE_DIR}/graph_metadef/graph/normal_graph/tensor.cc
        ${METADEF_DIR}/base/type/types_impl.cc
        ${GE_DIR}/graph_metadef/register/op_tiling/op_tiling_attr_utils.cc
        ${GE_DIR}/graph_metadef/register/op_tiling/op_tiling_info.cc
        ${GE_DIR}/graph_metadef/register/op_binary_resource_manager.cc
        ${GE_DIR}/graph_metadef/register/tuning_bank_key_registry.cc
        ${GE_DIR}/graph_metadef/register/tuning_tiling_registry.cc
        ${GE_DIR}/graph_metadef/third_party/transformer/src/expand_dimension.cc
        ${GE_DIR}/graph_metadef/third_party/transformer/src/transfer_shape_according_to_format.cc
        ${GE_DIR}/graph_metadef/third_party/transformer/src/transfer_shape_utils.cc
        ${GE_DIR}/graph_metadef/third_party/transformer/src/axis_util.cc
        ${GE_DIR}/graph_metadef/base/utils/aligned_ptr.cc
    )

    # Add tbe_tiling.cc here
    set(ASDOPS_TILING_FILES
        ${TBE_TILING_DIR}/op_tiling_util.cc
        ${TBE_TILING_DIR}/auto_tiling_context.cc
        ${TBE_TILING_DIR}/auto_tiling_rt2.cc
        ${TBE_TILING_DIR}/broadcast_v3.cc
        ${TBE_TILING_DIR}/cache_tiling.cc
        ${TBE_TILING_DIR}/compress_dequant_cache_tiling.cc
        ${TBE_TILING_DIR}/concat_dsl.cc
        ${TBE_TILING_DIR}/cube_tiling_runtime.cc
        ${TBE_TILING_DIR}/elewise_v3.cc
        ${TBE_TILING_DIR}/fusion.cc
        ${TBE_TILING_DIR}/gather_dsl.cc
        ${TBE_TILING_DIR}/gemm_ub_cache_tiling.cc
        ${TBE_TILING_DIR}/gemm.cc
        ${TBE_TILING_DIR}/lock.cc
        ${TBE_TILING_DIR}/norm.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3_compile_info.cc
        ${TBE_TILING_DIR}/slice_dsl.cc
        ${TBE_TILING_DIR}/sort_dsl.cc
        ${TBE_TILING_DIR}/split_dsl.cc
        ${TBE_TILING_DIR}/trans_data_fz2fzg.cc
        ${TBE_TILING_DIR}/trans_data_fzg_to_fz.cc
        ${TBE_TILING_DIR}/transdata_dsl.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04_backward.cc
        ${TBE_TILING_DIR}/transdata_dsl_entrance.cc
        ${TBE_TILING_DIR}/transdata_dsl_borrow.cc
        ${TBE_TILING_DIR}/transdata_dsl_general.cc
        ${TBE_TILING_DIR}/transdata_dsl_util.cc
        ${TBE_TILING_DIR}/vector_op_info.cc
        ${TBE_TILING_DIR}/vector_tiling_key.cc
        ${TBE_TILING_DIR}/vector_tiling_rt2.cc
        ${TBE_TILING_DIR}/vector_tiling_util.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/runtime_bank_manager.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_hash.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_runtime_bank.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/configuration.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/file_utils.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/system_utils.cc
        ${TBE_TILING_DIR}/cube/algorithm/cache_tiling_impl.cc
        ${TBE_TILING_DIR}/cube/algorithm/calculator/calculator.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/shape.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/status.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/cache_runinfo.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/hash.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/tiling_cache.cc
        ${TBE_TILING_DIR}/cube/impl/cache_tiling.cc
        ${TBE_TILING_DIR}/cube/impl/cube_run_info.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling_param.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling.cc
        ${TBE_TILING_DIR}/cube/platform/instruction_param.cc
        ${TBE_TILING_DIR}/cube/platform/platform_info.cc
        ${TBE_TILING_DIR}/cube/util/cube_util.cc
        ${TBE_TILING_DIR}/cube/util/math_util.cc
        ${TBE_TILING_DIR}/cube/util/timer.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block_calc.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_align_count.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_request_bytes.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_cycle_model.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_basic_block_est.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_est.cc
        ${CANNDEV_DIR}/conversion/as_strided/op_host/as_strided_tiling.cpp
        ${CANNDEV_DIR}/conversion/fill/op_host/fill_tiling.cpp
        ${CANNDEV_DIR}/math/cumsum/op_host/cumsum_tiling.cpp
        ${CANNDEV_DIR}/conversion/concat_d/op_host/concat_d_tiling.cpp
        ${CANNDEV_DIR}/conversion/concat/op_host/concat_tiling.cpp
        ${CANNDEV_DIR}/math/expand/op_host/expand_tiling.cpp
        ${TBE_TILING_DIR}/runtime/elewise_tiling.cc
        ${CANNDEV_DIR}/index/gather_v2/op_host/gather_v2_tiling.cpp
        ${CANNDEV_DIR}/index/inplace_index_add/op_host/inplace_index_add_tiling.cpp
        # Note: Only for 'layer_norm_v3_tiling_arch35.cpp' A2/3 and A5 use the same tiling.
        ${CANNDEV_DIR}/norm/layer_norm_v3/op_host/layer_norm_v3_tiling_arch35.cpp
        ${CANNDEV_DIR}/norm/layer_norm_v3/op_host/layer_norm_v3_tiling.cpp
        ${CANNDEV_DIR}/norm/layer_norm_v3/op_host/layer_norm_v3_tiling_base.cpp
        ${CANNDEV_DIR}/math/one_hot/op_host/one_hot_tiling.cpp
        ${CANNDEV_DIR}/index/reverse_v2/op_host/reverse_v2_tiling.cpp
        ${CANNDEV_DIR}/index/reverse_v2/op_host/reverse_v2_tiling_arch35.cpp
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${CANNDEV_DIR}/conversion/slice/op_host/slice_tiling.cpp
        ${TBE_TILING_DIR}/runtime/top_k.cc
        ${TBE_TILING_DIR}/runtime/trans_data.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_ntc.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_tc_201.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_ntc_100.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1010.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1011.cc
        ${CANNDEV_DIR}/conversion/transpose/op_host/transpose_tiling.cpp
        ${CANNDEV_DIR}/conversion/view_copy/op_host/view_copy_tiling.cpp
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/../fusion_pass/common/fp16_t.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/op_util.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/getcdim.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_cycle_est.cc
    )
else()
    set(ASDOPS_METADEF_FILES
        ${METADEF_DIR}/exe_graph/lowering/getcdim.cc
        ${METADEF_DIR}/exe_graph/lowering/shape_utils.cc
        ${METADEF_DIR}/exe_graph/runtime/compute_node_info.cc
        ${METADEF_DIR}/exe_graph/runtime/runtime_attrs.cc
        ${METADEF_DIR}/exe_graph/runtime/tiling_data.cc
        ${METADEF_DIR}/graph/attr/ge_attr_define.cc
        ${METADEF_DIR}/graph/normal_graph/any_value.cc
        ${METADEF_DIR}/graph/normal_graph/tensor.cc
        ${METADEF_DIR}/graph/type/ascend_string.cc
        ${METADEF_DIR}/graph/type/types.cc
        ${METADEF_DIR}/graph/utils/aligned_ptr.cc
        ${METADEF_DIR}/graph/utils/type_utils.cc
        ${METADEF_DIR}/graph/utils/type_utils_ex.cc
        ${METADEF_DIR}/register/op_impl_registry.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_attr_utils.cc
        ${METADEF_DIR}/register/op_tiling/op_tiling_info.cc
        ${METADEF_DIR}/register/op_binary_resource_manager.cc
        ${METADEF_DIR}/register/tuning_bank_key_registry.cc
        ${METADEF_DIR}/register/tuning_tiling_registry.cc
        ${METADEF_DIR}/register/ascendc/tilingdata_base.cc
        ${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc
        ${METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc
        ${METADEF_DIR}/third_party/transformer/src/axis_util.cc
    )

    # Add tbe_tiling.cc here
    set(ASDOPS_TILING_FILES
        ${TBE_TILING_DIR}/auto_tiling_context.cc
        ${TBE_TILING_DIR}/auto_tiling_rt2.cc
        ${TBE_TILING_DIR}/broadcast_v3.cc
        ${TBE_TILING_DIR}/cache_tiling.cc
        ${TBE_TILING_DIR}/compress_dequant_cache_tiling.cc
        ${TBE_TILING_DIR}/concat_dsl.cc
        ${TBE_TILING_DIR}/cube_tiling_runtime.cc
        ${TBE_TILING_DIR}/elewise_v3.cc
        ${TBE_TILING_DIR}/fusion.cc
        ${TBE_TILING_DIR}/gather_dsl.cc
        ${TBE_TILING_DIR}/gemm_ub_cache_tiling.cc
        ${TBE_TILING_DIR}/gemm.cc
        ${TBE_TILING_DIR}/lock.cc
        ${TBE_TILING_DIR}/norm.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3.cc
        ${TBE_TILING_DIR}/reduce_tiling_v3_compile_info.cc
        ${TBE_TILING_DIR}/slice_dsl.cc
        ${TBE_TILING_DIR}/sort_dsl.cc
        ${TBE_TILING_DIR}/split_dsl.cc
        ${TBE_TILING_DIR}/trans_data_fz2fzg.cc
        ${TBE_TILING_DIR}/trans_data_fzg_to_fz.cc
        ${TBE_TILING_DIR}/transdata_dsl.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04.cc
        ${TBE_TILING_DIR}/transdata_dsl_c04_backward.cc
        ${TBE_TILING_DIR}/transdata_dsl_entrance.cc
        ${TBE_TILING_DIR}/transdata_dsl_borrow.cc
        ${TBE_TILING_DIR}/transdata_dsl_general.cc
        ${TBE_TILING_DIR}/transdata_dsl_util.cc
        ${TBE_TILING_DIR}/vector_op_info.cc
        ${TBE_TILING_DIR}/vector_tiling_key.cc
        ${TBE_TILING_DIR}/vector_tiling_rt2.cc
        ${TBE_TILING_DIR}/vector_tiling_util.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/runtime_bank_manager.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_hash.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/op_runtime_bank.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/configuration.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/file_utils.cc
        ${TBE_TILING_DIR}/aoe/runtime_kb/common/utils/system_utils.cc
        ${TBE_TILING_DIR}/cube/algorithm/cache_tiling_impl.cc
        ${TBE_TILING_DIR}/cube/algorithm/calculator/calculator.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/shape.cc
        ${TBE_TILING_DIR}/cube/algorithm/entity/status.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/cache_runinfo.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/hash.cc
        ${TBE_TILING_DIR}/cube/algorithm/hash/tiling_cache.cc
        ${TBE_TILING_DIR}/cube/impl/cache_tiling.cc
        ${TBE_TILING_DIR}/cube/impl/cube_run_info.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling_param.cc
        ${TBE_TILING_DIR}/cube/impl/cube_tiling.cc
        ${TBE_TILING_DIR}/cube/platform/instruction_param.cc
        ${TBE_TILING_DIR}/cube/platform/platform_info.cc
        ${TBE_TILING_DIR}/cube/util/cube_util.cc
        ${TBE_TILING_DIR}/cube/util/math_util.cc
        ${TBE_TILING_DIR}/cube/util/timer.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block.cc
        ${TBE_TILING_DIR}/gemm/cache_tiling_basic_block_calc.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_align_count.cc
        ${TBE_TILING_DIR}/gemm/common/cache_tiling_request_bytes.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_cycle_model.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_basic_block_est.cc
        ${TBE_TILING_DIR}/gemm/estimate/cache_tiling_est.cc
        ${TBE_TILING_DIR}/runtime/as_strided.cc
        ${TBE_TILING_DIR}/runtime/as_strided_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc.cc
        ${TBE_TILING_DIR}/runtime/cumsum_tiling_ascendc_int.cc
        ${TBE_TILING_DIR}/runtime/concat_d.cc
        ${TBE_TILING_DIR}/runtime/concat_tiling.cc
        ${TBE_TILING_DIR}/runtime/expand.cc
        ${TBE_TILING_DIR}/runtime/elewise_tiling.cc
        ${TBE_TILING_DIR}/runtime/elementwise_template/fill_tiling.cc
        ${TBE_TILING_DIR}/runtime/gatherv2.cc
        ${TBE_TILING_DIR}/runtime/gather_v2_tiling.cpp
        ${TBE_TILING_DIR}/runtime/inplace_index_add.cc
        ${TBE_TILING_DIR}/runtime/inplace_index_add_tiling.cc
        ${TBE_TILING_DIR}/runtime/layer_norm_v3.cc
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot.cc
        ${TBE_TILING_DIR}/runtime/one_hot/one_hot_tiling.cc
        ${TBE_TILING_DIR}/runtime/pack_tiling.cc
        ${TBE_TILING_DIR}/runtime/reverse.cc
        ${TBE_TILING_DIR}/runtime/reverse_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/tensor_move_tiling.cc
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/runtime/slice.cc
        ${TBE_TILING_DIR}/runtime/slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/strided_slice_tiling.cc
        ${TBE_TILING_DIR}/runtime/top_k.cc
        ${TBE_TILING_DIR}/runtime/top_k_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/trans_data.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_ntc.cc
        ${TBE_TILING_DIR}/runtime/trans_data_negative_target_tc_201.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_ntc_100.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1010.cc
        ${TBE_TILING_DIR}/runtime/trans_data_positive_source_tc_1011.cc
        ${TBE_TILING_DIR}/runtime/transdata_tiling.cc
        ${TBE_TILING_DIR}/runtime/transpose.cc
        ${TBE_TILING_DIR}/runtime/transpose_tiling.cc
        ${TBE_TILING_DIR}/runtime/depth_to_space_tiling.cc
        ${TBE_TILING_DIR}/runtime/space_to_depth_tiling.cc
        ${TBE_TILING_DIR}/runtime/view_copy.cc
        ${TBE_TILING_DIR}/runtime/view_copy_tiling.cc
        ${TBE_TILING_DIR}/runtime/scatter_elements_v2_tiling.cc
        ${TBE_TILING_DIR}/runtime/runtime2_util.cc
        ${TBE_TILING_DIR}/../fusion_pass/common/fp16_t.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/op_util.cc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/src/getcdim.cc
    )
endif()

# Sink kernels
file(GLOB_RECURSE SINK_TILING_SRCS
    ${TBE_TILING_DIR}/runtime/rotary_pos_emb_infer/*
)
list(APPEND ASDOPS_TILING_FILES ${SINK_TILING_SRCS})

# Tiling Api
set(TILING_API_BASE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/api)
set(TILING_API_SRC_DIR ${PROJECT_SOURCE_DIR}/3rdparty/api/impl)

# Generate kernel_tiling.h
set(GEN_KERENL_TILING_DATA_SCRIPT ${DEVKIT_DIR}/impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py)
set(TILING_DATA_DEF_DIR ${DEVKIT_DIR}/lib)
set(KERNEL_TILING_HEAD  ${CMAKE_BINARY_DIR}/generated_include/kernel_tiling/kernel_tiling.h)

add_custom_command(OUTPUT ${KERNEL_TILING_HEAD}
    COMMAND python3 ${GEN_KERENL_TILING_DATA_SCRIPT} ${TILING_DATA_DEF_DIR} ${KERNEL_TILING_HEAD}
    DEPENDS ${GEN_KERENL_TILING_DATA_SCRIPT})

add_custom_target(GEN_KERNEL_TILING ALL
    DEPENDS ${KERNEL_TILING_HEAD})

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(TILING_API_SRCS
        ${DEVKIT_DIR}/impl/adv_api/tiling/quantization/ascend_dequant_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/quantization/ascend_quant_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/quantization/ascend_antiquant_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/filter/dropout_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/gelu_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/matmul/bmm_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/matmul/matmul_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/matmul/matmul_tiling_base.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/matmul/matmul_tiling_algorithm.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/matmul/math_util.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/hccl/hccl_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/clamp_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/acos_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/acosh_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/asin_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/asinh_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/atan_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/atanh_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/cos_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/cosh_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/erf_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/erfc_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/exp_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/frac_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/geglu_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/lgamma_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/digamma_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/log_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/sin_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/sinh_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/power_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/sigmoid_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/round_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/tan_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/tanh_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/trunc_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/axpy_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/hypot_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/swiglu_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/ceil_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/floor_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/softmax_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/logsoftmax_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/rmsnorm_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/batchnorm_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/sort/sort_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/sort/topk_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/deepnorm_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/select/selectwithbytesmask_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/layernorm_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/normalize_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/layernorm_grad_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/layernorm_grad_beta_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/groupnorm_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/normalization/welfordfinalize_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/pad/pad_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/transpose/confusion_transpose_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/pad/broadcast_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/pad/broadcast_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/xor_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/cumsum_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/reduce/mean_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/sign_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/activation/reglu_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/reduce/reduce_xor_sum_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/reduce/sum_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/reduce/reduce_tiling.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/index/arithprogression_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/tiling/math/fmod_tiling_impl.cpp
        ${DEVKIT_DIR}/impl/adv_api/detail/host_log.cpp
    )
else()
    set(TILING_API_SRCS
        ${TILING_API_SRC_DIR}/quantization/dequant/ascend_dequant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/quant/ascend_quant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/quantization/antiquant/ascend_antiquant_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/filter/dropout/dropout_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/gelu/gelu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/bmm_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_base.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/matmul_tiling_algorithm.cpp
        ${TILING_API_SRC_DIR}/matmul/tiling/math_util.cpp
        ${TILING_API_SRC_DIR}/hccl/hccl_tiling.cpp
        ${TILING_API_SRC_DIR}/math/clamp/clamp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acos/acos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/acosh/acosh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/asin/asin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/asinh/asinh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/atan/atan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/atanh/atanh_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cos/cos_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/cosh/cosh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erf/erf_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/erfc/erfc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/exp/exp_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/frac/frac_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/geglu/geglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/lgamma/lgamma_tiling.cpp
        ${TILING_API_SRC_DIR}/math/digamma/digamma_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/log/log_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sin/sin_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/sinh/sinh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/power/power_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/sigmoid/sigmoid_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/round/round_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tan/tan_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/tanh/tanh_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/trunc/trunc_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/axpy/axpy_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/hypot/hypot_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/swiglu/swiglu_tiling.cpp
        ${TILING_API_SRC_DIR}/math/ceil/ceil_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/floor/floor_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/softmax_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/softmax/logsoftmax_tiling.cpp
        ${TILING_API_SRC_DIR}/normalization/rmsnorm/rmsnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/batchnorm/batchnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/sort/sort_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/sort/topk/topk_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/deepnorm/deepnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/select/selectwithbytesmask/selectwithbytesmask_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernorm/layernorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/normalize/normalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/layernormgrad/layernorm_grad_beta_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/groupnorm/groupnorm_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/normalization/welfordfinalize/welfordfinalize_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/pad/pad_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/transpose/confusion_transpose/confusion_transpose_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/pad/broadcast/broadcast_tiling.cpp
        ${TILING_API_SRC_DIR}/math/xor/xor_tiling.cpp
        ${TILING_API_SRC_DIR}/math/cumsum/cumsum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/mean/mean_tiling.cpp
        ${TILING_API_SRC_DIR}/math/sign/sign_tiling.cpp
        ${TILING_API_SRC_DIR}/activation/reglu/reglu_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_xor_sum/reduce_xor_sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/sum/sum_tiling.cpp
        ${TILING_API_SRC_DIR}/reduce/reduce_tiling.cpp
        ${TILING_API_SRC_DIR}/index/arithprogression/arithprogression_tiling_impl.cpp
        ${TILING_API_SRC_DIR}/math/fmod/fmod_tiling_impl.cpp
    )
endif()

if("${PACKAGE_COMPILE}" STREQUAL "ON" AND NOT "${LOCAL_RELEASE_COMPILE}" STREQUAL "ON")
    set(ASDOPS_INC_DIRS
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/platform
        ${CMAKE_CURRENT_LIST_DIR}/platform/tiling
        ${KERNEL_TILING_DIR}
        ${DEVKIT_DIR}
        ${DEVKIT_DIR}/tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/conversion/fill/op_kernel
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/ophost_comm/op_tiling
        ${METADEF_DIR}/inc/external/graph
        ${GE_DIR}/graph_metadef
        ${GE_DIR}/inc/graph_metadef
        ${GE_DIR}/inc/graph_metadef/base/runtime
        ${GE_DIR}/graph_metadef/exe_graph/lowering
        ${GE_DIR}/base/metadef/pkg_inc/exe_graph/runtime
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc/graph
        ${GE_DIR}/inc/graph_metadef/graph
        ${GE_DIR}/base/metadef/pkg_inc
        ${METADEF_DIR}
        ${METADEF_DIR}/pkg_inc
        ${METADEF_DIR}/inc
        ${GE_DIR}/inc
        ${GE_DIR}/base/metadef/pkg_inc/common
        ${METADEF_DIR}/inc/exe_graph
        ${GE_DIR}/inc/graph_metadef/external
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/external/exe_graph
        ${METADEF_DIR}/inc/external/exe_graph/lowering
        ${METADEF_DIR}/inc/external/exe_graph/runtime
        ${GE_DIR}/base/metadef/pkg_inc/graph
        ${GE_DIR}/graph_metadef/register
        ${GE_DIR}/graph_metadef/third_party/transformer/inc
        ${GE_DIR}/graph_metadef/third_party/transformer/src
        ${METADEF_DIR}/inc/common/ge_common/debug
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/utils/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling/runtime
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/third_party/fwkacllib/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/third_party/fwkacllib/inc/toolchain
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/tiling
        $ENV{ASCEND_HOME_PATH}/${ARCH}-linux
        $ENV{ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc
        $ENV{ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/base
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/mki
    )
else()
    set(ASDOPS_INC_DIRS
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc
        ${CMAKE_CURRENT_LIST_DIR}/stubs/include/metadef/inc/graph
        ${CMAKE_CURRENT_LIST_DIR}/platform
        ${CMAKE_CURRENT_LIST_DIR}/platform/tiling
        ${KERNEL_TILING_DIR}
        ${TILING_API_BASE_DIR}
        ${TILING_API_BASE_DIR}/tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/conversion/fill/op_kernel
        ${METADEF_DIR}
        ${METADEF_DIR}/exe_graph/lowering
        ${METADEF_DIR}/exe_graph/runtime
        ${METADEF_DIR}/graph
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/common
        ${METADEF_DIR}/inc/exe_graph
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/inc/external/exe_graph
        ${METADEF_DIR}/inc/external/exe_graph/lowering
        ${METADEF_DIR}/inc/external/exe_graph/runtime
        ${METADEF_DIR}/inc/external/graph
        ${METADEF_DIR}/register
        ${METADEF_DIR}/third_party/transformer/inc
        ${METADEF_DIR}/inc/common/ge_common/debug
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/utils/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/built-in/op_tiling/runtime
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/ops/common/inc
        ${PROJECT_SOURCE_DIR}/3rdparty/canndev/third_party/fwkacllib/inc
        ${METADEF_DIR}/inc/graph
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include/tiling
    )
endif()

# Ignore warnings from 3rdparty srcs
# DO NOT USE THIS FOR OTHER SRCS
set_source_files_properties(
    ${TILING_API_SRCS}
    ${ASDOPS_TILING_FILES}
    ${ASDOPS_METADEF_FILES}
    ${OPS_SOURCE}
    PROPERTIES
    COMPILE_FLAGS "-w"
)

# MIX OPS
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/mixops/)
execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/update_tbe_tactic_json.py
                                --src_ini_path ${PROJECT_SOURCE_DIR}/src/kernels/configs/mixops/tbe_tactic_info.ini
                                --dst_ini_path ${CMAKE_BINARY_DIR}/mixops/tbe_tactic_json.ini
                OUTPUT_VARIABLE MIX_PYTHON_OUTPUT
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "tbe info update failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()
 
string(REPLACE "\n" ";" MIX_REUSE_BINARY_LIST "${MIX_PYTHON_OUTPUT}")
list(POP_BACK MIX_REUSE_BINARY_LIST)
set_source_files_properties(${MIX_REUSE_BINARY_LIST} PROPERTIES GENERATED TRUE)
 
add_custom_command(
    OUTPUT ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
    DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/configs/mixops/tbe_tactic_info.ini
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type tbe
                    --tbe_ini_path ${CMAKE_BINARY_DIR}/mixops/tbe_tactic_json.ini
    COMMAND cmake -E sleep 10
    COMMAND cmake -E echo "wait 10 sec done"
    COMMAND cmake -E touch ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
)
 
add_custom_target(MIX_REUSE_SRC_TARGET ALL
    DEPENDS ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp
)
add_library(mix_reuse_kernels OBJECT ${MIX_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/mix_wait_flag.cpp)
add_dependencies(mix_reuse_kernels MIX_REUSE_SRC_TARGET)
target_compile_definitions(mix_reuse_kernels PRIVATE OpSpace=AtbOps)
 
# Ops
execute_process(COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/update_tbe_tactic_json.py
                OUTPUT_VARIABLE OPS_PYTHON_OUTPUT
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "tbe info update failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()
 
string(REPLACE "\n" ";" OPS_REUSE_BINARY_LIST "${OPS_PYTHON_OUTPUT}")
list(POP_BACK OPS_REUSE_BINARY_LIST)
set_source_files_properties(${OPS_REUSE_BINARY_LIST} PROPERTIES GENERATED TRUE)
 
add_custom_command(
    OUTPUT ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
    DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/configs/ops/tbe_tactic_info.ini
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type tbe
                    --tbe_ini_path ${CMAKE_BINARY_DIR}/tbe_tactic_json.ini
    COMMAND cmake -E sleep 10
    COMMAND cmake -E echo "wait 10 sec done"
    COMMAND cmake -E touch ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
)
 
add_custom_target(OPS_REUSE_SRC_TARGET ALL
    DEPENDS ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp
)
add_library(ops_reuse_kernels OBJECT ${OPS_REUSE_BINARY_LIST} ${CMAKE_BINARY_DIR}/ops_wait_flag.cpp)
add_dependencies(ops_reuse_kernels OPS_REUSE_SRC_TARGET)
target_compile_definitions(ops_reuse_kernels PRIVATE OpSpace=AsdOps)

# 部分算子源码使用的是迁移后的算子仓的源码，这部分代码新创建一个.o对象进行编译设置
add_library(tbe_adapter_ops OBJECT ${OPS_SOURCE})
target_include_directories(tbe_adapter_ops PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/platform
    ${CMAKE_CURRENT_LIST_DIR}/platform/tiling
    ${KERNEL_TILING_DIR}
    ${DEVKIT_DIR}
    ${DEVKIT_DIR}/tiling
    ${OPS_BASE_DIR}/include/op_common/op_host
    ${OPS_BASE_DIR}/include/op_common
    ${OPS_BASE_DIR}/include
    ${OPS_BASE_DIR}/pkg_inc
    ${OPS_BASE_DIR}/pkg_inc/op_common
    ${OPS_BASE_DIR}/pkg_inc/op_common/op_host
    ${OPS_NN_DIR}
    ${OPS_NN_DIR}/common/inc
    ${OPS_MATH_DIR}
    ${OPS_MATH_DIR}/common/inc
    ${METADEF_DIR}/inc/external
    ${CANNDEV_DIR}/built-in/op_tiling
    ${TBE_TILING_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/tiling
    $ENV{ASCEND_HOME_PATH}/include/toolchain
    $ENV{ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/base
)
target_compile_definitions(tbe_adapter_ops PUBLIC
    fe=AsdOpsFe
    ge=AsdOpsGe
    gert=AsdOpsGeRt
    optiling=AsdOpsTiling
    error_message=AsdOpsErrorMessage
    ErrorManager=AsdOpsErrorManager
    td_dsl=AsdOpsTdDsl
    tuningtiling=AsdOpsTuningTiling
    RuntimeKb=AsdOpsRuntimeKb
    gemm_cache_tiling=AsdOpsGemmCacheTiling
    transformer=AsdOpsTransformer
    ops=AsdOpsOps
    AscendC=AsdOpsAscendC
    matmul_tiling=AsdOpsMatmulTiling
    platform_ascendc=AsdOpsPlatformAscendC
    TILING_REG_FLAG=1
)
 
# Target
add_library(tbe_adapter SHARED ${SOURCE_FILES} ${ASDOPS_METADEF_FILES} ${ASDOPS_TILING_FILES} ${TILING_API_SRCS})
target_compile_options(tbe_adapter PRIVATE -Wno-error) # For redefine of "dlog_error" in dlog_pub.h and slog.h
add_dependencies(tbe_adapter GEN_KERNEL_TILING)
target_include_directories(tbe_adapter PRIVATE ${ASDOPS_INC_DIRS})
target_link_libraries(tbe_adapter PRIVATE mmpa c_sec ascendalog mix_reuse_kernels ops_reuse_kernels tbe_adapter_ops tiling_api)

add_definitions(-DLOG_CPP)

target_compile_definitions(tbe_adapter PUBLIC
    OPS_UTILS_LOG_SUB_MOD_NAME="ASDOPS_TILING"
)

target_compile_definitions(tbe_adapter PUBLIC
    fe=AsdOpsFe
    ge=AsdOpsGe
    gert=AsdOpsGeRt
    optiling=AsdOpsTiling
    error_message=AsdOpsErrorMessage
    ErrorManager=AsdOpsErrorManager
    td_dsl=AsdOpsTdDsl
    tuningtiling=AsdOpsTuningTiling
    RuntimeKb=AsdOpsRuntimeKb
    gemm_cache_tiling=AsdOpsGemmCacheTiling
    transformer=AsdOpsTransformer
    ops=AsdOpsOps
    AscendC=AsdOpsAscendC
    matmul_tiling=AsdOpsMatmulTiling
    platform_ascendc=AsdOpsPlatformAscendC
)

install(TARGETS tbe_adapter DESTINATION lib)
