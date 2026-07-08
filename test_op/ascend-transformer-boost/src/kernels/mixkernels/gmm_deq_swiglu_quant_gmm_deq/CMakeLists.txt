# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(gmm_deq_swiglu_quant_gmm_deq_srcs
    ${CMAKE_CURRENT_LIST_DIR}/gmm_deq_swiglu_quant_gmm_deq_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gmm_deq_swiglu_quant_gmm_deq_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/gmm_deq_swiglu_quant_gmm_deq_tiling.cpp
)

add_operation(GmmDeqSwigluQuantGmmDeqOperation "${gmm_deq_swiglu_quant_gmm_deq_srcs}")

add_kernel(
    gmm_deq_swiglu_quant_gmm_deq_n256 ascend910b mix
    op_kernel/gmm_deq_swiglu_quant_gmm_deq_n256.cpp
    GmmDeqSwigluQuantGmmDeqN256Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
                        ${OPS_PROJECT_ROOT_DIR}/mixkernels/utils/common/kernel/catlass
)

add_kernel(
    gmm_deq_swiglu_quant_gmm_deq_n128 ascend910b mix
    op_kernel/gmm_deq_swiglu_quant_gmm_deq_n128.cpp
    GmmDeqSwigluQuantGmmDeqN128Kernel
    INCLUDE_DIRECTORIES ${OPS_THIRD_PARTY_DIR}/catlass/include
                        ${OPS_PROJECT_ROOT_DIR}/mixkernels/utils/common/kernel/catlass
)
