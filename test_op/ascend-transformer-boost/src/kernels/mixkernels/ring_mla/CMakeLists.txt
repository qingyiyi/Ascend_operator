# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(mla_srcs
        ${CMAKE_CURRENT_LIST_DIR}/ring_mla_operation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ring_mla_kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tiling/ring_mla_tiling.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tiling/ring_mla_tiling_dependency.cpp
)

add_operation(RINGMLAOperation "${mla_srcs}")

add_kernel(ring_mla_prefill ascend910b mix
        op_kernel/ring_mla_prefill.cce
    RINGMLAPrefillKernel)

add_kernel(ring_mla_prefill_bf16 ascend910b mix
        op_kernel/ring_mla_prefill_bf16.cce
    RINGMLAPrefillBF16Kernel)
