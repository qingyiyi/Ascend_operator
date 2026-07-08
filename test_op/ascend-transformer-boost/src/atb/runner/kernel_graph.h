/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_KERNEL_GRAPH_H
#define ATB_KERNEL_GRAPH_H

#include <functional>
#include <memory>
#include <mki/op_desc.h>
#include "atb/types.h"
#include "atb/context/context_base.h"
#include "atb/svector.h"
#include "atb/runner/atb_kernel_method.h"

namespace atb {
struct KernelGraphNode {
    Mki::OpDesc opDesc;
    SVector<Mki::Tensor *> inTensors;
    SVector<Mki::Tensor *> outTensors;
    SVector<ViewFunc> inTensorViewFuncs;
    AsdOpsInferShapePreFunc inferShapePreFunc = nullptr;
    MkiInferShapePreFunc mkiInferShapePreFunc = nullptr;
    bool tilingCacheEnable = true;
    SVector<TensorType> inTensorsType;
    SVector<TensorType> outTensorsType;
    std::shared_ptr<AtbKernelMethod> impl;
    bool CreateImplement();
    void Reset(); // reset runInfo
    std::string GetName() const;
};

struct KernelGraph {
    SVector<Mki::Tensor> inTensors;
    SVector<Mki::Tensor> outTensors;
    SVector<Mki::Tensor> internalTensors;
    std::vector<KernelGraphNode> nodes;
    std::string ToString() const;
    void Init();

private:
    bool IsInternalTensor(const Mki::Tensor *tensor) const;
};

} // namespace atb
#endif
