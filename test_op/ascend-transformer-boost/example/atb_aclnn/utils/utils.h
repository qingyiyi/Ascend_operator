#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"
#include "utils/log.h"

#define CHECK_RET(cond, str) \
    do                       \
    {                        \
        if (cond)            \
        {                    \
            LOG_ERROR(str);  \
            exit(0);         \
        }                    \
    } while (0)

// 设置各个输入tensor的属性
void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs);

// 设置各个输入tensor并且为各个输入tensor分配内存空间，此处的输入tensor为手动设置，工程实现上可以使用torchTensor转换或者其他简单数据结构转换的方式
void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs);

// 设置各个输出tensor并且为输出tensor分配内存空间，同输入tensor设置
void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs);

void CreateTensorFromDesc(atb::Tensor &tensor, atb::TensorDesc &tensorDescs);

// 输出打印
void PrintOutTensorValue(atb::Tensor &outTensor);

// 创建图算子
atb::Status CreateGraphOperation(atb::Operation **operation);

#endif

