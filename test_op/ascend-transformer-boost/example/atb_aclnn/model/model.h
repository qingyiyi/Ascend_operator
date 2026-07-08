#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"
#include "utils/log.h"

enum class TensorType
{
    INTERNAL_TENSOR = 0,
    NOT_INTERNAL_TENSOR,
};

// 图节点，每个Node表示一个Operation或者GraphOperation
struct Node
{
    // Node对应的operation或者graphOperation。
    atb::Operation *operation_ = nullptr;

    // Node的输入tensors
    atb::SVector<atb::Tensor *> inTensors_{};

    // Node的输出tensors
    atb::SVector<atb::Tensor *> outTensors_{};

    // Node的输出是中间tensor类型
    atb::SVector<TensorType> outTensorTypes_{};

    atb::VariantPack variantPack_{};

    uint64_t workspaceSize_ = 0;
    int workspaceBlockId_ = -1;
    void *workspace_ = nullptr;
};

// 所有的Node组成一个完整的图。
class Model
{
public:
    // 描述该模型的输入
    enum InTensorId : int
    { // 定义各TensorID
        IN_TENSOR_A = 0,
        IN_TENSOR_B,
        IN_TENSOR_C,
        IN_TENSOR_D,
        Mode_INPUT_SIZE,
    };

    enum OutTensorId : int
    {
        GLUE_OUT = 0,
        Mode_OUTPUT_SIZE,
    };

    explicit Model(std::string &&modelName = "") : modelName_(std::move(modelName))
    {
        LOG_INFO("Create model: " + modelName_);
    }

    // 模型初始化，设置模型的
    void InitResource(uint32_t deviceId);

    // 创建模型图
    void CreateModelGraph();

    // 创建模型的输入tensors
    void CreateModelInput();

    // 创建模型的输入tensors
    void CreateModelOutput();

    // model执行
    void Execute();

    // stream流同步
    void WaitFinish();

    // 资源释放
    void FreeResource();

    // 模型的输入tensors
    atb::SVector<atb::Tensor> modelInTensors_;

    // 模型的输出tensors
    atb::SVector<atb::Tensor> modelOutTensors_;

private:
    // 创建图算子的operation
    void CreateGraphOpLayer(size_t nodeId);

    // 创建aclnn算子的operation
    void CreateAclnnOpLayer(size_t nodeId);

    // 构造对应nodeId的node的VariantPack
    void BuildNodeVariantPack(int nodeId);

    // 下发nodeId对应的Operation
    atb::Status ExecuteNode(int nodeId);

    // workspace创建函数
    void CreateWorkspaceBuffer(int nodeId, int workspaceSizeNeeded);

    // 模型图的shape推导函数
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs);

    std::string modelName_;
    uint32_t deviceId_ = 1;
    atb::Context *modeContext_ = nullptr;
    aclrtStream modelStream_ = nullptr;
    std::vector<Node> nodes_;

    // 模型的中间tensors，layer之间以internalTensors进行连接，这里要注意顺序
    std::vector<atb::Tensor> internalTensors_;
};

#endif

