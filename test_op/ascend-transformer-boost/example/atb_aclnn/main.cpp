#include <thread>
#include "model/model.h"
#include "memory/memory_utils.h"
#include "utils/utils.h"

void ModelExecute(uint32_t deviceId, Model &model)
{
    // 初始化模型，创建需要的context，stream
    model.InitResource(deviceId);

    // 创建模型图
    model.CreateModelGraph();

    // 创建模型输入，并填入值
    model.CreateModelInput();

    // 创建模型的输出大小
    model.CreateModelOutput();

    // 模型执行
    model.Execute();

    // 打印输出Tensor的值
    PrintOutTensorValue(model.modelOutTensors_.at(0));

    // 资源释放
    model.FreeResource();
}

int main()
{
    // AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret, "aclInit failed. ret: " + std::to_string(ret));

    // 创建内存池
    size_t poolSize = 104857600; // Allocated memory 100 MB.
    GetMemoryManager().CreateMemoryPool(poolSize);

    // 创建模型图
    uint32_t deviceCount = 0;
    CHECK_RET(aclrtGetDeviceCount(&deviceCount), "get devicecount fail");
    std::vector<Model> modelArray(deviceCount);

    // 分多个线程进行模型图的下发
    std::vector<std::thread> threadArray(deviceCount);
    for (size_t i = 0; i < deviceCount; i++) {
        Model &model = modelArray.at(i);
        threadArray.at(i) = std::thread([i, &model]{ModelExecute(i, model);}); // 线程创建及函数绑定
    }
    for (size_t i = 0; i < deviceCount; i++) {
        threadArray.at(i).join(); // 等待子线程结束
    }

    aclFinalize();
    return 0;
}

