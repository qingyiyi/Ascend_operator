#include <acl/acl.h>
#include "utils/log.h"
#include "utils/utils.h"
#include "memory_utils.h"

// 全局MemoryManager实例
static MemoryManager g_memoryManager;

MemoryManager::MemoryManager()
{}

void MemoryManager::CreateMemoryPool(size_t poolSize)
{
    uint32_t deviceCount = 0;

    // 获取全部Device的数量
    CHECK_RET(aclrtGetDeviceCount(&deviceCount), "get devicecount fail");
    for (size_t i = 0; i < deviceCount; i++) {

        // 指定操作的Device
        aclrtSetDevice(i);

        // 创建内存池，poolSize参数指定预分配空间大小
        std::shared_ptr<MemoryPool> memoryPool = std::make_shared<MemoryPool>(poolSize);
        memoryPools_.push_back(memoryPool);
        LOG_INFO("create mempool for device " + std::to_string(i) + " success");
    }
}

int32_t MemoryManager::GetDeviceId()
{
    int32_t deviceId = -1;
    CHECK_RET(aclrtGetDevice(&deviceId), "get device ID fail");
    return deviceId;
}

std::shared_ptr<MemoryPool> &MemoryManager::GetMemoryPool()
{
    // 获取当前操作的Device，返回对应的内存池
    size_t deviceId = static_cast<size_t>(GetDeviceId());
    CHECK_RET(deviceId >= memoryPools_.size(), "Invalid device id " + deviceId);
    return memoryPools_[deviceId];
}

void MemoryManager::AllocateBlock(uint32_t size, int &blockId)
{
    GetMemoryPool()->AllocateBlock(size, blockId);
}

void MemoryManager::FreeBlock(int blockId)
{
    GetMemoryPool()->FreeBlock(blockId);
}

void MemoryManager::GetBlockPtr(int blockId, void *&addr)
{
    GetMemoryPool()->GetBlockPtr(blockId, addr);
}

MemoryManager &GetMemoryManager()
{
    return g_memoryManager;
}

