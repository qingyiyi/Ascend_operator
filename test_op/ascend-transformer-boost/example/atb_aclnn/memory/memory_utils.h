#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <memory>
#include <vector>
#include "memorypool.h"

// 内存管理类，管理每个Device上的内存池
class MemoryManager {
public:
    MemoryManager();

    // 在每个Device上创建对应的内存池
    void CreateMemoryPool(size_t poolSize);

    // 获取当前线程对应的Device
    int32_t GetDeviceId();

    // 获取当前线程对应设备上的内存池
    std::shared_ptr<MemoryPool> &GetMemoryPool();

    // 分配内存块
    void AllocateBlock(uint32_t size, int &blockId);

    // 释放内存块
    void FreeBlock(int blockId);

    // 获取内存块的物理地址
    void GetBlockPtr(int blockId, void *&addr);

private:
    std::vector<std::shared_ptr<MemoryPool>> memoryPools_;
};

// 获取全局MemoryManager实例
MemoryManager &GetMemoryManager();

#endif

