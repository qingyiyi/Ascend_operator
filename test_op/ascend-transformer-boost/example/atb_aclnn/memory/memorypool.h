#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "memory_env.h"

// Device内存池
class MemoryPool {
public:
    explicit MemoryPool(size_t poolSize);
    ~MemoryPool();

    // 分配内存块
    void AllocateBlock(uint32_t size, int &blockId);

    // 释放内存块
    void FreeBlock(int blockId);

    // 获取内存块的物理地址
    void GetBlockPtr(int blockId, void *&addr);

private:
    // 生成内存块索引
    uint64_t GenerateBlocksId();

    std::atomic<uint64_t> id_ = 0;
    std::mutex blockMutex_;
    void *baseMemPtr_ = nullptr;
    void *curMemPtr_ = nullptr;
    int64_t remainSize_ = 0;
    std::unordered_map<int, MemoryBlock> freeBlocks_;
    std::unordered_map<int, MemoryBlock> usedBlocks_;
};

#endif
