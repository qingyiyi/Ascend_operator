#include <atb/types.h>
#include <acl/acl.h>
#include "utils/log.h"
#include "utils/utils.h"
#include "memorypool.h"

constexpr size_t POOL_SIZE = 104857600;  // 预分配内存大小 100 MiB，大小可按实际需求决定

MemoryPool::MemoryPool(size_t poolSize = POOL_SIZE)
{
    // 调用acl接口预分配内存
    CHECK_RET(aclrtMalloc(&baseMemPtr_, poolSize, ACL_MEM_MALLOC_HUGE_FIRST),
        "malloc huge size memrory " + std::to_string(poolSize) + " bytes fail");
    curMemPtr_ = baseMemPtr_;
    remainSize_ = poolSize;
}

MemoryPool::~MemoryPool()
{
    // 销毁时通过acl接口释放内存
    if (baseMemPtr_ != nullptr) {
        CHECK_RET(aclrtFree(baseMemPtr_), "free huge memory fail");
    }
    LOG_INFO("release MemoryPool success");
}

uint64_t MemoryPool::GenerateBlocksId()
{
    return static_cast<uint64_t>(id_.fetch_add(1, std::memory_order_relaxed));
}

void MemoryPool::AllocateBlock(uint32_t size, int &blockId)
{
    std::unique_lock<std::mutex> lock(blockMutex_);

    size_t alignSize = ((size + 31) & ~31) + 32;  // 分配的空间需要32字节对齐后再加32字节

    // 寻找是否有足够大小的空闲内存块
    for (auto it = freeBlocks_.begin(); it != freeBlocks_.end(); it++) {
        if (it->second.blockSize >= alignSize) {
            blockId = it->second.blockId;
            usedBlocks_.insert(*it);
            freeBlocks_.erase(it);
            LOG_INFO("find free block id " + std::to_string(blockId) + " to allocate");
            return;
        }
    }

    // 没有找到符合的内存块，需要从剩余的内存空间中创建新内存块
    if (remainSize_ > alignSize) {
        blockId = GenerateBlocksId();
        uint64_t curMemPtrAlign = (reinterpret_cast<uint64_t>(curMemPtr_) + 63) & ~63;  // 内存地址需要64字节对齐
        remainSize_ -= (curMemPtrAlign - reinterpret_cast<uint64_t>(curMemPtr_));
        curMemPtr_ = reinterpret_cast<void *>(curMemPtrAlign);

        MemoryBlock block = {blockId, alignSize, curMemPtr_};
        usedBlocks_.insert({blockId, block});
        remainSize_ -= alignSize;
        curMemPtr_ = reinterpret_cast<uint8_t *>(curMemPtr_) + alignSize;
        LOG_INFO("allocate block id " + std::to_string(blockId) + " for size " + std::to_string(alignSize));
        return;
    }

    // 剩余的内存空间不足，无法分配内存块
    LOG_ERROR("allocate block fail");
}

void MemoryPool::FreeBlock(int blockId)
{
    std::unique_lock<std::mutex> lock(blockMutex_);

    // 内存块索引合法性校验
    if (blockId < 0) {
        LOG_INFO("skip over the invalid block id " + std::to_string(blockId));
        return;
    }

    // 将需要释放的内存块标记为空闲
    auto it = usedBlocks_.find(blockId);
    if (it != usedBlocks_.end()) {
        freeBlocks_.insert(*it);
        usedBlocks_.erase(it);
    } else {
        LOG_ERROR("Double free block id " + std::to_string(blockId));
    }
}

void MemoryPool::GetBlockPtr(int blockId, void *&addr)
{
    std::unique_lock<std::mutex> lock(blockMutex_);

    if (blockId < 0) {
        LOG_INFO("Invalid block id " + std::to_string(blockId) + "to get ptr");
        return;
    }

    // 寻找内存块，返回物理内存地址
    auto it = usedBlocks_.find(blockId);
    if (it != usedBlocks_.end()) {
        addr = it->second.address;
    } else {
        LOG_ERROR("Get block address error, block id " + std::to_string(blockId));
    }
}

