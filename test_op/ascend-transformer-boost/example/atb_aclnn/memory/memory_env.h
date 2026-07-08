#ifndef MEMORY_ENV_H
#define MEMORY_ENV_H

#include <iostream>

struct MemoryBlock {
    int64_t blockId;          // 内存块索引
    size_t blockSize;         // 内存块大小
    void *address = nullptr;  // 物理内存地址
};

#endif

