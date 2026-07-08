/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils.h"
#include <complex>
#include <limits>
#include <securec.h>
#include <sstream>
#include <string>
#include <mki/utils/env/env.h>
#include <mki/utils/log/log_sink_stdout.h>
#include <mki/utils/log/log_sink_file.h>
#include "atb/utils/log.h"

namespace atb {
std::string Utils::GetAtbVersion()
{
    // ATBVERSION is a placeholder and will be replaced by real version number when CI build runs
    return "ATBVERSION";
}

uint64_t Utils::GetTensorSize(const Tensor &tensor)
{
    return GetTensorSize(tensor.desc);
}

uint64_t Utils::GetTensorSize(const TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }

    uint64_t dataItemSize = 0;
    switch (tensorDesc.dtype) {
        case ACL_DT_UNDEFINED:
            dataItemSize = sizeof(bool);
            break;
        case ACL_BOOL:
            dataItemSize = sizeof(bool);
            break;
        case ACL_FLOAT:
            dataItemSize = sizeof(float);
            break;
        case ACL_FLOAT16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_INT8:
            dataItemSize = sizeof(int8_t);
            break;
        case ACL_INT16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_INT32:
            dataItemSize = sizeof(int32_t);
            break;
        case ACL_INT64:
            dataItemSize = sizeof(int64_t);
            break;
        case ACL_UINT8:
            dataItemSize = sizeof(uint8_t);
            break;
        case ACL_UINT16:
            dataItemSize = sizeof(uint16_t);
            break;
        case ACL_UINT32:
            dataItemSize = sizeof(uint32_t);
            break;
        case ACL_UINT64:
            dataItemSize = sizeof(uint64_t);
            break;
        case ACL_BF16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_DOUBLE:
            dataItemSize = sizeof(double);
            break;
        case ACL_STRING:
            dataItemSize = sizeof(std::string);
            break;
        case ACL_COMPLEX64:
            dataItemSize = sizeof(std::complex<float>);
            break;
        case ACL_COMPLEX128:
            dataItemSize = sizeof(std::complex<double>);
            break;
        case ACL_HIFLOAT8:
            dataItemSize = sizeof(int8_t);
            break;
        case ACL_FLOAT8_E5M2:
            dataItemSize = sizeof(int8_t);
            break;
        case ACL_FLOAT8_E4M3FN:
            dataItemSize = sizeof(int8_t);
            break;
        case ACL_FLOAT8_E8M0:
            dataItemSize = sizeof(int8_t);
            break;
        default:
            ATB_LOG(ERROR)
                << "Tensor not support dtype:" << tensorDesc.dtype
                << ". Only support below dtype now: ACL_DT_UNDEFINED, ACL_BOOL, ACL_FLOAT, ACL_FLOAT16, ACL_INT8, "
                << "ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_BF16, "
                   "ACL_DOUBLE,"
                << " ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128.";
            return 0;
    }

    uint64_t elementCount = GetTensorNumel(tensorDesc);
    if (elementCount == 0) {
        ATB_LOG(ERROR) << "GetTensorSize result is zero!";
        return 0;
    }
    if (std::numeric_limits<uint64_t>::max() / elementCount < dataItemSize) {
        ATB_LOG(ERROR) << "GetTensorSize Overflow!";
        return 0;
    }
    return dataItemSize * elementCount;
}

uint64_t Utils::GetTensorNumel(const Tensor &tensor)
{
    return GetTensorNumel(tensor.desc);
}

uint64_t Utils::GetTensorNumel(const TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }
    uint64_t elementCount = 1;
    uint64_t maxVal = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < tensorDesc.shape.dimNum; i++) {
        if (tensorDesc.shape.dims[i] <= 0) {
            ATB_LOG(ERROR) << "dims[" << i << "] is <= 0!";
            return 0;
        }
        if (maxVal / tensorDesc.shape.dims[i] < elementCount) {
            ATB_LOG(ERROR) << "GetTensorNumel Overflow!";
            return 0;
        }
        elementCount *= tensorDesc.shape.dims[i];
    }

    return elementCount;
}

void Utils::QuantParamConvert(const float *src, uint64_t *dest, uint64_t itemCount)
{
    if (src == nullptr || dest == nullptr) {
        return;
    }
    for (uint64_t i = 0; i < itemCount; i++) {
        uint32_t temp;
        int ret = memcpy_s(&temp, sizeof(temp), &src[i], sizeof(temp));
        ATB_LOG_IF(ret != EOK, ERROR) << "memcpy_s Error! Error Code: " << ret;
        dest[i] = static_cast<uint64_t>(temp);
    }
}

Status AtbToMkiLogLevel(const atb::LogLevel logLevel, Mki::LogLevel &mkiLevel)
{
    switch (logLevel) {
        case atb::LogLevel::DEBUG:
            mkiLevel = Mki::LogLevel::TRACE;
            return NO_ERROR;
        case atb::LogLevel::INFO:
            mkiLevel = Mki::LogLevel::INFO;
            return NO_ERROR;
        case atb::LogLevel::WARN:
            mkiLevel = Mki::LogLevel::WARN;
            return NO_ERROR;
        case atb::LogLevel::ERROR:
            mkiLevel = Mki::LogLevel::ERROR;
            return NO_ERROR;
        case atb::LogLevel::NONE:
            mkiLevel = Mki::LogLevel::ERROR;
            return NO_ERROR;
        default:
            return ERROR_INVALID_PARAM;
    }
}

Status ParseEnvLogLevel(const char *val, atb::LogLevel &atbLevel)
{
    if (val == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    if (strcmp(val, "0") == 0) {
        atbLevel = atb::LogLevel::DEBUG;
        return NO_ERROR;
    }
    if (strcmp(val, "1") == 0) {
        atbLevel = atb::LogLevel::INFO;
        return NO_ERROR;
    }
    if (strcmp(val, "2") == 0) {
        atbLevel = atb::LogLevel::WARN;
        return NO_ERROR;
    }
    if (strcmp(val, "3") == 0) {
        atbLevel = atb::LogLevel::ERROR;
        return NO_ERROR;
    }
    if (strcmp(val, "4") == 0) {
        atbLevel = atb::LogLevel::NONE;
        return NO_ERROR;
    }
    return ERROR_INVALID_PARAM;
}

bool IsLogToStdoutEnabled()
{
    const char *envStdout = Mki::GetEnv("ASCEND_SLOG_PRINT_TO_STDOUT");
    return envStdout != nullptr && strcmp(envStdout, "1") == 0;
}

std::vector<std::string> SplitString(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

Status GetAtbModuleLogLevelStr(const char *moduleLogLevel, std::string &atbLogLevel)
{
    std::string envStr(moduleLogLevel);
    auto items = SplitString(envStr, ':');
    for (const auto &item : items) {
        auto kv = SplitString(item, '=');
        if (kv.size() != 2) {
            continue;
        }
        const std::string &moduleName = kv[0];
        const std::string &levelStr = kv[1];
        if (moduleName == "OP") {
            atbLogLevel = levelStr;
            return NO_ERROR;
        }
    }
    return ERROR_INVALID_PARAM;
}

// Check if log disabled
bool IsLogAtDisableLevel(const char *val)
{
    if (val == nullptr) {
        return false;
    }
    return strcmp(val, "4") == 0;
}

bool IsLogFileDisabled()
{
    const char *moduleLogLevel = Mki::GetEnv("ASCEND_MODULE_LOG_LEVEL");
    std::string atbModuleLogLevel;
    if (moduleLogLevel != nullptr) {
        Status st = GetAtbModuleLogLevelStr(moduleLogLevel, atbModuleLogLevel);
        if (st == NO_ERROR && atbModuleLogLevel == "4") {
            return true;
        }
    }

    const char *globalLogLevel = Mki::GetEnv("ASCEND_GLOBAL_LOG_LEVEL");
    return IsLogAtDisableLevel(globalLogLevel);
}

Status Utils::SetLogLevel(const atb::LogLevel atbLogLevel)
{
    if (atbLogLevel == atb::LogLevel::NONE) {
        Mki::LogCore::Instance().RemoveSink<Mki::LogSinkStdout>();
        Mki::LogCore::Instance().RemoveSink<Mki::LogSinkFile>();
        // Env cannot set the FATAL level. Use FATAL as an internal sentinel to mean "logging is disabled".
        Mki::LogCore::Instance().SetLogLevel(Mki::LogLevel::FATAL);
        return NO_ERROR;
    }

    Mki::LogLevel mkiLogLevel;
    Status status = AtbToMkiLogLevel(atbLogLevel, mkiLogLevel);
    if (status != NO_ERROR) {
        return status;
    }

    auto &logCore = Mki::LogCore::Instance();

    // If logging was previously disabled either by env configuration (IsLogFileDisabled() == True) or by ATB API
    // (sentinel level FATAL), there should be no sink registered and thus need to be added.
    if ((IsLogFileDisabled() || logCore.GetLogLevel() == Mki::LogLevel::FATAL) && logCore.GetAllSinks().empty()) {
        logCore.AddSink(std::make_shared<Mki::LogSinkFile>());
        if (IsLogToStdoutEnabled()) {
            logCore.AddSink(std::make_shared<Mki::LogSinkStdout>());
        }
    }

    logCore.SetLogLevel(mkiLogLevel);
    return NO_ERROR;
}

Status Utils::ResetLogLevel()
{
    const char *envModuleLogLevel = Mki::GetEnv("ASCEND_MODULE_LOG_LEVEL");
    const char *envGlobalLogLevel = Mki::GetEnv("ASCEND_GLOBAL_LOG_LEVEL");

    atb::LogLevel atbLevel = atb::LogLevel::ERROR;

    Status st = NO_ERROR;
    std::string atbModuleLogLevel;
    if (envModuleLogLevel != nullptr && GetAtbModuleLogLevelStr(envModuleLogLevel, atbModuleLogLevel) == NO_ERROR) {
        st = ParseEnvLogLevel(atbModuleLogLevel.c_str(), atbLevel);
        if (st != NO_ERROR) {
            return st;
        }
    } else if (envGlobalLogLevel != nullptr) {
        st = ParseEnvLogLevel(envGlobalLogLevel, atbLevel);
        if (st != NO_ERROR) {
            return st;
        }
    } else {
        atbLevel = atb::LogLevel::ERROR;
    }

    return SetLogLevel(atbLevel);
}
} // namespace atb