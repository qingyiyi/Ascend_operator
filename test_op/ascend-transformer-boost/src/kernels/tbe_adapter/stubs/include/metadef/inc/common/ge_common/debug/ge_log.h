/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STUB_GE_LOG_H
#define ASCEND_OPS_STUB_GE_LOG_H

#include <mki/utils/log/log.h>
#if defined(HAVE_DLOG)
    #include <dlog_pub.h>
#else
    #include <toolchain/slog.h>
#endif
#include "ge_error_codes.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge_common/ge_api_error_codes.h"
#include "external/base/err_msg.h"
#include "log.h"

#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

using string = std::string;

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#define INTERNAL_ERROR 4
#ifdef __cplusplus
extern "C" {
#endif

#define GE_MODULE_NAME static_cast<int32_t>(45) // GE: 45, defined in slog
#define GE_MODULE_NAME_U16 static_cast<uint16_t>(45)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

class GE_FUNC_VISIBILITY GeLog{public: static uint64_t GetTid(){
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
return tid;
}};
#ifdef __cplusplus
}
#endif

#define GELOGE(ERROR_CODE, fmt, ...)                                                                                   \
    MKI_FLOG_ERROR("[%s] error code: %d, " #fmt, __FUNCTION__, ERROR_CODE, ##__VA_ARGS__)

#define GELOGW(fmt, ...) MKI_FLOG_WARN("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)
#define GELOGI(fmt, ...) MKI_FLOG_INFO("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)
#define GELOGD(fmt, ...) MKI_FLOG_DEBUG("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)

#define GE_CHECK_NOTNULL_JUST_RETURN(...)
#define GE_CHECK_NOTNULL(...)

#define REPORT_CALL_ERROR REPORT_INNER_ERROR

#define GE_IF_BOOL_EXEC(...)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...)       \
    do {                                                 \
        const bool b = (expr);                           \
        if (!b) {                                        \
            REPORT_INNER_ERROR("E19999", __VA_ARGS__);   \
            GELOGE((_status), __VA_ARGS__);              \
            return (_status);                            \
        }                                                \
    } while (false)

// Get error code description
#define GE_GET_ERRORNO_STR(value) ge::StatusFactory::Instance()->GetErrDesc(value)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)                                                                   \
  do {                                                                                                                 \
    dlog_error((MOD_NAME), "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s)" fmt, GeLog::GetTid(), &__FUNCTION__[0U],     \
               (ERROR_CODE), ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()), ##__VA_ARGS__);                               \
  } while (false)

#define GE_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, fmt, ##__VA_ARGS__)

using Status = uint32_t;

#ifndef GE_ERRORNO_DEFINE
#define GE_ERRORNO_DEFINE(runtime, type, level, sysid, modid, name, value)                                \
  constexpr ge::Status name = ((static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(runtime))) << 30U) | \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(type))) << 28U) |     \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(level))) << 25U) |    \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(sysid))) << 17U) |    \
                              (static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(modid))) << 12U) |    \
                              (static_cast<uint32_t>(0x0FFFU) & (static_cast<uint32_t>(value))))
#endif

#ifndef GE_ERRORNO_EXTERNAL
#define GE_ERRORNO_EXTERNAL(name, desc) const ErrorNoRegisterar g_errorno_##name((name), (desc))
#endif

#ifndef GE_ERRORNO
// Code compose(4 byte), runtime: 2 bit,  type: 2 bit,   level: 3 bit,  sysid: 8 bit, modid: 5 bit, value: 12 bit
#define GE_ERRORNO(runtime, type, level, sysid, modid, name, value, desc)     \
  GE_ERRORNO_DEFINE(runtime, type, level, sysid, modid, name, value);         \
  GE_ERRORNO_EXTERNAL(name, desc)
#endif

#define GE_ERRORNO_COMMON(name, value, desc)                                 \
  GE_ERRORNO(ge::InnLogRuntime::RT_HOST, ge::InnErrorCodeType::ERROR_CODE,   \
             ge::InnErrorLevel::COMMON_LEVEL, ge::InnSystemIdType::SYSID_GE, \
             ge::InnSubModuleId::COMMON_MODULE, name, (value), (desc))

namespace ge {
using Status = uint32_t;
#ifndef GE_PARAM_INVALID_DEFINED
#define GE_PARAM_INVALID_DEFINED
constexpr uint32_t PARAM_INVALID = 2;
#endif

// System ID
enum class InnSystemIdType : std::uint8_t { SYSID_GE = 8 };
// Runtime location
enum class InnLogRuntime : std::uint8_t {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

// Sub model
enum class InnSubModuleId {
  COMMON_MODULE = 0,
  CLIENT_MODULE = 1,
  INIT_MODULE = 2,
  SESSION_MODULE = 3,
  GRAPH_MODULE = 4,
  ENGINE_MODULE = 5,
  OPS_MODULE = 6,
  PLUGIN_MODULE = 7,
  RUNTIME_MODULE = 8,
  EXECUTOR_MODULE = 9,
  GENERATOR_MODULE = 10,
  LLM_ENGINE_MODULE = 11,
};

// Error code type
enum class InnErrorCodeType {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

// Error level
enum class InnErrorLevel {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

GE_ERRORNO_COMMON(GE_PLGMGR_FUNC_NOT_EXIST, 32, "Failed to find any function!");     // 1343225888
GE_ERRORNO_COMMON(GE_PLGMGR_INVOKE_FAILED, 33, "Failed to invoke any function!");    // 1343225889
} // namespace ge
#endif // ASCEND_OPS_STUB_GE_LOG_H
