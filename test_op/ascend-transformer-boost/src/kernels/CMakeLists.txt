#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
option(USE_MSDEBUG "USE_MSDEBUG" OFF)
option(USE_ASCENDC_DUMP "USE_ASCENDC_DUMP" OFF)
message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")
message(STATUS "USE_MSDEBUG:${USE_MSDEBUG}")
message(STATUS "USE_ASCENDC_DUMP:${USE_ASCENDC_DUMP}")

set(OPS_PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(OPS_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(MKI_PACKAGE_DIR "${OPS_THIRD_PARTY_DIR}/mki")
set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")

set(DLOG_HEADER_PATH "$ENV{ASCEND_HOME_PATH}/pkg_inc/base/dlog_pub.h")
if (EXISTS "${DLOG_HEADER_PATH}")
    message(STATUS "Found dlog_pub.h, use HAVE_DLOG")
    add_compile_definitions(HAVE_DLOG)
endif()

if(USE_ASCENDC_DUMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_ASCENDC_DUMP")
endif()

include_directories(
    ${OPS_PROJECT_ROOT_DIR}
    ${OPS_PROJECT_ROOT_DIR}/tbe_adapter/include
)

add_subdirectory(mixkernels)
add_subdirectory(kernels)
add_subdirectory(lcal)
if (BUILD_TBE_ADAPTER)
    add_subdirectory(tbe_adapter)
endif()