#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

cmake_minimum_required(VERSION 3.15)
project("torch_atb")
set(CMAKE_CXX_STANDARD 17)

IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_LINK_OPTIONS(-s)
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ENDIF()

add_subdirectory(
    ${PROJECT_SOURCE_DIR}/../../3rdparty/pybind11
    ${CMAKE_BINARY_DIR}/pybind11
)

file(GLOB_RECURSE pybind11_source_files "*.cpp")
pybind11_add_module(_C ${pybind11_source_files})
set_target_properties(_C PROPERTIES OUTPUT_NAME "_C" SUFFIX ".so")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_C PRIVATE -Wno-odr -Wno-error=odr)
    target_link_options(_C PRIVATE -Wno-odr -Wno-error=odr)
endif()
target_link_options(_C PRIVATE -rdynamic -ldl -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none -fexceptions)
target_include_directories(_C PRIVATE
    $ENV{ASCEND_HOME_PATH}/include
    ${PROJECT_SOURCE_DIR}/../../include
    ${PROJECT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/../../3rdparty/mki/include
    ${PROJECT_SOURCE_DIR}/../../3rdparty/nlohmannJson/include
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
)
target_link_directories(_C PRIVATE
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
)
target_link_libraries(_C PRIVATE torch torch_npu)
install(TARGETS _C DESTINATION ${CMAKE_SOURCE_DIR}/../../output/torch_atb)
