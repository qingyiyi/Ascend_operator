# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.15)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include)

link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64)
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
add_executable(multiStreamDemo multiStream_singleGraph_demo.cpp)
target_link_libraries(multiStreamDemo atb ascendcl runtime)