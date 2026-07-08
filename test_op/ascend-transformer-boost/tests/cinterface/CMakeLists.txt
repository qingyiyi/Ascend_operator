#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#


file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(atb_cinterface ${SOURCE_FILES})
target_include_directories(atb_cinterface PRIVATE ${PROJECT_SOURCE_DIR}/)
target_include_directories(atb_cinterface PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(atb_cinterface PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(atb_cinterface PRIVATE ${ASCEND_HOME_PATH}/include/)
target_link_directories(atb_cinterface PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib ${ASCEND_HOME_PATH}/lib64/ ${ATB_HOME_PATH}/lib)
target_compile_options(atb_cinterface PRIVATE "-Wno-write-strings")
target_compile_options(atb_cinterface PRIVATE -Wno-sign-compare -Wno-narrowing)
target_link_libraries(atb_cinterface PRIVATE atb nnopbase tbe_adapter -lgtest -lgtest_main -lc_sec)
install(TARGETS atb_cinterface DESTINATION bin)