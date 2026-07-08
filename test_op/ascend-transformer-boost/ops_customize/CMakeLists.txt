# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.15)

option(IS_STANDALONE_BUILD "Build user-customize operators independently" OFF)
option(BUILD_CUSTOMIZE_OPS_TEST "Build user-customize operators tests files" OFF)

message(STATUS "IS_STANDALONE_BUILD:${IS_STANDALONE_BUILD}")
message(STATUS "BUILD_CUSTOMIZE_OPS_TEST:${BUILD_CUSTOMIZE_OPS_TEST}")

file(GLOB ALL_OPS_IMPL_DIRS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/ops/*/operation_implement"
    "${CMAKE_CURRENT_LIST_DIR}/ops/*/operation_implement/tiling"
)

file(GLOB_RECURSE CUSTOMIZE_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/*/operation_implement/*.cpp"
)


if (IS_STANDALONE_BUILD)
    # 单独编译
    project("atb_customize")
    set(CMAKE_CXX_STANDARD 17)
    if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ARCH aarch64)
    elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(ARCH x86_64)
    endif()
    option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
    message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

    set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
    set(CMAKE_SKIP_RPATH TRUE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        ADD_LINK_OPTIONS(-s)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
    endif()

    set(LD_FLAGS_GLOBAL "-rdynamic -ldl -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none")
    set(CMAKE_SHARED_LINKER_FLAGS "${LD_FLAGS_GLOBAL} -fexceptions")
    set(CMAKE_EXE_LINKER_FLAGS "${LD_FLAGS_GLOBAL} -pie -fPIE")
    if(USE_CXX11_ABI)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
        set(cxx_abi 1)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(cxx_abi 0)
    endif()

    set(OPS_PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
    set(OPS_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
    set(MKI_SOURCE_DIR "${OPS_THIRD_PARTY_DIR}/ascend-boost-comm")
    set(ATB_SOURCE_DIR "${OPS_THIRD_PARTY_DIR}/ascend-transformer-boost")

    include_directories(
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/ops
        ${ALL_OPS_IMPL_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}/include
        $ENV{ASCEND_HOME_PATH}/include
        ${OPS_THIRD_PARTY_DIR}/nlohmannJson/include
        ${ATB_SOURCE_DIR}/include
        ${ATB_SOURCE_DIR}/src
        ${ATB_SOURCE_DIR}/src/atb
        ${ATB_SOURCE_DIR}/src/kernels/include
        ${ATB_SOURCE_DIR}/src/kernels
        ${ATB_SOURCE_DIR}/src/kernels/kernels
        ${ATB_SOURCE_DIR}/src/kernels/mixkernels
        ${MKI_SOURCE_DIR}/src/include)

    link_directories(
        $ENV{ASCEND_HOME_PATH}/lib64
        $ENV{ATB_HOME_PATH}/lib)

    set(CUSTOMIZE_DEPS
        atb
        mki
        customize_ops
        ascendcl
        profapi
        pthread
    )

    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/ops_customize/cxx_abi_${cxx_abi}")
    message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")
else()
    # 作为ATB的一部分进行编译
    option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
    option(USE_MSDEBUG "USE_MSDEBUG" OFF)
    message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")
    message(STATUS "USE_MSDEBUG:${USE_MSDEBUG}")

    set(OPS_PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
    set(OPS_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
    set(MKI_PACKAGE_DIR "${OPS_THIRD_PARTY_DIR}/mki")
    set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")


    include_directories(
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/ops
        ${ALL_OPS_IMPL_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}/include
        $ENV{ASCEND_HOME_PATH}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/kernels/include
        ${CMAKE_SOURCE_DIR}/src/kernels/tiling
        ${CMAKE_SOURCE_DIR}/src/kernels/mixkernels
        ${CMAKE_SOURCE_DIR}/3rdparty/mki/include)

    set(CUSTOMIZE_DEPS
        mki
        customize_ops
        ascendcl
        profapi
        pthread
        acl_op_compiler
    )

endif()

add_subdirectory(ops)

add_library(atb_customize SHARED ${CUSTOMIZE_SRCS})
add_library(atb_customize_static STATIC ${CUSTOMIZE_SRCS})


target_link_libraries(atb_customize PUBLIC ${CUSTOMIZE_DEPS})
target_link_libraries(atb_customize_static PUBLIC ${CUSTOMIZE_DEPS})

install(TARGETS atb_customize atb_customize_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/customize)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/customize_ops_configs" DESTINATION ./configs)

if (BUILD_CUSTOMIZE_OPS_TEST)
    set(GTEST_PATH "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
    find_package(GTest 1.13.0 REQUIRED
                PATHS "${GTEST_PATH}"
                NO_DEFAULT_PATH)
    enable_testing()
    add_subdirectory(ops/customize_blockcopy/tests)
endif()