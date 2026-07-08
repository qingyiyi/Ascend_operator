cmake_minimum_required(VERSION 3.16)
project("test_model")
option(USE_CXX11_ABI "USE_CXX11_ABI" OFF)
set(CMAKE_BUILD_TYPE Debug)
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    )

set(TEST_MODEL_CXX
    main.cpp
    aclnn/aclnn_gelu_operation.cpp
    aclnn/aclnn_operation_base.cpp
    atb/atb_graph_op.cpp
    utils/utils.cpp
    utils/log.cpp
    model/model.cpp
    memory/memorypool.cpp
    memory/memory_utils.cpp
)

# 列出所有的头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/aclnn
    ${CMAKE_SOURCE_DIR}/atb
    ${CMAKE_SOURCE_DIR}/model
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/memory
)

add_executable(test_model ${TEST_MODEL_CXX})

target_link_libraries(test_model PRIVATE atb ascendcl opapi nnopbase pthread)

