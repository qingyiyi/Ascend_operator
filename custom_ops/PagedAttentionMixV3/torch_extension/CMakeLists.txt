cmake_minimum_required(VERSION 3.16.0)
project(PagedAttentionMixV3TorchExtension)

set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(_EXTENSION_NAME paged_attention_mix_v3_ext)

add_library(${_EXTENSION_NAME} SHARED
    csrc/host/paged_attention_mix_v3_ext.cpp
)

target_include_directories(${_EXTENSION_NAME} PRIVATE
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(${_EXTENSION_NAME} PRIVATE
    ${TORCH_NPU_PATH}/lib
    ${TORCH_PATH}/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(${_EXTENSION_NAME} PRIVATE
    torch_npu
    ascendcl
    dl
)