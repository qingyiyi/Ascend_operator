set(CMAKE_SKIP_RPATH FALSE)

if (IS_STANDALONE_BUILD)
    link_directories("${ATB_HOME_PATH}/lib")
    link_directories("${CMAKE_INSTALL_PREFIX}/lib")
else()
    link_directories("${CMAKE_BINARY_DIR}/ops_customize/lib")
endif()

add_executable(customize_blockcopy_test customize_blockcopy_test.cpp)
target_link_libraries(customize_blockcopy_test
  PRIVATE
    atb_customize
    atb
    pthread
    GTest::gtest_main
)

if (IS_STANDALONE_BUILD)
    set_target_properties(customize_blockcopy_test PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/output/bin")
endif()

include(GoogleTest)
add_test(NAME customize_blockcopy_test
         COMMAND customize_blockcopy_test)