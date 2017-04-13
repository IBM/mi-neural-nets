# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build linear layer tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(softmaxTestsRunner SoftmaxTests.cpp)
	target_link_libraries(softmaxTestsRunner ${GTEST_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(softmaxTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	add_test(softmaxTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmaxTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
