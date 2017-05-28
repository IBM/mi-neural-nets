# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build convolution layer tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(convolutionTestsRunner
		Convolution_tests.cpp
		Convolution_convergence_tests.cpp
		)
	target_link_libraries(convolutionTestsRunner ${GTEST_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(convolutionTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	add_test(convolutionTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/convolutionTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
