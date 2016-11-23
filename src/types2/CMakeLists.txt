# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build matrix tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(matrixTestsRunner MatrixTests.cpp)
	target_link_libraries(matrixTestsRunner
		${GTEST_LIBRARIES}
		${Boost_LIBRARIES}
		${ARMADILLO_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(matrixTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
		
	add_test(matrixTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/matrixTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)


# =======================================================================
# Build matrix array tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(matrixArrayTestsRunner MatrixArrayTests.cpp)
	target_link_libraries(matrixArrayTestsRunner
		${GTEST_LIBRARIES}
		${Boost_LIBRARIES}
		${ARMADILLO_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(matrixArrayTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
		
	add_test(matrixArrayTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/matrixArrayTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
