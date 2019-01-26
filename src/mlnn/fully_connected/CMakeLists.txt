# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build linear layer tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(linearTestsRunner Linear_tests.cpp)
	target_link_libraries(linearTestsRunner logger ${Boost_LIBRARIES} ${GTEST_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(linearTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	add_test(linearTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/linearTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
