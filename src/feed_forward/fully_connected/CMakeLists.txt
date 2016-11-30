# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build linear layer tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(fullyConnectedLayerTestsRunner FullyConnectedLayerTests.cpp)
	target_link_libraries(fullyConnectedLayerTestsRunner ${GTEST_LIBRARIES})
	add_test(fullyConnectedLayerTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fullyConnectedLayerTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
