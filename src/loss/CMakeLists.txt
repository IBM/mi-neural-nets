# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

install(FILES
	Loss.hpp
	CrossEntropyLoss.hpp
	SquaredErrorLoss.hpp
	LossTypes.hpp
	DESTINATION include/loss)

# =======================================================================
# Build loss function unit tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(lossTestsRunner LossTests.cpp)
	target_link_libraries(lossTestsRunner ${GTEST_LIBRARIES})
	add_test(lossTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lossTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
