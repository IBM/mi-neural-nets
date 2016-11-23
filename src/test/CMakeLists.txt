# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build and install - matrix test application.
# =======================================================================

set(BUILD_TEST_MATRIX ON CACHE BOOL "Build the application for testing matrix multiplication")

if(${BUILD_TEST_MATRIX})
	# Create exeutable.
	ADD_EXECUTABLE(matrix_test matrix_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(matrix_test 
#		types
		${Boost_LIBRARIES}
		${ARMADILLO_LIBRARIES} 
		)
#	if(OpenBLAS_FOUND)
#		target_link_libraries(matrix_test  ${OpenBLAS_LIB} )
#		endif(OpenBLAS_FOUND)

	# install test to bin directory
	install(TARGETS matrix_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_MATRIX})


# =======================================================================
# Build and install - matrix array test.
# =======================================================================

set(BUILD_TEST_MATRIX_ARRAY ON CACHE BOOL "Build the application for testing creation/serialization of matrix array class")

if(${BUILD_TEST_MATRIX_ARRAY})
	# Create exeutable.
	ADD_EXECUTABLE(matrix_array_test matrix_array_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(matrix_array_test
		configuration
		types
		data_io
		${Boost_LIBRARIES}
		${ARMADILLO_LIBRARIES} 
		)

	# install test to bin directory
	install(TARGETS matrix_array_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_MATRIX_ARRAY})