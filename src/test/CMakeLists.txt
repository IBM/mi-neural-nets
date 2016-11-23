# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build and install - matrix test application.
# =======================================================================

set(BUILD_TEST_MATRIX ON CACHE BOOL "Build the application for testing matrix multiplication")

if(${BUILD_TEST_MATRIX})
	# Create exeutable.
	ADD_EXECUTABLE(matrixSpeedTest matrixSpeedTest.cpp)
	# Link it with shared libraries.
	if(OpenBLAS_FOUND)
		target_link_libraries(matrixSpeedTest  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	target_link_libraries(matrixSpeedTest 
		${Boost_LIBRARIES}
		${ARMADILLO_LIBRARIES} 
		)

	# install test to bin directory
	install(TARGETS matrixSpeedTest RUNTIME DESTINATION bin)

endif(${BUILD_TEST_MATRIX})
