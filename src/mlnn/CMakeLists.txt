# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Get list of subdirectories.
FILE(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
FOREACH(child ${children})
    IF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        LIST(APPEND mlnn_INCLUDE_DIRS ${child})
    ENDIF()
ENDFOREACH()

# Iterate through subdirectories...,
foreach (dir ${mlnn_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    # Get list of headers.
    FILE(GLOB mlnn_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.hpp)
    # ... and install headers.
    install(FILES ${mlnn_HEADERS} DESTINATION include/mlnn/${dir})
endforeach()
# And finally install selected headers from the "main" directory.
install(FILES
	MultiLayerNeuralNetwork.hpp
	Layer.hpp
  DESTINATION include/mlnn)


# Create MLNN shared library using all cpp files.
FILE(GLOB mlnn_SRC 
	MultiLayerNeuralNetwork.cpp
	Layer.cpp
	activation_function/*.cpp
	convolution/*.cpp
	cost_function/*.cpp
	fully_connected/*.cpp
	regularisation/*.cpp)
	
add_library(mlnn SHARED
	${mlnn_SRC}
	)

# Link MLNN with required libraries. 
target_link_libraries(mlnn
	types
	)
if(OpenBLAS_FOUND)
	target_link_libraries(mlnn  ${OpenBLAS_LIB} )
endif(OpenBLAS_FOUND)

# Add to variable storing all libraries/targets.
set(MIAlgorithms_LIBRARIES ${MIAlgorithms_LIBRARIES} "mlnn" CACHE INTERNAL "" FORCE)

# Install target library.
install(TARGETS mlnn LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

# =======================================================================
# Build nn tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(mlnnTestsRunner MultiLayerNeuralNetworkTests.cpp)
	target_link_libraries(mlnnTestsRunner
		logger
		mlnn
		${Boost_LIBRARIES}
		${GTEST_LIBRARIES})
	add_test(mlnnTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mlnnTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)

