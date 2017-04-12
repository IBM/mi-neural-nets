# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Install Layer headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)
install(FILES
	layer/Layer.hpp
	layer/LayerTypes.hpp
	DESTINATION include/mlnn/layer)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/activation_function)
install(FILES
	activation_function/ELU.hpp
	activation_function/ReLU.hpp
	activation_function/Sigmoid.hpp
	DESTINATION include/mlnn/activation_function)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/convolution)
#install(FILES
#	convolution/Convolution.hpp
#	convolution/Padding.hpp
#	convolution/Pooling.hpp
#	DESTINATION include/mlnn/convolution)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cost_function)
install(FILES
	cost_function/Regression.hpp
	cost_function/Softmax.hpp
	DESTINATION include/mlnn/cost_function)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fully_connected)
install(FILES
	fully_connected/Identity.hpp
	fully_connected/Linear.hpp
	fully_connected/SparseLinear.hpp
	DESTINATION include/mlnn/fully_connected)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/regularisation)
install(FILES
	regularisation/Dropout.hpp
	DESTINATION include/mlnn/regularisation)

# Install MLNN headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
install(FILES MultiLayerNeuralNetwork.hpp DESTINATION include/mlnn)


# =======================================================================
# Build nn tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(mlnnTestsRunner MultiLayerNeuralNetworkTests.cpp)
	target_link_libraries(mlnnTestsRunner
		logger
		types
		mlnn
		${Boost_LIBRARIES}
		${GTEST_LIBRARIES})
	add_test(mlnnTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mlnnTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)

# =======================================================================
# Add subdirectories
# =======================================================================

add_subdirectory(fully_connected)
