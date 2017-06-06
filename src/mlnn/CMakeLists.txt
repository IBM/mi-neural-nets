# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Install Layer headers.
install(FILES
	layer/Layer.hpp
	layer/LayerTypes.hpp
	DESTINATION include/mlnn/layer)

install(FILES
	activation_function/ELU.hpp
	activation_function/ReLU.hpp
	activation_function/Sigmoid.hpp
	DESTINATION include/mlnn/activation_function)

install(FILES
	convolution/Convolution.hpp
#	convolution/Padding.hpp
#	convolution/Pooling.hpp
	DESTINATION include/mlnn/convolution)

install(FILES
	cost_function/Softmax.hpp
	DESTINATION include/mlnn/cost_function)

install(FILES
	fully_connected/Linear.hpp
	fully_connected/SparseLinear.hpp
	fully_connected/HebbianLinear.hpp
	fully_connected/BinaryCorrelator.hpp
	DESTINATION include/mlnn/fully_connected)
	
install(FILES
        experimental/TiledHebbian.hpp
	DESTINATION include/mlnn/experimental)

install(FILES
	regularisation/Dropout.hpp
	DESTINATION include/mlnn/regularisation)

# Install MLNN headers.
install(FILES
	MultiLayerNeuralNetwork.hpp
	BackpropagationNeuralNetwork.hpp
	HebbianNeuralNetwork.hpp
	DESTINATION include/mlnn)


# =======================================================================
# Build nn tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(mlnnTestsRunner MultiLayerNeuralNetworkTests.cpp)
	target_link_libraries(mlnnTestsRunner
		logger
		types
		${Boost_LIBRARIES}
		${GTEST_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(mlnnTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	add_test(mlnnTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mlnnTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)

# =======================================================================
# Add subdirectories
# =======================================================================

add_subdirectory(cost_function)

add_subdirectory(convolution)

add_subdirectory(fully_connected)
