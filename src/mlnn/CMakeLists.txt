# Copyright (C) tkornuta, IBM Corporation 2015-2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
	convolution/Cropping.hpp
	convolution/Padding.hpp
	convolution/MaxPooling.hpp
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
        experimental/ConvHebbian.hpp
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
