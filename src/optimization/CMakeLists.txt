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

install(FILES
	OptimizationFunction.hpp
	OptimizationArray.hpp
	AdaDelta.hpp
	AdaGrad.hpp
	Adam.hpp
	GradientDescent.hpp
	Momentum.hpp
	RMSProp.hpp
	OptimizationFunctionTypes.hpp
	HebbianRule.hpp
	NormalizedHebbianRule.hpp
	NormalizedZerosumHebbianRule.hpp
	BinaryCorrelatorLearningRule.hpp
	DESTINATION include/optimization)


# =======================================================================
# Build artificial landscapes and optimization functions tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(artificialLandscapesTestsRunner ArtificialLandscapesTests.cpp)
	target_link_libraries(artificialLandscapesTestsRunner ${GTEST_LIBRARIES})
	add_test(artificialLandscapesTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/artificialLandscapesTestsRunner)

	add_executable(optimizationFunctionsTestsRunner
		OptimizationFunctionsTests.cpp
		GradientDescentTests.cpp
		MomentumTests.cpp
		AdaGradTests.cpp
		RMSPropTests.cpp
		AdaDeltaTests.cpp
		AdamTests.cpp
		GradPIDTests.cpp
		AdamIDTests.cpp
		)
	target_link_libraries(optimizationFunctionsTestsRunner ${GTEST_LIBRARIES})
	add_test(optimizationFunctionsTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optimizationFunctionsTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)


# =======================================================================
# Build hebbian learning test
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(hebbianLearningTestsRunner
		HebbianRuleTests.cpp
		)
	target_link_libraries(hebbianLearningTestsRunner ${GTEST_LIBRARIES})
	add_test(optimizationFunctionsTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hebbianLearningTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
