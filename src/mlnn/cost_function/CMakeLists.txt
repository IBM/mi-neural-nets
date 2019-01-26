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

# =======================================================================
# Build linear layer tests
# =======================================================================

# Link tests with GTest
if(GTEST_FOUND AND BUILD_UNIT_TESTS)

	add_executable(softmaxTestsRunner SoftmaxTests.cpp)
	target_link_libraries(softmaxTestsRunner logger ${Boost_LIBRARIES} ${GTEST_LIBRARIES})
	if(OpenBLAS_FOUND)
		target_link_libraries(softmaxTestsRunner  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)
	add_test(softmaxTestsRunner ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmaxTestsRunner)

endif(GTEST_FOUND AND BUILD_UNIT_TESTS)
