/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * @file: HebbianRuleTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#include "HebbianRule.hpp"

#include <gtest/gtest.h>
#include <cmath>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#include <optimization/HebbianRule.hpp>


/*!
 * Tests hebbian update on 3x4matrix.
 * \author tkornuta
 */
TEST(HebbianRule, Weight3x4Update) {
	// Input and outputs.
	mic::types::MatrixPtr<double> x = MAKE_MATRIX_PTR(double, 3, 2);
	(*x) << 0, 1, 1, 0, 0, 0;
	mic::types::MatrixPtr<double> y = MAKE_MATRIX_PTR(double, 4, 2);
	(*y) << 0, 1, 1, 0, 0, 1, 1, 1;

	// Desired result - delta.
	mic::types::MatrixPtr<double> result_delta = MAKE_MATRIX_PTR(double, 4, 3);
	(*result_delta) << 0.001, 0, 0, 0, 0.001, 0, 0.001, 0, 0, 0.001, 0.001, 0;

	// Rule.
	mic::neural_nets::learning::HebbianRule<double> hebb(result_delta->rows(), result_delta->cols());
	// Calculate update.
	mic::types::MatrixPtr<double> delta = hebb.calculateUpdate(x, y, 0.001);

/*	std::cout << "x = \n" << (*x) << std::endl;
	std::cout << "y = \n" << (*y) << std::endl;
	std::cout << "delta = \n" << (*delta) << std::endl;
	std::cout << "result_delta = \n" << (*result_delta) << std::endl;*/

	// Check dimensions.
	ASSERT_EQ(delta->rows(), result_delta->rows());
	ASSERT_EQ(delta->cols(), result_delta->cols());

	// Before update.
	for (size_t i=0; i< (size_t)result_delta->size(); i++)
		ASSERT_LE(std::fabs((*delta)[i] - (*result_delta)[i]), 0) << " at element i=" << i;

	// After update.
	hebb.update(result_delta, x, y, 0.001);
	for (size_t i=0; i< (size_t)result_delta->size(); i++)
		ASSERT_LE(std::fabs((*result_delta)[i] - 2*(*delta)[i]), 0) << " at element i=" << i;
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


