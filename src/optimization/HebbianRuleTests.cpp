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
	mic::types::MatrixPtr<double> result_delta = MAKE_MATRIX_PTR(double, 3, 4);
	(*result_delta) << 0.001, 0, 0.001, 0.001, 0, 0.001, 0, 0.001, 0, 0, 0, 0;

	// Rule.
	mic::neural_nets::learning::HebbianRule<double> hebb(result_delta->rows(), result_delta->cols());
	// Calculate update.
	mic::types::MatrixPtr<double> delta = hebb.calculateUpdate(x, y, 0.001);

/*	std::cout << "x = \n" << (*x) << std::endl;
	std::cout << "y = \n" << (*y).transpose() << std::endl;
	std::cout << "delta = \n" << (*delta) << std::endl;
	std::cout << "delta = \n" << (*delta) << std::endl;*/

	for (size_t i=0; i< (size_t)result_delta->size(); i++)
		ASSERT_LE(std::fabs((*delta)[i] - (*result_delta)[i]), 0) << " at element i=" << i;

//	std::cout << "              -> Converged after " << iteration << " iterations\n";
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


