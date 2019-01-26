/*!
 * @file: AdaDeltaTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include <gtest/gtest.h>
#include <cmath>

#include <optimization/ArtificialLandscapesTests.hpp>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#include <optimization/AdaDelta.hpp>



/////////////////////////////////////////////////////////////////////////
// AdaDelta
/////////////////////////////////////////////////////////////////////////

/*!
 * Tests AdaDelta on a 1d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, AdaDelta_Convergence) {
	// Optimization function - 1d AdaGrad with default values.
	mic::neural_nets::optimization::AdaDelta<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with default learning rate.
		opt.update(x, dx, 0.001);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::fabs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::fabs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


/*!
 * Tests AdaDelta on a 20d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, AdaDelta_Convergence) {
	// Optimization function - 20d AdaDelta (with default values).
	mic::neural_nets::optimization::AdaDelta<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with default learning rate.
		opt.update(x, dx, 0.001);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::fabs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::fabs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


/*!
 * Tests AdaDelta on a 2d Beale's function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Beale2DLandscape, DISABLED_AdaDelta_Convergence) {
	// Optimization AdaDelta.
	mic::neural_nets::optimization::AdaDelta<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with default learning rate.
		opt.update(x, dx, 0.001);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::fabs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::fabs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}



