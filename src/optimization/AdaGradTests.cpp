/*!
 * @file: AdaGradTests.cpp
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
#include <optimization/AdaGrad.hpp>


/////////////////////////////////////////////////////////////////////////
// AdaGrad
/////////////////////////////////////////////////////////////////////////


/*!
 * Tests AdaGrad on a 1d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, DISABLED_AdaGrad_Convergence) {
	// Optimization function - 1d AdaGrad with learning rate = 0.1 (and default 0.9 momentum).
	mic::neural_nets::optimization::AdaGrad<double> opt(1);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		opt.update(x, dx);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


/*!
 * Tests AdaGrad on a 20d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, DISABLED_AdaGrad_Convergence) {
	// Optimization function - 20d AdaGrad (with default values).
	mic::neural_nets::optimization::AdaGrad<double> opt(20);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		opt.update(x, dx);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


