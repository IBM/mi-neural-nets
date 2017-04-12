/*!
 * @file: GradientDescentTests.cpp
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
#include <optimization/GradientDescent.hpp>



/////////////////////////////////////////////////////////////////////////
// (S)GD
/////////////////////////////////////////////////////////////////////////


/*!
 * Tests gradient descent on a 1d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, GradientDescent_Convergence) {
	// Optimization function - (S)DG.
	mic::neural_nets::optimization::GradientDescent<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with learning rate = 0.1.
		opt.update(x, dx, 0.1);

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
 * Tests gradient descent on a 20d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, GradientDescent_Convergence) {
	// Optimization function - (S)DG with learning rate = 0.1.
	mic::neural_nets::optimization::GradientDescent<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with learning rate = 0.1.
		opt.update(x, dx, 0.1);

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
 * Tests gradient descent on a 2d Beale's function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Beale2DLandscape, GradientDescent_Convergence) {
	// Optimization function - (S)DG with learning rate = 0.1.
	mic::neural_nets::optimization::GradientDescent<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with learning rate = 0.01.
		opt.update(x, dx, 0.01);

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
 * Tests gradient descent on a 2d Rosenbrock function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, GradientDescent_Convergence) {
	// Optimization function - (S)DG.
	// Such a simple gradient descent has a HUUGE problem with instability for bigger ni!
	// Eg. 0.001 - gradiens explodes after 5 iterations!!
	double ni = 0.00001;
	mic::neural_nets::optimization::GradientDescent<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		// Update with learning rate = ni.
		opt.update(x, dx, ni);

/*		std::cout << "i=" << iteration << std::endl;
		std::cout << "(*dx)[0]=" << (*dx)[0] << " (*dx)[1]=" << (*dx)[1] << std::endl;
		std::cout << "(*x)[0]=" << (*x)[0] << " (*x)[1]=" << (*x)[1] << std::endl;
		std::cout << " min value = " << fun.minValue() << std::endl;*/

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
//		std::cout << " value = " << value << std::endl;
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration << " for ni =" << ni;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
//		if (iteration % 1000 == 1)
//			std::cout << "iteration =" << iteration << " abs value diff = " << abs_diff << std::endl;
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}
