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
 * @file: GradPIDTests.cpp
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
#include <optimization/GradPID.hpp>


/////////////////////////////////////////////////////////////////////////
// GradPID
/////////////////////////////////////////////////////////////////////////


/*!
 * Tests GradPID on a 1d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, GradPID_Convergence) {
	// Optimization function - 1d GradPID (with default values).
	mic::neural_nets::optimization::GradPID<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double value = fun.calculateValue(x);
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		// Calculate gradient.
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		opt.update(x, dx, 0.1);

		// Check whether value of the function is finite.
		value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}



/*!
 * Tests GradPID on a 20d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, GradPID_Convergence) {
	for (size_t i=0; i<20; i++)
		(*x)[i] = i;
	// Optimization function - 20d GradPID (with default values).
	mic::neural_nets::optimization::GradPID<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		opt.update(x, dx, 0.1);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
/*		if (iteration >135)
			exit(1);*/
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


/*!
 * Tests GradPID on a 2d Beale's function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Beale2DLandscape, GradPID_Convergence) {
	// Optimization function - 2d GradPID.
	mic::neural_nets::optimization::GradPID<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
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
 * Tests GradPID on a 2d Rosenbrock function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, GradPID_Convergence) {
	// Optimization function - 2d GradPID.
	double ni = 0.0001;
	mic::neural_nets::optimization::GradPID<double> opt(x->rows(), x->cols());

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
		opt.update(x, dx, ni);

		// Check whether value of the function is finite.
		double value = fun.calculateValue(x);
		ASSERT_EQ(true, std::isfinite(value)) << " at iteration i=" << iteration << " for ni =" << ni;

		// Calculate diff - std:abs!!
		abs_diff = std::abs(value - fun.minValue());
		iteration++;
	}//: while
	ASSERT_GE(eps, std::abs(fun.calculateValue(x) - fun.minValue()));
	std::cout << "              -> Converged after " << iteration << " iterations\n";
}


