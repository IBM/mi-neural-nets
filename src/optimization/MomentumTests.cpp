/*!
 * @file: MomentumTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#include <gtest/gtest.h>
#include <cmath>

#include <optimization/ArtificialLandscapesTests.hpp>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#include <optimization/Momentum.hpp>




/////////////////////////////////////////////////////////////////////////
// Momentum
/////////////////////////////////////////////////////////////////////////


/*!
 * Tests gradient descent with momentum on a 1d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, Momentum_Convergence) {
	// Optimization function - 1d momentum with learning rate = 0.1 (and default 0.9 momentum).
	mic::neural_nets::optimization::Momentum<double> opt(1, 0.1);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types2::MatrixPtr<double> dx = fun.calculateGradient(x);
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
 * Tests gradient descent with momentum on a 20d square "artificial landscape".
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, Momentum_Convergence) {
	// Optimization function - 20d momentum with learning rate = 0.1 (and default 0.9 momentum).
	mic::neural_nets::optimization::Momentum<double> opt(20, 0.01);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types2::MatrixPtr<double> dx = fun.calculateGradient(x);
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
 * Tests gradient descent with momentum on a 2d Beale's function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Beale2DLandscape, Momentum_Convergence) {
	// Optimization function - 2d momentum with learning rate = 0.1 (and default 0.9 momentum).
	mic::neural_nets::optimization::Momentum<double> opt(2, 0.01);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types2::MatrixPtr<double> dx = fun.calculateGradient(x);
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
 * Tests gradient descent with momentum on a 2d Rosenbrock function "artificial landscape".
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, Momentum_Convergence) {
	// Optimization function - 2d momentum with learning rate = 0.1 (and default 0.9 momentum).
	mic::neural_nets::optimization::Momentum<double> opt(2, 0.00001);

	// Simulate a simple gradient descent.
	size_t iteration = 0;
	double abs_diff = 1.0;
	while (abs_diff > eps) {
		mic::types2::MatrixPtr<double> dx = fun.calculateGradient(x);
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
