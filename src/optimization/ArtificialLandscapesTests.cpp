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
 * @file: ArtificialLandscapesTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 12, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include <optimization/ArtificialLandscapesTests.hpp>


/*!
 * Tests 1d square function value.
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, Value) {
	(*x)[0] = 0.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 0.0), eps);
	(*x)[0] = 1.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 1.0), eps);
	(*x)[0] = 2.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 4.0), eps);
}


/*!
 * Tests 1d square function gradient.
 * \author tkornuta
 */
TEST_F(Sphere1DLandscape, Gradient) {
	(*x)[0] = 0.0;
	ASSERT_LE(std::abs((*fun.calculateGradient(x))[0] - 0.0), eps);
	(*x)[0] = 1.0;
	ASSERT_LE(std::abs((*fun.calculateGradient(x))[0] - 2.0), eps);
	(*x)[0] = 2.0;
	ASSERT_LE(std::abs((*fun.calculateGradient(x))[0] - 4.0), eps);
}


/*!
 * Tests 3d square function gradient.
 * \author tkornuta
 */
TEST(Sphere3DLandscape, Gradient) {
	mic::neural_nets::optimization::artificial_landscapes::SphereFunction<double> fun(3);
	mic::types::MatrixPtr<double> x = MAKE_MATRIX_PTR(double, 3,1);
	mic::types::MatrixPtr<double> dx;
	/// Eps.
	double eps = 1e-10;

	// 0.0
	for (size_t i=0; i<3; i++)
		(*x)[i] = 0.0;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<3; i++)
		ASSERT_LE(std::abs((*dx)[i] - 0.0), eps);

	// 2.0
	for (size_t i=0; i<3; i++)
		(*x)[i] = 2.0;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<3; i++)
		ASSERT_LE(std::abs((*dx)[i] - 4.0), eps);

	// 0,1,2
	for (size_t i=0; i<3; i++)
		(*x)[i] = i;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<3; i++)
		ASSERT_LE(std::abs((*dx)[i] - 2*i), eps);
}


/*!
 * Tests 20d square function value.
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, Value) {
	for (size_t i=0; i<20; i++)
		(*x)[i] = 0.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 0.0), eps);
	for (size_t i=0; i<20; i++)
		(*x)[i] = 1.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 20.0), eps);
	for (size_t i=0; i<20; i++)
		(*x)[i] = 2.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 80.0), eps);

	for (size_t i=0; i<20; i++)
		(*x)[i] = i;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 2470.0), eps);
}


/*!
 * Tests 320d square function gradient.
 * \author tkornuta
 */
TEST_F(Sphere20DLandscape, Gradient) {
	mic::types::MatrixPtr<double> dx;

	// 0.0
	for (size_t i=0; i<20; i++)
		(*x)[i] = 0.0;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<20; i++)
		ASSERT_LE(std::abs((*dx)[i] - 0.0), eps);

	// 2.0
	for (size_t i=0; i<20; i++)
		(*x)[i] = 2.0;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<20; i++)
		ASSERT_LE(std::abs((*dx)[i] - 4.0), eps);

	// 0,1,2,...
	for (size_t i=0; i<20; i++)
		(*x)[i] = i;
	dx = fun.calculateGradient(x);
	for (size_t i=0; i<3; i++)
		ASSERT_LE(std::abs((*dx)[i] - 2*i), eps);
}

/*!
 * Checks gradient value in minimum of  Beale's function.
 * \author tkornuta
 */
TEST_F(Beale2DLandscape, GradientInMin) {
	(*x)[0] = 3.0;
	(*x)[1] = 0.5;
	mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
	ASSERT_LE(std::abs((*dx)[0]), eps);
	ASSERT_LE(std::abs((*dx)[1]), eps);
}


/*!
 * Tests 2d Rosenbrock function value.
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, Value) {
	(*x)[0] = 0.0;
	(*x)[1] = 0.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 1.0), eps);
	(*x)[0] = 1.0;
	(*x)[1] = 1.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 0.0), eps);
	(*x)[0] = 2.0;
	(*x)[1] = 2.0;
	ASSERT_LE(std::abs(fun.calculateValue(x) - 401.0), eps);
}

/*!
 * Checks Rosenbrock function gradient in minimum.
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, GradientInMin) {
	(*x)[0] = 1;
	(*x)[1] = 1;
	mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
	ASSERT_LE(std::abs((*dx)[0]), eps);
	ASSERT_LE(std::abs((*dx)[1]), eps);
}

/*!
 * Tests 2D Rosenbrock function gradients.
 * \author tkornuta
 */
TEST_F(Rosenbrock2DLandscape, Gradient) {
	(*x)[0] = 2;
	(*x)[1] = 0;
	mic::types::MatrixPtr<double> dx = fun.calculateGradient(x);
	ASSERT_LE(std::abs((*dx)[0] - 3202.0), eps);
	ASSERT_LE(std::abs((*dx)[1] + 800.0), eps);
	(*x)[0] = 2;
	(*x)[1] = 2;
	dx = fun.calculateGradient(x);
	ASSERT_LE(std::abs((*dx)[0] - 1602.0), eps);
	ASSERT_LE(std::abs((*dx)[1] + 400.0), eps);
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


