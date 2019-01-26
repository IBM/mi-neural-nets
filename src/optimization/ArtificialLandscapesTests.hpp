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
 * @file: ArtificialLandscapesTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 12, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef ARTIFICIALLANDSCAPESTESTS_HPP_
#define ARTIFICIALLANDSCAPESTESTS_HPP_

#include <gtest/gtest.h>
#include <cmath>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#include <optimization/ArtificialLandscapes.hpp>


/*!
 * \brief Test fixture - artificial landscape - sphere function 1D (square function).
 * \author tkornuta
 */
class Sphere1DLandscape : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Sphere1DLandscape () : fun(1) {
		x = MAKE_MATRIX_PTR(double, 1, 1);
	}

	// Sets values
	virtual void SetUp() {
		(*x)[0] = 20.0;
	}

private:
	/// Sphere function 1d.
	mic::neural_nets::optimization::artificial_landscapes::SphereFunction<double> fun;

	/// Eps.
	double eps = 1e-5;

	// Variables.
	mic::types::MatrixPtr<double> x;
};

/*!
 * \brief Test fixture - artificial landscape - sphere function 20D (square function).
 * \author tkornuta
 */
class Sphere20DLandscape : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Sphere20DLandscape () : fun(20) {
		x = MAKE_MATRIX_PTR(double, 20, 1);
	}

	// Sets values
	virtual void SetUp() {
		for (size_t i=0; i<20; i++)
			(*x)[i] = i;
	}

private:
	/// Sphere function 1d.
	mic::neural_nets::optimization::artificial_landscapes::SphereFunction<double> fun;

	/// Eps.
	double eps = 1e-5;

	// Variables.
	mic::types::MatrixPtr<double> x;
};


/*!
 * \brief Test fixture - artificial landscape - Beale's function 2D.
 * \author tkornuta
 */
class Beale2DLandscape : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Beale2DLandscape () {
		x = MAKE_MATRIX_PTR(double, 2, 1);
	}

	// Sets values
	virtual void SetUp() {
		(*x)[0] = 0.0;
		(*x)[1] = 0.0;
	}


private:
	/// Beale's function 2d.
	mic::neural_nets::optimization::artificial_landscapes::Beale2DFunction<double> fun;

	/// Eps.
	double eps = 1e-5;

	// Variables.
	mic::types::MatrixPtr<double> x;
};



/*!
 * \brief Test fixture - artificial landscape - Rosenbrock function 2D.
 * \author tkornuta
 */
class Rosenbrock2DLandscape : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Rosenbrock2DLandscape () : fun(1,100) {
		x = MAKE_MATRIX_PTR(double, 2, 1);
	}

	// Sets values
	virtual void SetUp() {
		(*x)[0] = 10.0;
		(*x)[1] = 4.0;
	}

private:
	/// Rosenbrock function 2d.
	mic::neural_nets::optimization::artificial_landscapes::Rosenbrock2DFunction<double> fun;

	/// Eps.
	double eps = 1e-5;

	// Variables.
	mic::types::MatrixPtr<double> x;
};



#endif /* ARTIFICIALLANDSCAPESTESTS_HPP_ */
