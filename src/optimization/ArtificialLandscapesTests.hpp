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
	mic::neural_nets::optimization::SphereFunction<double> fun;

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
	mic::neural_nets::optimization::SphereFunction<double> fun;

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
	mic::neural_nets::optimization::Beale2DFunction<double> fun;

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
	mic::neural_nets::optimization::Rosenbrock2DFunction<double> fun;

	/// Eps.
	double eps = 1e-5;

	// Variables.
	mic::types::MatrixPtr<double> x;
};



#endif /* ARTIFICIALLANDSCAPESTESTS_HPP_ */
