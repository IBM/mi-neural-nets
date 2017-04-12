/*!
 * @file: LinearTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 4, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef LINEARLAYERTESTS_HPP_
#define LINEARLAYERTESTS_HPP_

#include <gtest/gtest.h>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#define protected public
#include <mlnn/fully_connected/Linear.hpp>
#include <loss/SquaredErrorLoss.hpp>

/*!
 * \brief Test Fixture - layer of size 1x1, floats, sets W[0] = 1.0 and b[0] = 1.0.
 * \author tkornuta
 */
class Linear1x1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Linear1x1Float () : layer(1,1) { }

protected:
	// Sets W and b.
	virtual void SetUp() {
		(*layer.p["W"])[0] = 1.0;
		(*layer.p["b"])[0] = 1.0;
	}


private:
	// Object to be tested.
	mic::mlnn::fully_connected::Linear<float> layer;
};

/*!
 * \brief Test Fixture - layer of size 5x2, floats.
 * \author tkornuta
 */
class Linear5x2Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Linear5x2Float () : layer(5,2) { }

private:
	// Object to be tested.
	mic::mlnn::fully_connected::Linear<float> layer;
};


/*!
 * \brief Test Fixture - layer of size 2x3, floats, sets all internal and external values.
 * \author tkornuta
 */
class Linear2x3Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Linear2x3Float () : layer(2,3) {
		const_x = MAKE_MATRIX_PTR(float, 2, 1);
		const_dy = MAKE_MATRIX_PTR(float, 3, 1);
		target_y = MAKE_MATRIX_PTR(float, 3, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W"]) << 1, 2, 3, 5, 6, 9;
		(*layer.p["b"]) << -3, -2, -1;

		(*const_x) << -1, 1;

		(*const_dy) << -1, -2, 1;

		(*target_y) << -1, -2, 1;

		// Reset state and gradients.
		layer.s["x"]->setZero();
		layer.s["y"]->setZero();
		layer.g["x"]->setZero();
		layer.g["y"]->setZero();
	}


private:
	// Object to be tested.
	mic::mlnn::fully_connected::Linear<float> layer;

	// Test input x - used in forward pass.
	mic::types::MatrixPtr<float> const_x;

	// Test gradient dy - used in backward pass.
	mic::types::MatrixPtr<float> const_dy;

	// Target y values.
	mic::types::MatrixPtr<float> target_y;

	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
};


/*!
 * \brief Test Fixture - layer of size 2x3, doubles, sets all internal and external values.
 * \author tkornuta
 */
class Linear2x3Double : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Linear2x3Double () : layer(2,3) {
		const_x = MAKE_MATRIX_PTR(double, 2, 1);
		const_dy = MAKE_MATRIX_PTR(double, 3, 1);
		target_y = MAKE_MATRIX_PTR(double, 3, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W"]) << 1, 2, 3, 5, 6, 9;
		(*layer.p["b"]) << -3, -2, -1;

		(*const_x) << -1, 1;

		(*const_dy) << -1, -2, 1;

		(*target_y) << -1, -2, 1;

		// Reset state and gradients.
		layer.s["x"]->setZero();
		layer.s["y"]->setZero();
		layer.g["x"]->setZero();
		layer.g["y"]->setZero();
	}

private:
	// Object to be tested.
	mic::mlnn::fully_connected::Linear<double> layer;

	// Test input x - used in forward pass.
	mic::types::MatrixPtr<double> const_x;

	// Test gradient dy - used in backward pass.
	mic::types::MatrixPtr<double> const_dy;

	// Target y values.
	mic::types::MatrixPtr<double> target_y;

	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<double> loss;
};


/*!
 * \brief Test Fixture - layer of size 50x100, doubles, randomly sets  all internal and external values required for numerical gradient verification.
 * \author tkornuta
 */
class Linear50x100Double : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Linear50x100Double () : layer(50,100) {
		const_x = MAKE_MATRIX_PTR (double, layer.input_size, 1);
		target_y = MAKE_MATRIX_PTR (double, layer.output_size, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {

		// Initialize random number generator with normal distribution.
		std::random_device rd;
		std::mt19937 mt(rd());

		// Initialize W.
		double range = sqrt(6.0 / double(layer.output_size + layer.input_size));
		std::uniform_real_distribution<double> distW(-range, range);
		for (size_t i = 0; i < layer.output_size * layer.input_size; i++)
			(*layer.p["W"])[i] = (double)distW(rd);

		// Initialize b.
		std::uniform_real_distribution<double> distb(-5.0, 5.0);
		for(size_t row=0; row < layer.output_size; row++)
			(*layer.p["b"])[row] = (double)distb(rd);

		// Initialize x and y.
		std::uniform_real_distribution<double> distxy(-5.0, 5.0);
		for (size_t i = 0; i <layer.input_size; i++)
			(*const_x)[i] = (double)distxy(rd);

		for (size_t i = 0; i < layer.output_size; i++)
			(*target_y)[i] = (double)distxy(rd);

		// Reset state and gradients.
		layer.s["x"]->setZero();
		layer.s["y"]->setZero();
		layer.g["x"]->setZero();
		layer.g["y"]->setZero();
	}

private:
	// Object to be tested.
	mic::mlnn::fully_connected::Linear<double> layer;

	// Test input x - used in forward pass.
	mic::types::MatrixPtr<double> const_x;

	// Target y values.
	mic::types::MatrixPtr<double> target_y;

	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<double> loss;
};
#endif /* LINEARLAYERTESTS_HPP_ */
