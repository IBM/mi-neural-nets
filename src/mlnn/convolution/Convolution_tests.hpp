/*!
 * @file: Linear_tests.hpp
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
#include <mlnn/convolution/Convolution.hpp>
#include <loss/SquaredErrorLoss.hpp>


/*!
 * \brief Test Fixture - layer of input size 5x5x1 and with filter bank of 3x3 with stride 1 (floats).
 * Math example taken from: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
 * \author tkornuta
 */
class Conv5x5x1Filter3x3x1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv5x5x1Filter3x3x1Float () : layer(5,5,1,3,1,1) {

		input = MAKE_MATRIX_PTR(float, 25, 1);

		desired_output = MAKE_MATRIX_PTR(float, 9, 1);

		gradient = MAKE_MATRIX_PTR(float, 9, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W0"]) << 1, 0, 1, 0, 1, 0, 1, 0, 1;
		(*layer.p["b0"]) << 0;

		(*input) << 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0;
		(*desired_output) << 4, 3, 4, 2, 4, 3, 2, 3, 4;

		(*gradient) << 1, 1, -1, 1, 0, -1, 0, 1, 0;
	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test input - used in forward pass.
	mic::types::MatrixPtr<float> input;

	/// Desired output for a given input.
	mic::types::MatrixPtr<float> desired_output;

	/// Gradient passed to backpropagation.
	mic::types::MatrixPtr<float> gradient;
};



/*!
 * \brief Test Fixture - layer of input size 3x3x2 and with filter bank of 2x3 with stride 1, floats.
 * Math example taken from my whiteboard;)
 * \author tkornuta
 */
class Conv3x3x2Filter2x2x3Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv3x3x2Filter2x2x3Float () : layer(3,3,2,2,1,3) {

		input = MAKE_MATRIX_PTR(float, 18, 1);

		desired_output = MAKE_MATRIX_PTR(float, 24, 1);

	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W0"]) << 0, 1, 1, 0;
		(*layer.p["b0"]) << 1;
		(*layer.p["W1"]) << -1, 0, 0, 1;
		(*layer.p["b1"]) << 0;
		(*layer.p["W2"]) << 0, 0, 1, 1;
		(*layer.p["b2"]) << -1;

		(*input) << 1, 4, 7, 2, 5, 8, 3, 6, 9, 9, 6, 3, 8, 5, 2, 7, 4, 1;
		(*desired_output) << 7, 13, 9, 15, 4, 4, 4, 4, 6, 12, 8, 14, 15, 9, 13, 7, -4, -4, -4, -4, 12, 6, 10, 4;

	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test input - used in forward pass.
	mic::types::MatrixPtr<float> input;

	/// Desired output for a given input.
	mic::types::MatrixPtr<float> desired_output;

};


#endif /* LINEARLAYERTESTS_HPP_ */
