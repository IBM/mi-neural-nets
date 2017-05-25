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
 * \brief Test Fixture - layer of input size 5x5x1 and with filterbank 3x3x1, floats.
 * Math example taken from: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
 * \author tkornuta
 */
class Conv5x5x1Filter3x3x1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv5x5x1Filter3x3x1Float () : layer(5,5,1,3,1,1) {

		input = MAKE_MATRIX_PTR(float, 5, 5);

		desired_output = MAKE_MATRIX_PTR(float, 3, 3);

	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W0"]) << 1, 0, 1, 0, 1, 0, 1, 0, 1;
		(*layer.p["b0"]) << 0;

		(*input) << 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0;
		(*desired_output) << 4, 3, 4, 2, 4, 3, 2, 3, 4;

	}

private:
	// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	// Test input - used in forward pass.
	mic::types::MatrixPtr<float> input;

	mic::types::MatrixPtr<float> desired_output;

};




#endif /* LINEARLAYERTESTS_HPP_ */
