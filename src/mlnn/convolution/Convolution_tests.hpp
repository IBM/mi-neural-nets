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

namespace mic { namespace neural_nets { namespace unit_tests {


/*!
 * \brief Test Fixture - layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1, floats.
 * Math example taken from my own calculations;)
 * \author tkornuta
 */
class Conv2x2x2Filter2x1x1s1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv2x2x2Filter2x1x1s1Float () : layer(2,2,2,2,1,1) {

		x = MAKE_MATRIX_PTR(float, 8, 1);

		desired_y = MAKE_MATRIX_PTR(float, 8, 1);

		dy = MAKE_MATRIX_PTR(float, 8, 1);

		desired_dx = MAKE_MATRIX_PTR(float, 8, 1);

	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W00"]) << 0;
		(*layer.p["W01"]) << -1;
		(*layer.p["b0"]) << 1;
		(*layer.p["W10"]) << 1;
		(*layer.p["W11"]) << 2;
		(*layer.p["b1"]) << 0;

		(*x) << 2, 1, 0, 0, 1, 2, -1, 0;
		(*desired_y) << 0, -1, 2, 1, 4, 5, -2, 0;

		(*dy) << 1, 0, -1, 2, 0, 1, 1, -1;
		(*desired_dx) <<  0, 1, 1, -1, -1, 2, 3, -4;

	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test x - used in forward pass.
	mic::types::MatrixPtr<float> x;

	/// Desired output for a given x.
	mic::types::MatrixPtr<float> desired_y;

	/// Gradient passed to backpropagation.
	mic::types::MatrixPtr<float> dy;

	/// Desired gradient dy from backpropagation.
	mic::types::MatrixPtr<float> desired_dx;

};



/*!
 * \brief Test Fixture - layer of input size 5x5x1 and with filter bank of 1 filter of size 3x3 with stride 1 (floats).
 * Math example taken from: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
 * \author tkornuta
 */
class Conv5x5x1Filter1x3x3s1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv5x5x1Filter1x3x3s1Float () : layer(5,5,1,1,3,1) {

		x = MAKE_MATRIX_PTR(float, 25, 1);

		desired_y = MAKE_MATRIX_PTR(float, 9, 1);

		dy = MAKE_MATRIX_PTR(float, 9, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W00"]) << 1, 0, 1, 0, 1, 0, 1, 0, 1;
		(*layer.p["b0"]) << 0;

		(*x) << 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0;
		(*desired_y) << 4, 3, 4, 2, 4, 3, 2, 3, 4;

	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test x - used in forward pass.
	mic::types::MatrixPtr<float> x;

	/// Desired output for a given x.
	mic::types::MatrixPtr<float> desired_y;

	/// Gradient passed to backpropagation.
	mic::types::MatrixPtr<float> dy;

	/// Desired gradient dy from backpropagation.
	mic::types::MatrixPtr<float> desired_dx;
};



/*!
 * \brief Test Fixture - layer of input size 3x3x2 and with filter bank of 3 filters of size 2x2 with stride 1, floats.
 * Math example taken from my whiteboard;)
 * \author tkornuta
 */
class Conv3x3x2Filter3x2x2s1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv3x3x2Filter3x2x2s1Float () : layer(3,3,2,3,2,1) {

		x = MAKE_MATRIX_PTR(float, 18, 1);

		desired_y = MAKE_MATRIX_PTR(float, 12, 1);

	}

protected:
	// Sets values
	virtual void SetUp() {
		(*layer.p["W00"]) << 0, 1, 1, 0;
		(*layer.p["W01"]) << 0, -1, -1, 0;
		(*layer.p["b0"]) << 1;
		(*layer.p["W10"]) << -1, 0, 0, 1;
		(*layer.p["W11"]) << 1, 0, 0, -1;
		(*layer.p["b1"]) << 0;
		(*layer.p["W20"]) << 0, 0, 1, 1;
		(*layer.p["W21"]) << 0, 0, -1, -1;
		(*layer.p["b2"]) << -1;

		(*x) << 1, 4, 7, 2, 5, 8, 3, 6, 9, 9, 6, 3, 8, 5, 2, 7, 4, 1;
		(*desired_y) << -7, 5, -3, 9, 8, 8, 8, 8, -7, 5, -3, 9;

	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test x - used in forward pass.
	mic::types::MatrixPtr<float> x;

	/// Desired output for a given x.
	mic::types::MatrixPtr<float> desired_y;

};



/*!
 * \brief Test Fixture - layer of input size 7x7x3 and with filter bank of 2 filters of 3x3 with stride 2 (floats).
 * Math example taken from: http://cs231n.github.io/convolutional-networks/
 * \author tkornuta
 */
class Conv7x7x3Filter3x3x3s2Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv7x7x3Filter3x3x3s2Float () : layer(7,7,3,2,3,2) {

		x = MAKE_MATRIX_PTR(float, 7*7*3, 1);

		desired_y = MAKE_MATRIX_PTR(float, 3*3*2, 1);

	}

protected:
	// Sets values
	virtual void SetUp() {

		(*layer.p["W00"]) << 0, -1, 0, 0, 1, -1, 1, 1, -1;
		(*layer.p["W01"]) << 1, 0, 1, 0, -1, -1, 1, 1, -1;
		(*layer.p["W02"]) << 1, 1, 0, -1, 1, -1, 1, 0, 1;
		(*layer.p["b0"]) << 1;

		(*layer.p["W10"]) << 1, 1, -1, -1, -1, 1, 0, -1, -1;
		(*layer.p["W11"]) << 0, 1, 1, -1, 1, -1, 0, -1, -1;
		(*layer.p["W12"]) << 0, 0, 0, 1, 1, -1, -1, 0, 1;
		(*layer.p["b1"]) << 0;


		(*x) <<
				// x[:,:,0]
				0, 0, 0, 0, 0, 0, 0,
				0, 0, 2, 1, 1, 2, 0,
				0, 2, 0, 1, 0, 1, 0,
				0, 2, 2, 2, 0, 0, 0,
				0, 2, 2, 1, 2, 0, 0,
				0, 0, 1, 1, 2, 0, 0,
				0, 0, 0, 0, 0, 0, 0,
				// x[:,:,1]
				0, 0, 0, 0, 0, 0, 0,
				0, 0, 2, 0, 2, 0, 0,
				0, 1, 2, 2, 2, 2, 0,
				0, 0, 0, 0, 1, 2, 0,
				0, 1, 0, 0, 1, 2, 0,
				0, 0, 0, 1, 0, 2, 0,
				0, 0, 0, 0, 0, 0, 0,
				// x[:,:,2]
				0, 0, 0, 0, 0, 0, 0,
				0, 0, 2, 1, 2, 2, 0,
				0, 1, 2, 0, 2, 0, 0,
				0, 2, 0, 1, 0, 2, 0,
				0, 1, 2, 2, 2, 0, 0,
				0, 2, 2, 0, 2, 0, 0,
				0, 0, 0, 0, 0, 0, 0;

		(*desired_y) <<
				// o[:,:,0]
				-2, 3, 10, 7, 12, 11, -1, -1, 0,
				// o[:,:,1]
				-5, -10, -6, 4, -3, 2, 2, 3, 6;
	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test x - used in forward pass.
	mic::types::MatrixPtr<float> x;

	/// Desired output for a given x.
	mic::types::MatrixPtr<float> desired_y;

};




/*!
 * \brief Test Fixture - layer of input size 5x6x1 and with filter bank of 1 filter of size 4x4 with stride 1, floats.
 * Math example taken from: http://soumith.ch/ex/pages/2014/08/07/why-rotate-weights-convolution-dy/
 * \author tkornuta
 */
class Conv5x6x1Filter1x4x4s1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Conv5x6x1Filter1x4x4s1Float () : layer(5,6,1,1,4,1) {

		x = MAKE_MATRIX_PTR(float, 6, 5);

		desired_y = MAKE_MATRIX_PTR(float, 6,1);

		dy = MAKE_MATRIX_PTR(float, 6, 1);
		desired_dx = MAKE_MATRIX_PTR(float, 30, 1);
	}

protected:
	// Sets values
	virtual void SetUp() {
		for (size_t i=0; i<16; i++)
			(*layer.p["W00"])(i) = i+1;
		(*layer.p["W00"]).resize(4,4);
		(*layer.p["W00"]).transposeInPlace();
		//std::cout<<"*layer.p[W00] = \n" << (*layer.p["W00"]) << std::endl;
		(*layer.p["W00"]).resize(1, 4*4);
		(*layer.p["b0"]) << 0;

		for (size_t i=0; i<30; i++)
			(*x)(i) = i+1;
		(*x).transposeInPlace();
		//std::cout<<"*x = \n" << (*x) << std::endl;
		(*x).resize(30,1);

		(*desired_y) << 2064, 2880, 2200, 3016, 2336, 3152;
		//std::cout<<"*desired_y = \n" << (*desired_y) << std::endl;

		(*dy) << 1, 4, 2, 5, 3, 6;
		// IMPROPER VALUES!
		// (*desired_dx) <<  1, 9, 29, 49, 52, 4, 29, 77, 125, 121, 10, 62, 146, 230, 208, 16, 83, 167, 251, 223, 17, 75, 139, 203, 170, 12, 48, 84, 120, 96;
	}

private:
	/// Object to be tested.
	mic::mlnn::convolution::Convolution<float> layer;

	/// Test x - used in forward pass.
	mic::types::MatrixPtr<float> x;

	/// Desired output for a given x.
	mic::types::MatrixPtr<float> desired_y;

	/// Gradient passed to backpropagation.
	mic::types::MatrixPtr<float> dy;

	/// Desired gradient dy from backpropagation.
	mic::types::MatrixPtr<float> desired_dx;

};


} } } //: namespaces

#endif /* LINEARLAYERTESTS_HPP_ */
