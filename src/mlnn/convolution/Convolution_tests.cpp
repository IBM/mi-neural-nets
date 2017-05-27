/*!
 * @file: Linear_tests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 4, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "Convolution_tests.hpp"

namespace mic { namespace neural_nets { namespace unit_tests {

/*!
 * Checks numbers of receptive fields for different strides.
 * \author tkornuta
 */
TEST(Convolutions, NumberOfReceptiveFields) {

	// Stride = 1.
	mic::mlnn::convolution::Convolution<float> l1(5,7,1,1,3,1);

	ASSERT_EQ(l1.number_of_receptive_fields_vertical, 3);
	ASSERT_EQ(l1.number_of_receptive_fields_horizontal, 5);
	ASSERT_EQ(l1.s["y"]->rows(), 3*5*1);

	// Stride = 2.
	mic::mlnn::convolution::Convolution<float> l2(5,7,1,1,3,2);

	ASSERT_EQ(l2.number_of_receptive_fields_vertical, 2);
	ASSERT_EQ(l2.number_of_receptive_fields_horizontal, 3);
	ASSERT_EQ(l1.s["y"]->rows(), 3*5*1);

}


/*!
 * Checks whether the forward is working for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Float, Forward) {

	std::cout<<"W00 = \n" << (*layer.p["W00"]) <<std::endl;
	std::cout<<"W01 = \n" << (*layer.p["W01"]) <<std::endl;
	std::cout<<"W10 = \n" << (*layer.p["W10"]) <<std::endl;
	std::cout<<"W11 = \n" << (*layer.p["W11"]) <<std::endl;
	std::cout<<"x = \n" << (*x) <<std::endl;
	std::cout<<"desired_y = \n" << (*desired_y) <<std::endl;

	// Forward pass.
	mic::types::MatrixPtr<float> y = layer.forward(x);
	//std::cout<<"output = \n" << (*output) <<std::endl;

	// Check output.
	for (size_t i=0; i<8; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the backward pass is working for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Float,  Backward) {

	// Forward pass.
	mic::types::MatrixPtr<float> dx = layer.backward(dy);
	//std::cout<<"dx = \n" << (*dx).transpose() <<std::endl;

	// Check output.
	for (size_t i=0; i<8; i++)
		ASSERT_EQ((*desired_dx)[i], (*dx)[i]) << "at position " << i;

	// Second backward - just to assure that all the "internal dimensions" are ok after the first pass.
	//layer.backward(dy);
}


/*!
 * Checks whether the forward is working for layer of input size 3x3x2 and with filter bank of 3 filters of size 2x2 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv3x3x2Filter3x2x2s1Float, Forward) {

/*	std::cout<<"W0 = \n" << (*layer.p["W0"]) <<std::endl;
	std::cout<<"W1 = \n" << (*layer.p["W1"]) <<std::endl;
	std::cout<<"W2 = \n" << (*layer.p["W2"]) <<std::endl;
	std::cout<<"x = \n" << (*x) <<std::endl;
	std::cout<<"desired_y = \n" << (*desired_y) <<std::endl;*/

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(x);
	//std::cout<<"output = \n" << (*output) <<std::endl;

	// Check output.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_y)[i]) << "at position " << i;

	// Second forward - just to check whether "internal dimensions" are ok after the first pass.
	output = layer.forward(x);
}



/*!
 * Checks whether dimensions of xs, outputs and filters are ok.
 * Convolutional dimensions are nicely explained in this lecture:
 * http://cs231n.github.io/convolutional-networks/
  * \author tkornuta
 */
TEST_F(Conv5x5x1Filter1x3x3s1Float, Dimensions) {

	// Check filter size - W.
	ASSERT_EQ((*layer.p["W00"]).rows(), 1);
	ASSERT_EQ((*layer.p["W00"]).cols(), 9);

	// Check filter size - b.
	ASSERT_EQ((*layer.p["b"]).rows(), 1);
	ASSERT_EQ((*layer.p["b"]).cols(), 1);

	// Assert input size.
	ASSERT_EQ((*layer.s["x"]).rows(), 25);
	ASSERT_EQ((*layer.s["x"]).cols(), 1);

	// Assert output size.
	ASSERT_EQ((*layer.s["y"]).rows(), 9);
	ASSERT_EQ((*layer.s["y"]).cols(), 1);

}


/*!
 * Checks whether the forward is working for layer of input size 5x5x1 and with filter bank of 1 filter of size 3x3 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv5x5x1Filter1x3x3s1Float, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(x);

	// Check output.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the forward is working for layer of input size 5x6x1 and with filter bank of 1 filter of size 4x4 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv5x6x1Filter1x4x4s1Float, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(x);
	//std::cout<<"output = \n" << (*output) <<std::endl;

	// Check output.
	for (size_t i=0; i<6; i++)
		ASSERT_EQ((*output)[i], (*desired_y)[i]) << "at position " << i;

}



/*!
 * Checks whether the forward is working for layer of input size 7x7x3 and with filter bank of 3 filters of 3x3 size with stride 2.
 * \author tkornuta
 */
TEST_F(Conv7x7x3Filter3x3x3s2Float, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(x);
	//std::cout<<"output = \n" << (*output) <<std::endl;

	// Check output.
	for (size_t i=0; i<18; i++)
		ASSERT_EQ((*output)[i], (*desired_y)[i]) << "at position " << i;

}




} } } //: namespaces

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
