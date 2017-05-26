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
	mic::mlnn::convolution::Convolution<float> l1(5,7,1,3,1,1);

	ASSERT_EQ(l1.number_of_receptive_fields_vertical, 3);
	ASSERT_EQ(l1.number_of_receptive_fields_horizontal, 5);
	ASSERT_EQ(l1.s["y"]->rows(), 3*3*1);

	// Stride = 2.
	mic::mlnn::convolution::Convolution<float> l2(5,7,1,3,2,1);

	ASSERT_EQ(l2.number_of_receptive_fields_vertical, 2);
	ASSERT_EQ(l2.number_of_receptive_fields_horizontal, 3);
	ASSERT_EQ(l1.s["y"]->rows(), 3*3*1);

}



/*!
 * Checks whether dimensions of inputs, outputs and filters are ok.
 * Convolutional dimensions are nicely explained in this lecture:
 * http://cs231n.github.io/convolutional-networks/
  * \author tkornuta
 */
TEST_F(Conv5x5x1Filter3x3x1Float, Dimensions) {

	// Check filter size - W.
	ASSERT_EQ((*layer.p["W0"]).rows(), 1);
	ASSERT_EQ((*layer.p["W0"]).cols(), 9);

	// Check filter size - b.
	ASSERT_EQ((*layer.p["b0"]).rows(), 1);
	ASSERT_EQ((*layer.p["b0"]).cols(), 1);

	// Assert input size.
	ASSERT_EQ((*layer.s["x"]).rows(), 25);
	ASSERT_EQ((*layer.s["x"]).cols(), 1);

	// Assert output size.
	ASSERT_EQ((*layer.s["y"]).rows(), 9);
	ASSERT_EQ((*layer.s["y"]).cols(), 1);

}


/*!
 * Checks whether the forward is working for layer of input size 5x5x1 and with filter bank of 3x3 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv5x5x1Filter3x3x1Float, Forward) {

/*	std::cout<<"W = \n" << (*layer.p["W0"]) <<std::endl;
	std::cout<<"input = \n" << (*input) <<std::endl;*/

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(input);

	// Check output.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_output)[i]);

}

/*!
 * Checks whether the forward is working.
 * \author tkornuta
 */
TEST_F(Conv3x3x2Filter2x2x3Float, Forward) {

/*	std::cout<<"W0 = \n" << (*layer.p["W0"]) <<std::endl;
	std::cout<<"W1 = \n" << (*layer.p["W1"]) <<std::endl;
	std::cout<<"W2 = \n" << (*layer.p["W2"]) <<std::endl;
	std::cout<<"input = \n" << (*input) <<std::endl;
	std::cout<<"desired_output = \n" << (*desired_output) <<std::endl;*/

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(input);
	// Second forward - just to check whether "internal dimensions" are ok after the first pass.
	output = layer.forward(input);
	//std::cout<<"output = \n" << (*output) <<std::endl;

	// Check output.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_output)[i]);

}


/*!
 * Checks whether the backward pass is working for layer of input size 5x5x1 and with filter bank of 3x3 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv5x5x1Filter3x3x1Float, Backward) {

	/*std::cout<<"W = \n" << (*layer.p["W0"]) <<std::endl;
	std::cout<<"input = \n" << (*input) <<std::endl;*/

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.backward(input);

	// Check output.
/*	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_output)[i]);*/

}

} } } //: namespaces

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
