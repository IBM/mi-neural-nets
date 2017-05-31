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
TEST(Convolutions, LayerDimensions) {

	// Stride = 1.
	mic::mlnn::convolution::Convolution<float> l1(5,7,1,1,3,1);

	ASSERT_EQ(l1.output_height, 3);
	ASSERT_EQ(l1.output_width, 5);
	ASSERT_EQ(l1.s["y"]->rows(), 3*5*1);
	// Stride = 2.
	mic::mlnn::convolution::Convolution<float> l2(5,7,1,1,3,2);
	ASSERT_EQ(l2.output_height, 2);
	ASSERT_EQ(l2.output_width, 3);
	ASSERT_EQ(l2.s["y"]->rows(), 2*3*1);

	// Stride = 3.
	mic::mlnn::convolution::Convolution<float> l3(10,13,1,1,4,3);
	ASSERT_EQ(l3.output_height, 3);
	ASSERT_EQ(l3.output_width, 4);
	ASSERT_EQ(l3.s["y"]->rows(), 3*4*1);

	// Stride = 3.
	mic::mlnn::convolution::Convolution<float> l4(5,8,1,1,2,3);
	ASSERT_EQ(l4.output_height, 2);
	ASSERT_EQ(l4.output_width, 3);
	ASSERT_EQ(l4.s["y"]->rows(), 2*3*1);

	// Stride = 4.
	mic::mlnn::convolution::Convolution<float> l5(11,15,1,1,3,4);
	ASSERT_EQ(l5.output_height, 3);
	ASSERT_EQ(l5.output_width, 4);
	ASSERT_EQ(l5.s["y"]->rows(), 3*4*1);

	// Stride = 4.
	mic::mlnn::convolution::Convolution<float> l6(6,10,1,1,2,4);
	ASSERT_EQ(l6.output_height, 2);
	ASSERT_EQ(l6.output_width, 3);
	ASSERT_EQ(l6.s["y"]->rows(), 2*3*1);

}


/*!
 * Checks whether the forward is working for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Double, Forward) {

	/*std::cout<<"W00 = \n" << (*layer.p["W00"]) <<std::endl;
	std::cout<<"W01 = \n" << (*layer.p["W01"]) <<std::endl;
	std::cout<<"W10 = \n" << (*layer.p["W10"]) <<std::endl;
	std::cout<<"W11 = \n" << (*layer.p["W11"]) <<std::endl;
	std::cout<<"x = \n" << (*x) <<std::endl;
	std::cout<<"desired_y = \n" << (*desired_y) <<std::endl;*/

	// Forward pass.
	mic::types::MatrixPtr<double> y = layer.forward(x);
	//std::cout<<"y = \n" << (*y) <<std::endl;

	// Check output.
	for (size_t i=0; i<8; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the backward pass is working for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Double, Backward) {

	// Backward pass - need to set x.
	layer.forward(x);
	mic::types::MatrixPtr<double> dx = layer.backward(dy);
	//std::cout<<"dx = \n" << (*dx).transpose() <<std::endl;

	// Check resulting dx gradient.
	for (size_t i=0; i<8; i++)
		ASSERT_EQ((*desired_dx)[i], (*dx)[i]) << "at position " << i;

	mic::types::MatrixPtr<double> db = layer.g["b"];
	// Check resulting db gradient.
	for (size_t i=0; i<2; i++)
		ASSERT_EQ((*desired_db)[i], (*db)[i]) << "at position " << i;

	// Check resulting dW gradient.
	std::cout<<"dW0x0 = \n" << (*layer.g["W0x0"]).transpose() <<std::endl;
	std::cout<<"dW1x1 = \n" << (*layer.g["W1x1"]).transpose() <<std::endl;
	std::cout<<"dW0x1 = \n" << (*layer.g["W0x1"]).transpose() <<std::endl;
	std::cout<<"dW1x0 = \n" << (*layer.g["W1x0"]).transpose() <<std::endl;

	ASSERT_EQ((*desired_dW)[0], (*layer.g["W0x0"])[0]);
	ASSERT_EQ((*desired_dW)[1], (*layer.g["W1x1"])[0]);
	ASSERT_EQ((*desired_dW)[2], (*layer.g["W0x1"])[0]);
	ASSERT_EQ((*desired_dW)[3], (*layer.g["W1x0"])[0]);

	// Second backward - just to assure that all the "internal dimensions" are ok after the first pass.
//	layer.backward(dy);
}



/*!
 * Checks whether the forward is working for layer of input size 3x3x2 and with filter bank of 3 filters of size 2x2 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv3x3x2Filter3x2x2s1Float, Forward) {

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
 * Checks whether the forward is working for layer of input size 4x4x1 and with filter bank of 1 filters of size 2x2 with stride 2.
 * \author tkornuta
 */
TEST_F(Conv4x4x1Filter1x2x2s2Float, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<float> y = layer.forward(x);

	// Check output.
	for (size_t i=0; i<4; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the backward gradient pass is working for layer of input size 4x4x1 and with filter bank of 1 filters of size 2x2 with stride 2.
 * \author tkornuta
 */
TEST_F(Conv4x4x1Filter1x2x2s2Float, Backward) {

	// Backward pass - need to set x.
	layer.forward(x);
	mic::types::MatrixPtr<float> dx = layer.backward(dy);

	// Check resulting dx gradient.
	for (size_t i=0; i<8; i++)
		ASSERT_EQ((*desired_dx)[i], (*dx)[i]) << "at position " << i;

	// Check resulting db gradient.
	mic::types::MatrixPtr<float> db = layer.g["b"];
	ASSERT_EQ((*desired_db)[0], (*db)[0]);

	// Check resulting dW gradient.
	mic::types::MatrixPtr<float> dW = layer.g["W0x0"];
	for (size_t i=0; i<4; i++)
		ASSERT_EQ((*desired_dW)[i], (*dW)[i]) << "at position " << i;
}



/*!
 * Checks whether the forward is working for layer of input size 4x4x1 and with filter bank of 3 filters of size 1x1 with stride 3, double.
 * \author tkornuta
 */
TEST_F(Conv4x4x1Filter3x1x1s3Double, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<double> y = layer.forward(x);

	// Check output.
	for (size_t i=0; i<12; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the backward gradient pass is working for layer of input size 4x4x1 and with filter bank of 3 filters of size 1x1 with stride 3, double.
 * \author tkornuta
 */
TEST_F(Conv4x4x1Filter3x1x1s3Double, Backward) {

	// Backward pass - need to set x.
	layer.forward(x);
	mic::types::MatrixPtr<double> dx = layer.backward(dy);

	// Check resulting dx gradient.
	for (size_t i=0; i<16; i++)
		ASSERT_EQ((*desired_dx)[i], (*dx)[i]) << "at position " << i;

	// Check resulting db gradient.
	mic::types::MatrixPtr<double> db = layer.g["b"];
	for (size_t i=0; i<3; i++)
		ASSERT_EQ((*desired_db)[i], (*db)[i]);

	// Check resulting dW gradient.
	ASSERT_EQ((*desired_dW)[0], (*layer.g["W0x0"])[0]);
	ASSERT_EQ((*desired_dW)[1], (*layer.g["W1x0"])[0]);
	ASSERT_EQ((*desired_dW)[2], (*layer.g["W2x0"])[0]);
}

/*!
 * Checks whether dimensions of xs, outputs and filters are ok.
 * Convolutional dimensions are nicely explained in this lecture:
 * http://cs231n.github.io/convolutional-networks/
  * \author tkornuta
 */
TEST_F(Conv5x5x1Filter1x3x3s1Float, Dimensions) {

	// Check filter size - W.
	ASSERT_EQ((*layer.p["W0x0"]).rows(), 1);
	ASSERT_EQ((*layer.p["W0x0"]).cols(), 9);

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
	mic::types::MatrixPtr<float> y = layer.forward(x);

	// Check y.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the forward is working for layer of input size 5x5x1 and with filter bank of 1 filter of size 2x2 with stride 3 (float).
 * \author tkornuta
 */
TEST_F(Conv5x5x1Filter1x2x2s3Float, Forward) {

	// Forward pass.
	mic::types::MatrixPtr<float> y = layer.forward(x);

	// Check output.
	for (size_t i=0; i<4; i++)
		ASSERT_EQ((*y)[i], (*desired_y)[i]) << "at position " << i;

}


/*!
 * Checks whether the backward gradient pass is working for layer of input size 4x4x1 and with filter bank of 3 filters of size 1x1 with stride 3, double.
 * \author tkornuta
 */
TEST_F(Conv5x5x1Filter1x2x2s3Float, Backward) {

	// Backward pass - need to set x.
	layer.forward(x);
	mic::types::MatrixPtr<float> dx = layer.backward(dy);

	// Check resulting dx gradient.
	for (size_t i=0; i<25; i++)
		ASSERT_EQ((*desired_dx)[i], (*dx)[i]) << "at position " << i;

	// Check resulting db gradient.
	mic::types::MatrixPtr<float> db = layer.g["b"];
	for (size_t i=0; i<1; i++)
		ASSERT_EQ((*desired_db)[i], (*db)[i]);

	// Check resulting dW gradient.
	for (size_t i=0; i<4; i++)
	ASSERT_EQ((*desired_dW)[i], (*layer.g["W0x0"])[i]);
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
