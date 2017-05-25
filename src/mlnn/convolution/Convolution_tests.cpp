/*!
 * @file: Linear_tests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 4, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "Convolution_tests.hpp"

TEST(Convolutions, NumberFfFilters) {

	// Stride = 1.
	mic::mlnn::convolution::Convolution<float> l1(5,7,1,3,1,1);

	ASSERT_EQ(l1.number_of_receptive_fields_vertical, 3);
	ASSERT_EQ(l1.number_of_receptive_fields_horizontal, 5);

	// Stride = 2.
	mic::mlnn::convolution::Convolution<float> l2(5,7,1,3,2,1);

	ASSERT_EQ(l2.number_of_receptive_fields_vertical, 2);
	ASSERT_EQ(l2.number_of_receptive_fields_horizontal, 3);

}



/*!
 * Checks whether the forward is working.
 */
TEST_F(Conv5x5x1Filter3x3x1Float, LayerDimensions) {

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
 * Checks whether the forward is working.
 */
TEST_F(Conv5x5x1Filter3x3x1Float, Forward) {

	std::cout<<"W = \n" << (*layer.p["W0"]) <<std::endl;
	std::cout<<"input = \n" << (*input) <<std::endl;

	// Forward pass.
	mic::types::MatrixPtr<float> output = layer.forward(input);

	// Check output.
	for (size_t i=0; i<9; i++)
		ASSERT_EQ((*output)[i], (*desired_output)[i]);

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
