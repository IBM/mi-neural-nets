/*!
 * @file: MultiLayerNeuralNetworkTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 10, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#include "MultiLayerNeuralNetworkTests.hpp"



/*!
 * Tests the dimensionality of nn.
 */
TEST_F(Simple2LayerRegressionNN, Dimensions) {

	// Number of layers.
	ASSERT_EQ(nn.layers.size(), 2);
	// Size of nn input.

	// Size of nn output (prediction).
}


/*!
 * Tests squared error loss function on vectors with four floats.
 */
TEST_F(Simple2LayerRegressionNN, Serialization) {


	// Save network to file.
	const char* fileName = "saved.txt";
	nn.save(fileName);
	std::cout << "Saved network: \n" << nn;


	// Load network from file.
	mic::mlnn::MultiLayerNeuralNetwork restored_nn("simple_linear_network_loaded");
	restored_nn.load(fileName);
	std::cout << "Restored network: \n" << restored_nn;

	// Compare networks.

//	ASSERT_EQ(loss.calculateLoss(predicted_y, target_y), (float)4.0);
}






int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
