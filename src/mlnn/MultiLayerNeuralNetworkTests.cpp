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
	ASSERT_EQ(nn.layers.size(), 4);
	// First fully connected dimensions.
	ASSERT_EQ(nn.layers[0]->inputSize(), 10);
	ASSERT_EQ(nn.layers[0]->batchSize(), 1);
	ASSERT_EQ(nn.layers[0]->outputSize(), 20);

	// Second fully connected dimensions.
	ASSERT_EQ(nn.layers[2]->inputSize(), 20);
	ASSERT_EQ(nn.layers[2]->batchSize(), 1);
	ASSERT_EQ(nn.layers[2]->outputSize(), 4);

}

/*!
 * Tests the batch resizing.
 */
TEST_F(Simple2LayerRegressionNN, BatchResize) {

	// Check original batch size.
	for (size_t i=0; i< nn.layers.size(); i++)
		ASSERT_EQ(nn.layers[0]->batchSize(), 1);

	// Resize.
	nn.resizeBatch(5);
	// Check new size.
	for (size_t i=0; i< nn.layers.size(); i++)
		ASSERT_EQ(nn.layers[0]->batchSize(), 5);
}


/*!
 * Tests squared error loss function on vectors with four floats.
 */
TEST_F(Simple2LayerRegressionNN, Serialization) {


	// Save network to file.
	const char* fileName = "saved.txt";
	nn.save(fileName);
	//std::cout << "Saved network: \n" << nn;


	// Load network from file.
	mic::mlnn::MultiLayerNeuralNetwork<double> restored_nn("simple_linear_network_loaded");
	restored_nn.load(fileName);
	//std::cout << "Restored network: \n" << restored_nn;

	// Compare number of layers.
	ASSERT_EQ(nn.layers.size(), restored_nn.layers.size());

	// Compare layers.
	for (size_t i=0; i< nn.layers.size(); i++) {
		// Compare sizes and types.
		ASSERT_EQ(nn.layers[i]->batchSize(), restored_nn.layers[i]->batchSize());
		ASSERT_EQ(nn.layers[i]->inputSize(), restored_nn.layers[i]->inputSize());
		ASSERT_EQ(nn.layers[i]->outputSize(), restored_nn.layers[i]->outputSize());
		ASSERT_EQ(nn.layers[i]->layer_type, restored_nn.layers[i]->layer_type);
	}//: for

}






int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
