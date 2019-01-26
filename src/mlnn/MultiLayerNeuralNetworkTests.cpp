/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * @file: MultiLayerNeuralNetworkTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 10, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#include <mlnn/MultiLayerNeuralNetworkTests.hpp>

namespace mic { namespace neural_nets { namespace unit_tests {

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
	std::cout << "Saved network: \n" << nn;

	// Load network from file.
	mic::mlnn::MultiLayerNeuralNetwork<double> restored_nn("simple_linear_network_loaded");
	restored_nn.load(fileName);
	std::cout << "Restored network: \n" << restored_nn;

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

	// TODO: Check loss function.
	// TODO: Check optimization function.
}


/*!
 * Tests a single iteration of a backpropagation algorithm.
 */
TEST_F(Tutorial2LayerNN, BackpropagationSingleStep) {
	double eps = 1e-5;

	// Forward pass.
	nn.forward(input_x);

	// Lin1 layer output.
	ASSERT_LE( fabs( (*nn.layers[0]->s["y"])[0] - (*ffpass1_lin1_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->s["y"])[1] - (*ffpass1_lin1_y)[1]), eps);
	// Sig1 layer output.
	ASSERT_LE( fabs( (*nn.layers[1]->s["y"])[0] - (*ffpass1_sig1_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[1]->s["y"])[1] - (*ffpass1_sig1_y)[1]), eps);
	// Lin1 layer output.
	ASSERT_LE( fabs( (*nn.layers[2]->s["y"])[0] - (*ffpass1_lin2_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->s["y"])[1] - (*ffpass1_lin2_y)[1]), eps);
	// Sig1 layer output.
	ASSERT_LE( fabs( (*nn.layers[3]->s["y"])[0] - (*ffpass1_sig2_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[3]->s["y"])[1] - (*ffpass1_sig2_y)[1]), eps);

	// Calculate loss.
	double loss = nn.loss->calculateLoss(target_y, nn.getPredictions());
	ASSERT_LE( fabs( loss - ffpass1_loss), eps);

	// Calculate gradient.
	mic::types::MatrixPtr<double> dy = MAKE_MATRIX_PTR(double, 2, 1);
	(*dy) = (*nn.loss->calculateGradient(target_y, nn.getPredictions()));

	// Check gradient.
	ASSERT_LE( fabs( (*dy)[0] - (*ffpass1_dy)[0]), eps);
	ASSERT_LE( fabs( (*dy)[1] - (*ffpass1_dy)[1]), eps);

	// Backpropagate the gradients from last layer to the first.
	nn.backward(dy);

	// Check weight gradients.
	ASSERT_LE( fabs( (*nn.layers[2]->g["W"])[0] - (*bwpass1_lin2_dW)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->g["W"])[1] - (*bwpass1_lin2_dW)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->g["W"])[2] - (*bwpass1_lin2_dW)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->g["W"])[3] - (*bwpass1_lin2_dW)[3]), eps);

	ASSERT_LE( fabs( (*nn.layers[0]->g["W"])[0] - (*bwpass1_lin1_dW)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->g["W"])[1] - (*bwpass1_lin1_dW)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->g["W"])[2] - (*bwpass1_lin1_dW)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->g["W"])[3] - (*bwpass1_lin1_dW)[3]), eps);

	// Apply changes.
	nn.update(0.5);

	// Check weight gradients after the update.
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[0] - (*bwpass1_lin2_pW_updated)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[1] - (*bwpass1_lin2_pW_updated)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[2] - (*bwpass1_lin2_pW_updated)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[3] - (*bwpass1_lin2_pW_updated)[3]), eps);

	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[0] - (*bwpass1_lin1_pW_updated)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[1] - (*bwpass1_lin1_pW_updated)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[2] - (*bwpass1_lin1_pW_updated)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[3] - (*bwpass1_lin1_pW_updated)[3]), eps);

}


/*!
 * Tests a single iteration of a backpropagation algorithm.
 */
TEST_F(Tutorial2LayerNN, TrainSingleStep) {
	double eps = 1e-5;

	// Perform a single training step.
	double loss = nn.train(input_x, target_y, 0.5);

	// Check loss
	ASSERT_LE( fabs( loss - ffpass1_loss), eps);

	// Lin1 layer output.
	ASSERT_LE( fabs( (*nn.layers[0]->s["y"])[0] - (*ffpass1_lin1_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->s["y"])[1] - (*ffpass1_lin1_y)[1]), eps);
	// Sig1 layer output.
	ASSERT_LE( fabs( (*nn.layers[1]->s["y"])[0] - (*ffpass1_sig1_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[1]->s["y"])[1] - (*ffpass1_sig1_y)[1]), eps);
	// Lin1 layer output.
	ASSERT_LE( fabs( (*nn.layers[2]->s["y"])[0] - (*ffpass1_lin2_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->s["y"])[1] - (*ffpass1_lin2_y)[1]), eps);
	// Sig1 layer output.
	ASSERT_LE( fabs( (*nn.layers[3]->s["y"])[0] - (*ffpass1_sig2_y)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[3]->s["y"])[1] - (*ffpass1_sig2_y)[1]), eps);

	// Check weight gradients after the update.
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[0] - (*bwpass1_lin2_pW_updated)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[1] - (*bwpass1_lin2_pW_updated)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[2] - (*bwpass1_lin2_pW_updated)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[2]->p["W"])[3] - (*bwpass1_lin2_pW_updated)[3]), eps);

	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[0] - (*bwpass1_lin1_pW_updated)[0]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[1] - (*bwpass1_lin1_pW_updated)[1]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[2] - (*bwpass1_lin1_pW_updated)[2]), eps);
	ASSERT_LE( fabs( (*nn.layers[0]->p["W"])[3] - (*bwpass1_lin1_pW_updated)[3]), eps);

}

} } }//: namespaces

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
