/*!
 * @file: MultiLayerNeuralNetsTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 10, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef MULTILAYERNEURALNETSTESTS_HPP_
#define MULTILAYERNEURALNETSTESTS_HPP_


#include <gtest/gtest.h>

// Redefine "private" and "protected" so every class field/method will be accessible for tests.
#define private public
#define protected public
#include <mlnn/BackpropagationNeuralNetwork.hpp>


namespace mic { namespace neural_nets { namespace unit_tests {


/*!
 * \brief Test Fixture - simple ff net with 2 layers.
 * \author tkornuta
 */
class Simple2LayerRegressionNN : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Simple2LayerRegressionNN () :
		nn("simple_linear_network")
	{
		// Add
		nn.pushLayer(new mic::mlnn::fully_connected::Linear<double>(10, 20, "First Linear"));
		nn.pushLayer(new mic::mlnn::activation_function::ReLU<double>(20, "First ReLU"));
		nn.pushLayer(new mic::mlnn::fully_connected::Linear<double>(20, 4, "Second Linear"));
		nn.pushLayer(new mic::mlnn::activation_function::ReLU<double>(4, "Second ReLU"));
		nn.setLoss< mic::neural_nets::loss::SquaredErrorLoss<double> >();
	}

protected:
	virtual void SetUp() {
		// Reset the gradients.
		nn.resetGrads();
		// Set batch size to 1.
		nn.resizeBatch(1);
	}

private:
	// Neural network.
	mic::mlnn::BackpropagationNeuralNetwork<double> nn;
};



/*!
 * \brief Test Fixture - feed-forward net with 2 layers.
 * A "formalized" example from a step-by-step tutorial:
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 * \author tkornuta
 */
class Tutorial2LayerNN : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Tutorial2LayerNN () :
		nn("simple_linear_network")
	{
		// Add
		nn.pushLayer(new mic::mlnn::fully_connected::Linear<double>(2, 2, "Linear1"));
		nn.pushLayer(new mic::mlnn::activation_function::Sigmoid<double>(2, "Sigmoid1"));
		nn.pushLayer(new mic::mlnn::fully_connected::Linear<double>(2, 2, "Linear2"));
		nn.pushLayer(new mic::mlnn::activation_function::Sigmoid<double>(2, "Sigmoid2"));
		nn.setLoss< mic::neural_nets::loss::SquaredErrorLoss<double> >();

		input_x = MAKE_MATRIX_PTR(double, 2, 1);
		target_y = MAKE_MATRIX_PTR(double, 2, 1);
		ffpass1_lin1_y = MAKE_MATRIX_PTR(double, 2, 1);
		ffpass1_sig1_y = MAKE_MATRIX_PTR(double, 2, 1);
		ffpass1_lin2_y = MAKE_MATRIX_PTR(double, 2, 1);
		ffpass1_sig2_y = MAKE_MATRIX_PTR(double, 2, 1);
		ffpass1_dy = MAKE_MATRIX_PTR(double, 2, 1);
		bwpass1_lin2_dW = MAKE_MATRIX_PTR(double, 2, 2);
		bwpass1_lin2_pW_updated = MAKE_MATRIX_PTR(double, 2, 2);
		bwpass1_lin1_dW = MAKE_MATRIX_PTR(double, 2, 2);
		bwpass1_lin1_pW_updated = MAKE_MATRIX_PTR(double, 2, 2);
}

protected:
	virtual void SetUp() {
		// Reset the gradients.
		nn.resetGrads();
		// Set batch size to 1.
		nn.resizeBatch(1);

		// Set weights and biases.
		(*nn.layers[0]->p["W"]) << 0.15, .20, .25, .30;
		(*nn.layers[0]->p["b"]) << .35, 0.35;

		(*nn.layers[2]->p["W"]) << 0.4, .45, .5, .55;
		(*nn.layers[2]->p["b"]) << .6, 0.6;

		(*input_x) << 0.05, 0.1;
		(*target_y) << 0.01, 0.99;

		// Values after the first forward pass.
		// Linear1
		(*ffpass1_lin1_y) << 0.3775, 0.3925;
		// Sigmoid1
		(*ffpass1_sig1_y) << 0.59327, 0.596884;
		// Linear2
		(*ffpass1_lin2_y) << 1.10591, 1.22492;
		// Sigmoid2
		(*ffpass1_sig2_y) << 0.751365, 0.772928;

		// Loss and gradient.
		ffpass1_loss = 0.298371;
		(*ffpass1_dy) <<  0.741365, -0.217072;

		// backward pass - weights gradients.
		(*bwpass1_lin2_dW) << 0.082167, 0.0826676, -0.0226025, -0.0227402;
		(*bwpass1_lin1_dW) << 0.000438568, 0.000877135, 0.000497713, 0.000995425;

		// Weights after update with  gradient descent learning rule with learning rate = 0.5 (in this case: delta rule).
		(*bwpass1_lin2_pW_updated) << 0.358916, 0.408666, 0.511301, 0.56137;
		(*bwpass1_lin1_pW_updated) << 0.149781, 0.199561, 0.249751, 0.299502;
	}


private:
	// Neural network.
	mic::mlnn::BackpropagationNeuralNetwork<double> nn;

	// Test input x - used in forward pass.
	mic::types::MatrixPtr<double> input_x;

	// Target y values.
	mic::types::MatrixPtr<double> target_y;

	//  Results of the first forward pass.
	mic::types::MatrixPtr<double> ffpass1_lin1_y, ffpass1_sig1_y, ffpass1_lin2_y, ffpass1_sig2_y;

	// Loss and gradient.
	double ffpass1_loss;
	mic::types::MatrixPtr<double> ffpass1_dy;

	// Backward pass.
	mic::types::MatrixPtr<double> bwpass1_lin2_dW;
	mic::types::MatrixPtr<double> bwpass1_lin2_pW_updated;
	mic::types::MatrixPtr<double> bwpass1_lin1_dW;
	mic::types::MatrixPtr<double> bwpass1_lin1_pW_updated;
};

} } }//: namespaces

#endif /* MULTILAYERNEURALNETSTESTS_HPP_ */
