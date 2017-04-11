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
#include "MultiLayerNeuralNetwork.hpp"


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
		nn.pushLayer(new mic::mlnn::cost_function::Regression<double>(4, "Final Regression"));

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
	mic::mlnn::MultiLayerNeuralNetwork<double> nn;
};


#endif /* MULTILAYERNEURALNETSTESTS_HPP_ */
