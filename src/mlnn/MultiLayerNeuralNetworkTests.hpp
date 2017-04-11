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
		nn.pushLayer(new mic::mlnn::Linear(10, 20, "First Linear"));
		nn.pushLayer(new mic::mlnn::ReLU(20, "First ReLU"));
		nn.pushLayer(new mic::mlnn::Linear(20, 4, "Second Linear"));
		nn.pushLayer(new mic::mlnn::ReLU(4, "Second ReLU"));
		nn.pushLayer(new mic::mlnn::Regression(4, "Final Regression"));

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
	mic::mlnn::MultiLayerNeuralNetwork nn;
};


#endif /* MULTILAYERNEURALNETSTESTS_HPP_ */
