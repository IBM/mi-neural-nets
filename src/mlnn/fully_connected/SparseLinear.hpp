/*!
 * @file: SparseLinear.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   05-04-2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef SPARSELINEAR_H_
#define SPARSELINEAR_H_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {

/*!
 * \brief Class implementing a linear, fully connected layer with sparsity regulation.
 * \author tkornuta
 */
class SparseLinear : public mic::mlnn::Layer {
public:
	/*!
	 * Default constructor used for creation of the layer.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param batch_size_ Size of the batch.
	 * @param name_ Name of the layer.
	 */
	SparseLinear(size_t inputs_, size_t outputs_, size_t batch_size_, std::string name_ = "SparseLinear");

	/*!
	 * Virtual constructor.
	 */
	virtual ~SparseLinear() {};

	/*!
	 * Forward pass.
	 */
	void forward(bool test_ = false);

	/*!
	 * Backward pass.
	 */
	void backward();

	/*!
	 * Reset the gradients.
	 */
	void resetGrads();

	/*!
	 * Apply the gradient update.
	 */
	void applyGrads(double alpha_, double decay_ = 0);

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the deserialization.
	 */
	SparseLinear() : Layer () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SPARSELINEAR_H_ */
