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

#include <mlnn/fully_connected/Linear.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {

/*!
 * \brief Class implementing a linear, fully connected layer with sparsity regulation.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class SparseLinear : public mic::mlnn::fully_connected::Linear<eT> {
public:
	/*!
	 * Default constructor used for creation of the layer.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param batch_size_ Size of the batch.
	 * @param name_ Name of the layer.
	 */
	SparseLinear<eT>(size_t inputs_, size_t outputs_, std::string name_ = "SparseLinear") :
		Linear<eT>(inputs_, outputs_, name_) {

		// Change type to SparseLinear.
		Layer<eT>::layer_type = LayerTypes::SparseLinear;

		// Prepare matrices in the "temporal memory".
		// For current sparsity vector.
		m.add ("ro", outputSize(), 1 );
		// For penalty.
		m.add ("penalty", outputSize(), 1 );

		// Set desired sparsity and penalty term.
		desired_ro = 0.1; // 10 %
		beta = 0.5;
	};


	/*!
	 * Virtual constructor.
	 */
	virtual ~SparseLinear() {};

	/*!
	 * Backward pass.
	 */
	void backward() {
		eT eps = 1e-10;
		// Calculate the current "activation sparsity".
		mic::types::MatrixPtr<eT> ro = m["ro"];
		(*ro) = ((*s['y']).rowwise().sum()/batch_size);

		// Calculate the sparsity penalty - for every output neuron.
		mic::types::MatrixPtr<eT> penalty = m["penalty"];
		for (size_t i=0; i<outputSize(); i++)
			(*penalty)[i] = beta*(-desired_ro/((*ro)[i] + eps) + (1-desired_ro)/(1-(*ro)[i] + eps));


		// Calculate derivatives of W,b and x.
		(*g['W']) = (*g['y']) * ((*s['x']).transpose());
		(*g['b']) = (*g['y']).rowwise().mean();
		(*g['x']) = (*p['W']).transpose() * (*g['y']);
	}

	/*!
	 * Applies the gradient update, using the selected optimization method.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_  = 0.0f) {
		//std::cout << "p['W'] = \n" << (*p['W']) << std::endl;
		//std::cout << "g['W'] = \n" << (*g['W']) << std::endl;

		// Apply selected learning rule to W.
		opt["W"]->update(p['W'], g['W'], alpha_, decay_);

		// Apply sparsity learning rule to b, incorporating the KL-divergence term.
		mic::types::MatrixPtr<eT> penalty = m["penalty"];
		// (*p['b']) -=  alpha_ * beta * (*penalty);
		opt["b"]->update(p['b'], g['b'], alpha_, 0.0);

		//std::cout << "p['W'] after update= \n" << (*p['W']) << std::endl;
	}

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
    using Layer<eT>::inputSize;
    using Layer<eT>::outputSize;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

    /*!
	 * Private constructor, used only during the deserialization.
	 */
	SparseLinear<eT>() : mic::mlnn::fully_connected::Linear<eT> () { }

	/// Desired sparsity of the layer.
	eT desired_ro;

	/// Controls the weight of the sparsity penalty term.
	eT beta;
};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SPARSELINEAR_H_ */
