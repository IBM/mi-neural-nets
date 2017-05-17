/*!
 * @file HebbianLinear.hpp
 * @brief
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef SRC_MLNN_HEBBIANLINEAR_HPP_
#define SRC_MLNN_HEBBIANLINEAR_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {

/*!
 * \brief Class implementing a linear, fully connected layer.
 * \author tkornuta
  * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class HebbianLinear : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates the linear (i.e fully connected) layer.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param name_ Name of the layer.
	 */
	HebbianLinear<eT>(size_t inputs_, size_t outputs_, std::string name_ = "HebbianLinear") :
		Layer<eT>(inputs_, outputs_, 1, LayerTypes::HebbianLinear, name_) {

		// Create the weights matrix.
		p.add ("W", outputs_, inputs_);

		// Initialize weights of the W matrix.
		double range = sqrt(6.0 / double(inputs_ + outputs_));
		Layer<eT>::p['W']->rand(-range, range);

		// Set hebbian learning as default optimization function.
		Layer<eT>::template setOptimization<mic::neural_nets::learning::HebbianRule<eT> > ();
	};


	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~HebbianLinear() {};

	/*!
	 * Forward pass.
	 * @param test_ It ise set to true in test mode (network verification).
	 */
	void forward(bool test_ = false) {
		// Get input matrices.
		mic::types::Matrix<eT> x = (*s['x']);
		mic::types::Matrix<eT> W = (*p['W']);
		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> y = s['y'];

		// Forward pass.
		(*y) = W * x;
	}

	/*!
	 * Backward pass.
	 */
	void backward() {
		//LOG(LERROR) << "Backward propagation should not be used with layers using Hebbian learning!";
	}

	/*!
	 * Applies the gradient update, using the selected hebbian rule.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 */
	void update(eT alpha_) {
		opt["W"]->update(p['W'], s['x'], s['y'], alpha_);
	}

	/*!
	 * Returns activations of neurons of a given layer (simple visualization).
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getActivations(size_t height_, size_t width_) {
		// Check if memory for the activations was allocated.
		if (neuron_activations.size() == 0) {
			for (size_t i=0; i < output_size; i++) {
				// Allocate memory for activation of every neuron.
				mic::types::MatrixPtr<eT> row = MAKE_MATRIX_PTR(eT, input_size, 1);
				neuron_activations.push_back(row);
			}//: for
		}//: if

		// Epsilon added for numerical stability.
		eT eps = 1e-10;

		mic::types::MatrixPtr<eT> W =  p["W"];
		// Iterate through "neurons" and generate "activation image" for each one.
		for (size_t i=0; i < output_size; i++) {
			// Get row.
			mic::types::MatrixPtr<eT> row = neuron_activations[i];
			// Copy data.
			(*row) = W->row(i);
			// Resize row.
			row->resize( height_, width_);
			// Calculate l2 norm.
			eT l2 = row->norm() + eps;
			// Normalize the inputs to <-0.5,0.5> and add 0.5f -> range <0.0, 1.0>.
			(*row) = row->unaryExpr ( [&] ( eT x ) { return ( x / l2 + 0.5f); } );
		}//: for

		// Return activations.
		return neuron_activations;
	}


	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
    using Layer<eT>::input_size;
    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

	/// Vector containing activations of neurons.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > neuron_activations;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	HebbianLinear<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_HEBBIANLINEAR_HPP_ */
