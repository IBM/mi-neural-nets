/*!
 * @file BinaryCorrelator.hpp
 * @brief
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef SRC_MLNN_BINRARYCORRELATOR_HPP_
#define SRC_MLNN_BINRARYCORRELATOR_HPP_

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
class BinaryCorrelator : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates a "binary correlator" (i.e fully connected) layer - reduced number of parameters.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param name_ Name of the layer.
	 */
	BinaryCorrelator(size_t inputs_, size_t outputs_, eT permanence_threshold_ = 0.5, eT proximal_threshold_ = 0.5, std::string name_ = "BinaryCorrelator") :
		BinaryCorrelator(inputs_, 1, 1, outputs_, 1, 1, permanence_threshold_, proximal_threshold_, name_)
	{

	}

	/*!
	 * Creates a "binary correlator" (i.e fully connected) layer.
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param input_depth_ Depth of the input sample.
	 * @param output_height_ Width of the output sample.
	 * @param output_width_ Height of the output sample.
	 * @param output_depth_ Depth of the output sample.
	 * @param name_ Name of the layer.
	 */
	BinaryCorrelator(size_t input_height_, size_t input_width_, size_t depth_,
			size_t output_height_, size_t output_width_,
			eT permanence_threshold_ = 0.5, eT proximal_threshold_ = 0.5,
			std::string name_ = "BinaryCorrelator") :
		Layer<eT>::Layer(input_height_, input_width_, depth_,
				output_height_, output_width_, depth_,
				LayerTypes::BinaryCorrelator, name_),
				permanence_threshold(permanence_threshold_),
				proximal_threshold(proximal_threshold_)
	{
		// Create the permanence matrix.
		p.add ("p", Layer<eT>::outputSize(), Layer<eT>::inputSize());
		// Create "connectivity" matrix.
		m.add ("c", Layer<eT>::outputSize(), Layer<eT>::inputSize());

		// Initialize permanence matrix.
		//double range = sqrt(6.0 / double(inputs_ + outputs_));
		p['p']->rand(0, 1);

		// Initialize connectivity.
		mic::types::MatrixPtr<eT> c = m['c'];
		mic::types::MatrixPtr<eT> perm = p['p'];
		// Threshold.
		for (size_t i = 0; i < (size_t)c->size(); i++) {
			(*c)[i] = ((*perm)[i] > permanence_threshold) ? 1.0f : 0.0f;
		}//: for

		// Set hebbian learning as default optimization function.
		Layer<eT>::template setOptimization<mic::neural_nets::learning::HebbianRule<eT> > ();
	};


	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~BinaryCorrelator() {};

	/*!
	 * Forward pass.
	 * @param test_ It ise set to true in test mode (network verification).
	 */
	void forward(bool test_ = false) {
		// Get input matrices.
		mic::types::Matrix<eT> x = (*s['x']);
		mic::types::Matrix<eT> c = (*m['c']);
		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> y = s['y'];

		// Forward pass.
		(*y) = c * x;
		for (size_t i = 0; i < (size_t)y->size(); i++) {
			// Threshold.
			(*y)[i] = ((*y)[i] > proximal_threshold) ? 1.0f : 0.0f;
		}//: for
	}

	/*!
	 * Backward pass.
	 */
	void backward() {
		throw std::logic_error("Backward propagation should not be used with layers using Hebbian learning!");
	}

	/*!
	 * Applies the gradient update, using the selected hebbian rule.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_ = 0.0f) {
		//std::cout<<"p before update: " << (*p['p']) << std::endl;
		// Update permanence using the learning rule.
		opt["p"]->update(p['p'], s['x'], s['y'], alpha_);
		//std::cout<<"p after update: " << (*p['p']) << std::endl;

		// Update connectivity matrix.
		mic::types::MatrixPtr<eT> c = m['c'];
		mic::types::MatrixPtr<eT> perm = p['p'];
		//std::cout<<"C before threshold: " << (*c) << std::endl;
		// Threshold.
		for (size_t i = 0; i < (size_t)c->size(); i++) {
			(*c)[i] = ((*perm)[i] > permanence_threshold) ? 1.0f : 0.0f;
		}//: for
		//std::cout<<"C after threshold: " << (*c) << std::endl;
	}

	/*!
	 * Returns activations of neurons of a given layer (simple visualization).
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getActivations(size_t height_, size_t width_) {
		// Check if memory for the activations was allocated.
		if (neuron_activations.size() == 0) {
			for (size_t i=0; i < outputSize(); i++) {
				// Allocate memory for activation of every neuron.
				mic::types::MatrixPtr<eT> row = MAKE_MATRIX_PTR(eT, inputSize(), 1);
				neuron_activations.push_back(row);
			}//: for
		}//: if

		// Epsilon added for numerical stability.
		eT eps = 1e-10;

		mic::types::MatrixPtr<eT> perm =  p["p"];
		// Iterate through "neurons" and generate "activation image" for each one.
		for (size_t i=0; i < outputSize(); i++) {
			// Get row.
			mic::types::MatrixPtr<eT> row = neuron_activations[i];
			// Copy data.
			(*row) = perm->row(i);
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
    using Layer<eT>::inputSize;
    using Layer<eT>::outputSize;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;
	template<typename tmp> friend class HebbianNeuralNetwork;

	/// Vector containing activations of neurons.
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > neuron_activations;

	// Permanence threshold - used for calculation of binary connectivity.
	eT permanence_threshold;

	// Proximal threshold - used for activation of a given dendrite segment.
	eT proximal_threshold;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	BinaryCorrelator<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_BINRARYCORRELATOR_HPP_ */
