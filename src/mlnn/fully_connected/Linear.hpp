/*!
 * \file Linear.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_LINEAR_HPP_
#define SRC_MLNN_LINEAR_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {

// Forward declaration of SparseLinear class.
template <typename eT>
class SparseLinear;

/*!
 * \brief Class implementing a linear, fully connected layer.
 * \author tkornuta
  * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Linear : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates the linear (i.e fully connected) layer.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param name_ Name of the layer.
	 */
	Linear<eT>(size_t inputs_, size_t outputs_, std::string name_ = "Linear") :
		Layer<eT>(inputs_, outputs_, 1, LayerTypes::Linear, name_) {

		// Create the weights matrix.
		p.add ("W", outputs_, inputs_);

		// Create the bias vector.
		p.add ("b", outputs_, 1);

		// Initialize weights of the W matrix.
		eT range = sqrt(6.0 / eT(inputs_ + outputs_));

		Layer<eT>::p['W']->rand(-range, range);
		Layer<eT>::p['b']->setZero();

		// Add W and b gradients.
		Layer<eT>::g.add ("W", outputs_, inputs_);
		Layer<eT>::g.add ("b", outputs_, 1 );

		// Set gradient descent as default optimization function.
		Layer<eT>::template setOptimization<mic::neural_nets::optimization::GradientDescent<eT> > ();
	};


	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~Linear() {};

	/*!
	 * Forward pass.
	 * @param test_ It ise set to true in test mode (network verification).
	 */
	void forward(bool test_ = false) {
		// Get input matrices.
		mic::types::Matrix<eT> x = (*s['x']);
		mic::types::Matrix<eT> W = (*p['W']);
		mic::types::Matrix<eT> b = (*p['b']);
		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> y = s['y'];

		// Forward pass.
		(*y) = W * x + b.replicate(1, x.cols());

/*		std::cout << "Linear forward: s['x'] = \n" << (*s['x']) << std::endl;
		std::cout << "Linear forward: p['W'] = \n" << (*p['W']) << std::endl;
		std::cout << "Linear forward: p['b'] = \n" << (*p['b']) << std::endl;
		std::cout << "Linear forward: s['y'] = \n" << (*s['y']) << std::endl;*/

	}

	/*!
	 * Backward pass.
	 */
	void backward() {
		// Get matrices.
		mic::types::Matrix<eT> dy = (*g['y']);
		mic::types::Matrix<eT> x = (*s['x']);
		mic::types::Matrix<eT> W = (*p['W']);
		// Get output pointers - so the results will be stored!
		mic::types::MatrixPtr<eT> dW = g['W'];
		mic::types::MatrixPtr<eT> db = g['b'];
		mic::types::MatrixPtr<eT> dx = g['x'];

		// Backward pass.
		(*dW) = dy * x.transpose();
		(*db) = dy.rowwise().sum(); // Sum for all samples in batch, similarly as it is done for dW.
		(*dx) = W.transpose() * dy;

/*		std::cout << "Linear backward: g['y'] = \n" << (*g['y']) << std::endl;
		std::cout << "Linear backward: g['x'] = \n" << (*g['x']) << std::endl;*/
	}

	/*!
	 * Resets the gradients for W and b.
	 */
	void resetGrads() {
		g['W']->setZero();
		g['b']->setZero();
	}


	/*!
	 * Applies the gradient update, using the selected optimization method.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_  = 0.0f) {
		//std::cout << "p['W'] = \n" << (*p['W']) << std::endl;
		//std::cout << "g['W'] = \n" << (*g['W']) << std::endl;

		opt["W"]->update(p['W'], g['W'], alpha_, decay_);
		opt["b"]->update(p['b'], g['b'], alpha_, 0.0);

		//std::cout << "p['W'] after update= \n" << (*p['W']) << std::endl;
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
    using Layer<eT>::g;
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

	// Friend class - required for accessing private constructor.
	template<typename tmp> friend class SparseLinear;

	/// Vector containing activations of neurons.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > neuron_activations;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Linear<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_LINEAR_HPP_ */
