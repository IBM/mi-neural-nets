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
	 * Creates a linear (i.e fully connected) layer - reduced number of parameters.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param name_ Name of the layer.
	 */
	Linear(size_t inputs_, size_t outputs_, std::string name_ = "Linear") :
		Linear(inputs_, 1, 1, outputs_, 1, 1, name_)
	{
		std::cout<<"constructor Linear 1!\n";
	}


	/*!
	 * Creates a linear (i.e fully connected) layer.
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param input_depth_ Depth of the input sample.
	 * @param output_height_ Width of the output sample.
	 * @param output_width_ Height of the output sample.
	 * @param output_depth_ Depth of the output sample.
	 * @param name_ Name of the layer.
	 */
	Linear(size_t input_height_, size_t input_width_, size_t input_depth_, size_t output_height_, size_t output_width_, size_t output_depth, std::string name_ = "Linear") :
		Layer<eT>::Layer(input_height_, input_width_, input_depth_,
				output_height_, output_width_, output_depth,
				LayerTypes::Convolution, name_)
	{
		std::cout<<"constructor Linear 2!\n";

		// Create the weights matrix.
		p.add ("W", Layer<eT>::outputSize(), Layer<eT>::inputSize());

		// Create the bias vector.
		p.add ("b", Layer<eT>::outputSize(), 1);

		// Initialize weights of the W matrix.
		eT range = sqrt(6.0 / eT(Layer<eT>::inputSize() + Layer<eT>::outputSize()));

		Layer<eT>::p['W']->rand(-range, range);
		Layer<eT>::p['b']->setZero();

		// Add W and b gradients.
		Layer<eT>::g.add ("W", Layer<eT>::outputSize(), Layer<eT>::inputSize());
		Layer<eT>::g.add ("b", Layer<eT>::outputSize(), 1 );

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
	 * Returns activations of weights.
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getWeightActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(w_activations, 1, Layer<eT>::outputSize()*Layer<eT>::inputSize(), 1);

		// Get matrix of a given "part of a given neuron".
		mic::types::MatrixPtr<eT> W = p["W"];

		// Get row.
		mic::types::MatrixPtr<eT> row = w_activations[0];
		// Copy data.
		(*row) = (*W);
		row->resize(Layer<eT>::outputSize(), Layer<eT>::inputSize());

		// Normalize.
		if (normalize_ )
			normalizeMatrixForVisualization(row);

		// Return activations.
		return w_activations;
	}



	/*!
	 * Returns activations of weight gradients (dx).
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getWeightGradientActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(dw_activations, 1, Layer<eT>::outputSize()*Layer<eT>::inputSize(), 1);

		// Get matrix of a given "part of a given neuron".
		mic::types::MatrixPtr<eT> dW = g["W"];

		// Get row.
		mic::types::MatrixPtr<eT> row = dw_activations[0];
		// Copy data.
		(*row) = (*dW);
		row->resize(Layer<eT>::outputSize(), Layer<eT>::inputSize());

		// Normalize.
		if (normalize_ )
			normalizeMatrixForVisualization(row);

		// Return activations.
		return dw_activations;
	}



	/*!
	 * Returns activations of neurons of a given layer (simple visualization).
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getInverseActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(inverse_activations, Layer<eT>::outputSize(), input_height*input_width, 1);

		// TODO: check different input-output depths.

		mic::types::MatrixPtr<eT> W =  p["W"];
		// Iterate through "neurons" and generate "activation image" for each one.
		for (size_t i=0; i < output_height*output_width; i++) {

			for (size_t j=0; j < input_depth; j++) {
				// "Access" activation row.
				mic::types::MatrixPtr<eT> row = inverse_activations[i*input_depth + j];
				// Copy data.
				(*row) = W->row(i).block(input_height*input_width, 1);
				// Resize row.
				row->resize( input_height, input_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for
		}//: for

		// Return activations.
		return inverse_activations;
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
    using Layer<eT>::opt;

    // Uncover "sizes" for visualization.
    using Layer<eT>::input_height;
    using Layer<eT>::input_width;
    using Layer<eT>::input_depth;
	using Layer<eT>::output_height;
	using Layer<eT>::output_width;
	using Layer<eT>::output_depth;
    using Layer<eT>::batch_size;

	 // Uncover methods useful in visualization.
	 using Layer<eT>::lazyAllocateMatrixVector;
	 using Layer<eT>::normalizeMatrixForVisualization;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

	// Friend class - required for accessing private constructor.
	template<typename tmp> friend class SparseLinear;

	/// Vector containing activations of weights/filters.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > w_activations;

	/// Vector containing activations of gradients of weights (dW).
	std::vector< std::shared_ptr <mic::types::MatrixXf> > dw_activations;

	/// Vector containing activations of neurons (y*W^T).
	std::vector< std::shared_ptr <mic::types::MatrixXf> > inverse_activations;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Linear<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_LINEAR_HPP_ */
