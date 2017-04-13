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

/*!
 * \brief Class implementing a linear, fully connected layer.
 * \author krocki/tkornuta
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
		Layer<eT>::p.add ("W", outputs_, inputs_);

		// Create the bias vector.
		Layer<eT>::p.add ("b", outputs_, 1);

		// Initialize weights of the W matrix.
		double range = sqrt(6.0 / double(inputs_ + outputs_));

		Layer<eT>::p['W']->rand(-range, range);
		Layer<eT>::p['b']->setZero();

		//mW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
		//Layer<eT>::m.add ("W", outputs_, inputs_);


		//mb = mic::types::VectorXf::Zero(b.rows());
		//Layer<eT>::m.add ("b", outputs_, 1);

		// Add W and b gradients.
		Layer<eT>::g.add ("W", outputs_, inputs_);
		Layer<eT>::g.add ("b", outputs_, 1 );

		// Create optimization functions.
		opt_W = std::make_shared<mic::neural_nets::optimization::GradientDescent<eT> > (mic::neural_nets::optimization::GradientDescent<eT> (outputs_, inputs_));
		opt_b = std::make_shared<mic::neural_nets::optimization::GradientDescent<eT> > (mic::neural_nets::optimization::GradientDescent<eT> (outputs_, 1));
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
		mic::types::Matrix<eT> x = (*Layer<eT>::s['x']);
		mic::types::Matrix<eT> W = (*Layer<eT>::p['W']);
		mic::types::Matrix<eT> b = (*Layer<eT>::p['b']);
		// Get output pointer!
		mic::types::MatrixPtr<eT> y = Layer<eT>::s['y'];

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
		mic::types::Matrix<eT> dy = (*Layer<eT>::g['y']);
		mic::types::Matrix<eT> x = (*Layer<eT>::s['x']);
		mic::types::Matrix<eT> W = (*Layer<eT>::p['W']);
		// Get output pointers!
		mic::types::MatrixPtr<eT> dW = Layer<eT>::g['W'];
		mic::types::MatrixPtr<eT> db = Layer<eT>::g['b'];
		mic::types::MatrixPtr<eT> dx = Layer<eT>::g['x'];

		// Backward pass.
		(*dW) = dy * x.transpose();
		(*db) = dy.rowwise().sum();
		(*dx) = W.transpose() * dy;

/*		std::cout << "Linear backward: g['y'] = \n" << (*g['y']) << std::endl;
		std::cout << "Linear backward: g['x'] = \n" << (*g['x']) << std::endl;*/
	}

	/*!
	 * Resets the gradients - empty, to be overridden by the inherited classes.
	 */
	void resetGrads() {
		(*Layer<eT>::g['W']).setZero();
		(*Layer<eT>::g['b']).setZero();
	}


	/*!
	 * Applies the gradient update.
	 */
	void applyGrads(double alpha_) {
		//adagrad
		//mW += dW.cwiseProduct(dW);
		//(*m['W']) += (*g['W']).cwiseProduct((*g['W']));

		//mb += db.cwiseProduct(db);
		//(*m['b']) += (*g['b']).cwiseProduct((*g['b']));

		//W = (1 - decay_) * W + alpha_ * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
		//(*p['W']) = (1.0f - decay_) * (*p['W']) + alpha_ * (*g['W']).cwiseQuotient((*m['W']).unaryExpr(std::ptr_fun<eT>(sqrt_eps)));
		/*std::cout << "p['W'] = \n" << (*p['W']) << std::endl;
		std::cout << "g['W'] = \n" << (*g['W']) << std::endl;*/

		opt_W->update(p['W'], g['W'], alpha_);
		opt_b->update(p['b'], g['b'], alpha_);

		//std::cout << "p['W'] after update= \n" << (*p['W']) << std::endl;

		//b += alpha_ * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));
		//(*p['b']) += alpha_ * (*g['b']).cwiseQuotient((*m['b']).unaryExpr(std::ptr_fun<eT>(sqrt_eps)));

		// 'plain' fixed learning rate update
		// b += alpha * db;
		// W += alpha * dW;
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

private:
	// Friend class - required for using boost serialization.
	template<typename tmp1, typename tmp2> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Linear<eT>() : Layer<eT> () { }

	/*!
	 * Optimization function for W.
	 */
	std::shared_ptr<mic::neural_nets::optimization::OptimizationFunction<eT> > opt_W;

	/*!
	 * Optimization function for b.
	 */
	std::shared_ptr<mic::neural_nets::optimization::OptimizationFunction<eT> > opt_b;

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_LINEAR_HPP_ */
