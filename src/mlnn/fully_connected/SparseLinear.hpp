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

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {

/*!
 * \brief Class implementing a linear, fully connected layer with sparsity regulation.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class SparseLinear : public mic::mlnn::Layer<eT> {
public:
	/*!
	 * Default constructor used for creation of the layer.
	 * @param inputs_ Length of the input vector.
	 * @param outputs_ Length of the output vector.
	 * @param batch_size_ Size of the batch.
	 * @param name_ Name of the layer.
	 */
	SparseLinear<eT>(size_t inputs_, size_t outputs_, size_t batch_size_ = 1, std::string name_ = "SparseLinear") :
		Layer<eT>(inputs_, outputs_, batch_size_, LayerTypes::SparseLinear, name_) {

		// Allocate parameters: weight and bias.
		p.add (std::make_tuple ( "W", outputs_, inputs_ ));
		p.add (std::make_tuple ( "b", outputs_, 1 ));

		// Initialize weights of the W matrix.
		double range = sqrt(6.0 / double(inputs_ + outputs_));
		p['W']->rand(-range, range);
		p['b']->rand(-range, range);// setZero();

		//mW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
		m.add (std::make_tuple ( "W", outputs_, inputs_ ));
		m.add (std::make_tuple ( "b", outputs_, 1 ));

		// Allocate gradients: W and b.
		g.add (std::make_tuple ( "W", outputs_, inputs_ ));
		g.add (std::make_tuple ( "b", outputs_, 1 ));

	};


	/*!
	 * Virtual constructor.
	 */
	virtual ~SparseLinear() {};

	/*!
	 * Forward pass.
	 */
	void forward(bool test_ = false) {
		// y = W * x + b
		(*s['y']) = (*p['W']) * (*s['x']) + (*p['b']).replicate(1, (*s['x']).cols());
	}

	/*!
	 * Backward pass.
	 */
	void backward() {
		/*dW = dy * x.transpose();
		db = dy.rowwise().sum();
		dx = W.transpose() * dy;*/

		(*g['W']) = (*g['y']) * ((*s['x']).transpose());
		(*g['b']) = (*g['y']).rowwise().sum();
		(*g['x']) = (*p['W']).transpose() * (*g['y']);
	}

	/*!
	 * Reset the gradients.
	 */
	void resetGrads() {
		(*g['W']).setZero();
		(*g['b']).setZero();
	}

	/*!
	 * Apply the gradient update.
	 */
	void applyGrads(double alpha_) {
		//adagrad
		//mW += dW.cwiseProduct(dW);
		(*m['W']) += (*g['W']).cwiseProduct((*g['W']));

		//mb += db.cwiseProduct(db);
		(*m['b']) += (*g['b']).cwiseProduct((*g['b']));

		//W = (1 - decay_) * W + alpha_ * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
		//(*p['W']) = (1.0f - decay_) * (*p['W']) + alpha_ * (*g['W']).cwiseQuotient((*m['W']).unaryExpr(std::ptr_fun<eT>(sqrt_eps)));

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
    /*!
	 * Private constructor, used only during the deserialization.
	 */
	SparseLinear<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SPARSELINEAR_H_ */
