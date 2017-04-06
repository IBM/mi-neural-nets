/*!
 * @file: AdaGrad.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef ADAGRAD_HPP_
#define ADAGRAD_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Update using AdaGrad - adaptive gradient descent.
 * \author tkornuta
 */
template <typename eT=float>
class AdaGrad : public OptimizationFunction<eT> {
public:

	/// Constructor.
	AdaGrad(size_t dims_, eT learning_rate_ = 0.01, eT eps_ = 1e-8) : learning_rate(learning_rate_), eps(eps_) {
		G = MAKE_MATRIX_PTR(eT, dims_, 1);
		// Reset G.
		G->zeros();
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == G->size());

		// Update G - add square of the gradients.
		for (size_t i=0; i<(size_t)x_->size(); i++)
				(*G)[i] += (*dx_)[i] * (*dx_)[i];

		// W = W - alpha * dW.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= learning_rate * (*dx_)[i] / (std::sqrt((*G)[i] + eps));
	}

protected:
	/// Learning rate.
	eT learning_rate;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Sum of all of the squares of the gradients up to time t ("diagonal matrix").
	mic::types::MatrixPtr<eT> G;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* ADAGRAD_HPP_ */
