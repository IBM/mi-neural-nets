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

	/*!
	 * Constructor. Sets dimensions and eps (default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	AdaGrad(size_t rows_, size_t cols_, eT eps_ = 1e-8) : eps(eps_) {
		G = MAKE_MATRIX_PTR(eT, rows_, cols_);
		// Reset G.
		G->zeros();
	}

	/*!
	 * Performs update according to the AdaGrad update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == G->size());

		// Update G - add square of the gradients.
		for (size_t i=0; i<(size_t)x_->size(); i++)
				(*G)[i] += (*dx_)[i] * (*dx_)[i];

		// W = W - alpha * dW.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= learning_rate_ * (*dx_)[i] / (std::sqrt((*G)[i] + eps));
	}

protected:
	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Sum of all of the squares of the gradients up to time t ("diagonal matrix").
	mic::types::MatrixPtr<eT> G;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* ADAGRAD_HPP_ */
