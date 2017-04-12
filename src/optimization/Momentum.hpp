/*!
 * @file: Momentum.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef MOMENTUM_HPP_
#define MOMENTUM_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Update in the direction of gradient descent - with momentum.
 * \author tkornuta
 */
template <typename eT=float>
class Momentum : public OptimizationFunction<eT> {
public:

	/*!
	 * Constructor. Sets dimensions and momentum (default=0.9).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	Momentum(size_t rows_, size_t cols_, eT momentum_ = 0.9) : momentum(momentum_) {
		v = MAKE_MATRIX_PTR(eT, rows_, cols_);
		// Reset momentum.
		v->zeros();
	}

	/*!
	 * Performs update according to the Momentum update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001). NOT USED!
	 */
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == v->size());

		// Update the update vector.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*v)[i] = momentum * (*v)[i] + learning_rate_ * (*dx_)[i];

		// Theta = Theta - update.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= (*v)[i];
	}

protected:
	/// Update vector.
	mic::types::MatrixPtr<eT> v;

	/// Momentum rate.
	eT momentum;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* MOMENTUM_HPP_ */
