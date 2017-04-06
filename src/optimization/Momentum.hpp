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

	/// Constructor. Sets learning rate and momentum.
	Momentum(size_t dims_, eT learning_rate_, eT momentum_ = 0.9) : learning_rate(learning_rate_), momentum(momentum_){
		v = MAKE_MATRIX_PTR(eT, dims_, 1);
		// Reset momentum.
		v->zeros();
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == v->size());

		// Update the update vector.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*v)[i] = momentum * (*v)[i] + learning_rate * (*dx_)[i];

		// Theta = Theta - update.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= (*v)[i];
	}

protected:
	/// Update vector.
	mic::types::MatrixPtr<eT> v;

	/// Learning rate.
	eT learning_rate;

	/// Momentum rate.
	eT momentum;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* MOMENTUM_HPP_ */
