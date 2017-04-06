/*!
 * @file: GradientDescent.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef GRADIENTDESCENT_HPP_
#define GRADIENTDESCENT_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Update in the direction of gradient descent.
 * \author tkornuta
 */
template <typename eT=float>
class GradientDescent : public OptimizationFunction<eT> {
public:

	/// Constructor.
	GradientDescent(eT learning_rate_) : learning_rate(learning_rate_){
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		// W = W - alpha * dW.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= learning_rate * (*dx_)[i];
	}

protected:
	/// Learning rate.
	eT learning_rate;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* GRADIENTDESCENT_HPP_ */
