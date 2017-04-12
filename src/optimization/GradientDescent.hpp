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

	/*!
	 * Constructor. Sets dimensions.
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	GradientDescent(size_t rows_, size_t cols_) {
	}

	/*!
	 * Performs update in the direction of gradient descent.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		// W = W - alpha * dW.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*x_)[i] -= learning_rate_ * (*dx_)[i];
	}

};

} //: optimization
} //: neural_nets
} //: mic

#endif /* GRADIENTDESCENT_HPP_ */
