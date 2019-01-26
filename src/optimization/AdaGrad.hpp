/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

		// Allocate and reset delta.
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();
	}

	/*!
	 * Performs update according to the AdaGrad update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == G->size());

		// Update G - add square of the gradients.
		for (size_t i=0; i<(size_t)x_->size(); i++)
				(*G)[i] += (*dx_)[i] * (*dx_)[i];

		// delta = alpha * dW.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*delta)[i] = learning_rate_ * (*dx_)[i] / (std::sqrt((*G)[i] + eps));

		// Return the update.
		return delta;
	}

protected:
	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Sum of all of the squares of the gradients up to time t ("diagonal matrix").
	mic::types::MatrixPtr<eT> G;

	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* ADAGRAD_HPP_ */
