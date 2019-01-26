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
	 * Calculates the update according to the Momentum update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001). NOT USED!
	 */
	mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == v->size());

		// Calculate the update vector (delta).
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*v)[i] = momentum * (*v)[i] + learning_rate_ * (*dx_)[i];

		// Return the update.
		return v;
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
