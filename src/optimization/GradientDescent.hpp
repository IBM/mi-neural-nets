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
		// Allocate and reset delta.
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();
	}

	/*!
	 * Calculates the update in the direction of gradient descent.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		// daltea = - alpha * dW.
		//for (size_t i=0; i<(size_t)x_->size(); i++)
			(*delta) = learning_rate_ * (*dx_);

		// Return the update.
		return delta;
	}

private:
	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

};

} //: optimization
} //: neural_nets
} //: mic

#endif /* GRADIENTDESCENT_HPP_ */
