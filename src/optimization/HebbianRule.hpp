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
 * @file: HebbianRule.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef HEBBIAN_HPP_
#define HEBBIAN_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace learning {

/*!
 * \brief
 * \author tkornuta
 */
/*!
 * \brief Updates according to classical Hebbian rule (wij += ni * x * y).
 * \author tkornuta
 */
template <typename eT=float>
class HebbianRule : public mic::neural_nets::optimization::OptimizationFunction<eT> {
public:
	/*!
	 * Constructor. Sets dimensions, momentum rates (beta1=0.9 and beta2=0.999) and eps(default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	HebbianRule(size_t rows_, size_t cols_) {
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();
	}

	// Virtual destructor - empty.
	virtual ~HebbianRule() { }


	/*!
	 * Calculates the update according to the hebbian rule.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_) {
		// delta + alpha * x * y.
		(*delta) = learning_rate_ * (*y_) * ((*x_).transpose());

		return delta;
	}


protected:
	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

};

} //: namespace learning
} /* namespace neural_nets */
} /* namespace mic */

#endif /* HEBBIAN_HPP_ */
