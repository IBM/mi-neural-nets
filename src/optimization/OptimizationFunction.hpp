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
 * @file: OptimizationFunctions.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 11, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef OPTIMIZATIONFUNCTIONS_HPP_
#define OPTIMIZATIONFUNCTIONS_HPP_

#include <types/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Abstract class representing interface to optimization function.
 * \author tkornuta
 * \tparam eT Template type (single/double precision)
 */
template <typename eT=float>
class OptimizationFunction {
public:
	/*!
	 * Constructor. Remembers dimensions.
	 */
	OptimizationFunction () { }

	/// Virtual destructor - empty.
	virtual ~OptimizationFunction () { }

	/*!
	 * Method responsible for performing the update using backpropagation and gradient descent. Calls abstract method calculateUpdate().
	 * @param p_ Pointer to the current parameter (matrix).
	 * @param dp_ Pointer to current gradient of that parameter (matrix).
	 * @param learning_rate_ Learning rate.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT = 0.0 means "no decay").
	 */
	virtual void update(mic::types::MatrixPtr<eT> p_, mic::types::MatrixPtr<eT> dp_, eT learning_rate_, eT decay_ = 0.0) {

		// Calculate the update.
		mic::types::MatrixPtr<eT> delta = calculateUpdate(p_, dp_, learning_rate_);

		//assert(std::isfinite((*delta)[i]));

		// Perform the update: x = x - delta (with optional weight decay).
		for (size_t i=0; i< (size_t)delta->size(); i++) {
			(*p_)[i] = (1.0f - decay_) * (*p_)[i] - (*delta)[i];
		}//: for
	}

	/*!
	 * Updates the weight matrix according to the hebbian rule.
	 * @param p_ Pointer to the parameter (weight) matrix.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	virtual void update(mic::types::MatrixPtr<eT> p_, mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_ = 0.001) {
		assert(p_->rows() == y_->rows());
		assert(p_->cols() == x_->rows());
		assert(x_->cols() == y_->cols());

		// Calculate the update using hebbian "fire together, wire together".
		mic::types::MatrixPtr<eT> delta = calculateUpdate(x_, y_, learning_rate_);

		// weight += delta;
		(*p_) += (*delta);
	}


	/*!
	 * Abstract method responsible for calculating the update.
	 * @param x_ Pointer to the current matrix (parameter) OR Pointer to current input matrix (in Hebbian learning).
	 * @param dx_ Pointer to current gradient of that matrix (parameter) OR Pointer to current output matrix (in Hebbian learning).
	 * @param learning_rate_ Learning rate.
	 */
	virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_) = 0;



};


} //: optimization
} //: neural_nets
} //: mic


#endif /* OPTIMIZATIONFUNCTIONS_HPP_ */
