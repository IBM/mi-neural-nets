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
 * @file: Loss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef LOSS_HPP_
#define LOSS_HPP_

#include <types/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Abstract class representing a loss function.
 * Defines interfaces.
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class Loss {
public:
	/*!
	 * \brief Function calculating loss - abstract.
	 */
	virtual dtype calculateLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) = 0;

	/*!
	 * \brief Calculates mean loss (i.e. divides the loss by the size of batch) - ACE for cross-entropy or MSE for regression.
	 */
	virtual dtype calculateMeanLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		return calculateLoss(target_y_, predicted_y_) / predicted_y_->cols();
	}

	/*!
	 * \brief Function calculating gradient - abstract.
	 */
	virtual mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) = 0;

};

} //: loss
} //: neural_nets
} //: mic

#endif /* LOSS_HPP_ */
