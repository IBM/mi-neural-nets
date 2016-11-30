/*!
 * @file: SquaredErrorLoss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */


#ifndef SQUAREDLOSS_HPP_
#define SQUAREDLOSS_HPP_

#include "Loss.hpp"

/*!
 * \brief Class representing a squared error loss function (regression).
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class SquaredErrorLoss : public Loss<dtype> {
public:
	/*!
	 * \brief Function calculating loss - for squared difference (regression).
	 */
	dtype calculateLoss (mic::types2::MatrixPtr<dtype> predicted_y_, mic::types2::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate loss.
		dtype loss =0;
		for (size_t i=0; i <predicted_y_->size(); i++) {
			loss += ((*predicted_y_)[i] - (*target_y_)[i])*((*predicted_y_)[i] - (*target_y_)[i]);
		}
		return loss;
	}

	/*!
	 * \brief Function calculating gradient - for squared difference (regression).
	 */
	mic::types2::MatrixPtr<dtype> calculateGradient (mic::types2::MatrixPtr<dtype> predicted_y_, mic::types2::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types2::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->size(), 1);
		for (size_t i=0; i <predicted_y_->size(); i++) {
			(*dy)[i] = 2.0*((*predicted_y_)[i] - (*target_y_)[i]);
		}
		return dy;
	}

};




#endif /* SQUAREDLOSS_HPP_ */
