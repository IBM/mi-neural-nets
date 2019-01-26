/*!
 * @file: SquaredErrorLoss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef SQUAREDLOSS_HPP_
#define SQUAREDLOSS_HPP_

#include "loss/Loss.hpp"

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Class representing a squared error loss function (regression). L = 1/2 sum (t - p)^2.
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class SquaredErrorLoss : public Loss<dtype> {
public:
	/*!
	 * \brief Function calculates squared difference loss (regression) and returns squared error (SE).
	 */
	dtype calculateLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate loss.
		dtype loss =0;
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			loss += ((*target_y_)[i] - (*predicted_y_)[i])*((*target_y_)[i] - (*predicted_y_)[i]);
		}
		// Return squared error (SE).
		// The mean squared error (MSE) is calculated by dividing the SE by the size of a batch.
		return loss/2.0;
	}

	/*!
	 * \brief Function calculating gradient - for squared difference (regression).
	 */
	mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->rows(), predicted_y_->cols());
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			(*dy)[i] = -((*target_y_)[i] - (*predicted_y_)[i]);
		}

		/*std::cout << " predicted_y_ = " << (*predicted_y_) << std::endl;
		std::cout << " target_y_ = " << (*target_y_) << std::endl;
		std::cout << " dy = (p-t) = " << (*dy) << std::endl;*/

		return dy;
	}

};

} //: loss
} //: neural_nets
} //: mic

#endif /* SQUAREDLOSS_HPP_ */
