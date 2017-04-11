/*!
 * @file: RegressionLoss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef SQUAREDLOSS_HPP_
#define SQUAREDLOSS_HPP_

#include <loss/Loss.hpp>

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Class representing a squared error loss function (regression).
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class RegressionLoss : public Loss<dtype> {
public:
	/*!
	 * \brief Function calculating loss - for squared difference (regression).
	 */
	dtype calculateLoss (mic::types::MatrixPtr<dtype> predicted_y_, mic::types::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate loss.
		dtype loss =0;
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			loss += ((*predicted_y_)[i] - (*target_y_)[i])*((*predicted_y_)[i] - (*target_y_)[i]);
		}
		return loss;
	}

	/*!
	 * \brief Function calculating gradient - for squared difference (regression).
	 */
	mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> predicted_y_, mic::types::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->size(), 1);
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			(*dy)[i] = 2.0*((*predicted_y_)[i] - (*target_y_)[i]);
		}
		return dy;
	}

};

} //: loss
} //: neural_nets
} //: mic

#endif /* SQUAREDLOSS_HPP_ */
