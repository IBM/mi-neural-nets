/*!
 * @file: CrossEntropyLoss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 9, 2016
 *
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef CROSSENTROPYLOSS_HPP_
#define CROSSENTROPYLOSS_HPP_

#include <cmath>
#include <loss/Loss.hpp>

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Class representing a cross-entropy loss function (classification).
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class CrossEntropyLoss : public Loss<dtype> {
public:
	/*!
	 * \brief Calculates cross entropy(using log)	 and returns cross-entropy error (CE).
	 */
	dtype calculateLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate loss (negative log probability).
		dtype loss =0;
		dtype eps = 1e-15;
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			// -t * log (y + eps!)
			loss -= (*target_y_)[i] * std::log((*predicted_y_)[i] + eps);
		}
		// Return cross-entropy error (CE).
		// The average cross entropy error (ACE) is loss divided by the batch size.
		//assert(std::isfinite(loss));
		return loss;
	}

	/*!
	 * \brief Gradient calculation for cross-entropy.
	 */
	mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->rows(), predicted_y_->cols());
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			// y - t
			(*dy)[i] = (*predicted_y_)[i] - (*target_y_)[i];
		}
		return dy;
	}

};

} //: loss
} //: neural_nets
} //: mic

#endif /* CROSSENTROPYLOSS_HPP_ */
