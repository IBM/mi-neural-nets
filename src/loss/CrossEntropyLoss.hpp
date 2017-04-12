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
	 * \brief Calculates cross entropy (using log 2) [BPC].
	 */
	dtype calculateLoss (mic::types::MatrixPtr<dtype> predicted_y_, mic::types::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate loss.
		dtype loss =0;
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			loss -= (*target_y_)[i] * std::log2((*predicted_y_)[i]);
		}
		return loss;
	}

	/*!
	 * \brief Gradient calculation for cross-entropy.
	 */
	mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> predicted_y_, mic::types::MatrixPtr<dtype> target_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->size(), 1);
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			(*dy)[i] = (*target_y_)[i] -(*predicted_y_)[i] ;
		}
		return dy;
	}

};

} //: loss
} //: neural_nets
} //: mic

#endif /* CROSSENTROPYLOSS_HPP_ */
