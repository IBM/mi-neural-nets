/*!
 * @file: LogLikelihoodLoss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 12, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef LOGLIKELIHOODLOSS_HPP_
#define LOGLIKELIHOODLOSS_HPP_






#include <cmath>
#include <loss/Loss.hpp>

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Class representing a log-likelihood cost (to be used with softmax logistic regression).
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class LogLikelihoodLoss : public Loss<dtype> {
public:
	/*!
	 * \brief Calculates log-likelihood cost.
	 */
	dtype calculateLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		size_t ind;
		// Calculate loss.
		dtype loss =0;
		// For each column (sample from batch).
		for (size_t i=0; i <(size_t)predicted_y_->cols(); i++) {
			// Get index of max coefficient in given column.
			target_y_.col(i).maxCoeff(ind);

			// Add loss.
			loss -= std::log((*predicted_y_)[ind]);
		}//: for
		// Return sum of log-likelihood cost.
		return loss;
		// Divide it by the batch size in order to calculate the mean loss.
	}

	/*!
	 * \brief Gradient calculation for log-likelihood cost. NOT FINISHED!!
	 */
	mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) {
		// Sizes must match.
		assert(predicted_y_->size() == target_y_->size());

		// Calculate gradient.
		mic::types::MatrixPtr<dtype> dy = MAKE_MATRIX_PTR(dtype, predicted_y_->rows(), predicted_y_->cols());
		for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
			(*dy)[i] = 0.0;
		}
		return dy;
	}

};

} //: loss
} //: neural_nets
} //: mic

#endif /* LOGLIKELIHOODLOSS_HPP_ */
