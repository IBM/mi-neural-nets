/*!
 * @file: RMSProp.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef RMSPROP_HPP_
#define RMSPROP_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Update using RMSProp - adaptive gradient descent with running average E[g^2].
 * \author tkornuta
 */
template <typename eT=float>
class RMSProp : public OptimizationFunction<eT> {
public:

	/*!
	 * Constructor. Sets dimensions, values of decay (default=0.9) and eps (default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	RMSProp(size_t rows_, size_t cols_, eT decay_ = 0.9, eT eps_ = 1e-8) : decay(decay_), eps(eps_) {
		EG = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		// Reset EG and delta.
		EG->zeros();
		delta->zeros();
	}

	/*!
	 * Calculates the update according to the RMSProp update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001). NOT USED!
	 */
	mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == EG->size());

		// Update decaying sum of squares of gradients - up to time t.
		for (size_t i=0; i<(size_t)x_->size(); i++) {
			(*EG)[i] = decay *(*EG)[i] + (1.0 - decay) * (*dx_)[i] * (*dx_)[i];
			assert(std::isfinite((*EG)[i]));
		}

		// Calculate updates - and store as previous (already) = - RMS(ED)/(RMS(G) * dx
		for (size_t i=0; i<(size_t)x_->size(); i++){
			(*delta)[i] = (learning_rate_ / std::sqrt((*EG)[i] + eps)) * (*dx_)[i];
			assert(std::isfinite((*delta)[i]));
		}

		// Return the update.
		return delta;
	}

protected:
	/// Decay ratio, similar to momentum.
	eT decay;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Decaying average of the squares of gradients up to time t ("diagonal matrix") - E[g^2].
	mic::types::MatrixPtr<eT> EG;

	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* RMSPROP_HPP_ */
