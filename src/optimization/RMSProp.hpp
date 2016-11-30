/*!
 * @file: RMSProp.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
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
template <typename dtype=float>
class RMSProp : public OptimizationFunction<dtype> {
public:

	/// Constructor.
	RMSProp(size_t dims_, dtype learning_rate_=0.001, dtype decay_ = 0.9, dtype eps_ = 1e-8) : learning_rate(learning_rate_), decay(decay_), eps(eps_) {
		EG = MAKE_MATRIX_PTR(dtype, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*EG)[i] = 0.0;

		delta = MAKE_MATRIX_PTR(dtype, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*delta)[i] = 0.0;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<dtype> x_, mic::types2::MatrixPtr<dtype> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == EG->size());

		// Update decaying sum of squares of gradients - up to time t.
		for (size_t i=0; i<x_->size(); i++) {
			(*EG)[i] = decay *(*EG)[i] + (1.0 - decay) * (*dx_)[i] * (*dx_)[i];
			assert(std::isfinite((*EG)[i]));
		}

		// Calculate updates - and store as previous (already) = - RMS(ED)/(RMS(G) * dx
		for (size_t i=0; i<x_->size(); i++){
			(*delta)[i] = - (learning_rate / std::sqrt((*EG)[i] + eps)) * (*dx_)[i];
			assert(std::isfinite((*delta)[i]));
		}

		// Perform the update.
		for (size_t i=0; i<x_->size(); i++) {
			(*x_)[i] += (*delta)[i];
		}
	}

protected:
	/// Learning rate.
	dtype learning_rate;

	/// Decay ratio, similar to momentum.
	dtype decay;

	/// Smoothing term that avoids division by zero.
	dtype eps;

	/// Decaying average of the squares of gradients up to time t ("diagonal matrix") - E[g^2].
	mic::types2::MatrixPtr<dtype> EG;

	/// Calculated update.
	mic::types2::MatrixPtr<dtype> delta;
};

} //: optimization
} //: neural_nets
} //: mic

#endif /* RMSPROP_HPP_ */
