/*!
 * @file: AdamID.hpp
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

#ifndef ADAMID_HPP_
#define ADAMID_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief BiasedGradPID - gradient descent with proportional, integral and derivative coefficients, with Adam-like bias correction.
 * \author tkornuta
 */
template <typename eT=float>
class AdamID : public OptimizationFunction<eT> {
public:

	/// Constructor.
	AdamID(size_t dims_, eT learning_rate_ = 0.1, eT beta1_ = 0.9, eT beta2_ = 0.999, eT eps_ = 1e-8)
		: learning_rate(learning_rate_), beta1(beta1_), beta2(beta2_), eps(eps_)
{
		Edx = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*Edx)[i] = 0.0;

		Edx2 = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*Edx2)[i] = 0.0;

		dx_prev = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*dx_prev)[i] = 0.0;

		delta = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*delta)[i] = 0.0;

		beta1_powt = beta1;
		beta2_powt = beta2;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == Edx->size());

		// Update decaying sum of gradients - up to time t. INTEGRAL.
		for (size_t i=0; i<Edx->size(); i++) {
			(*Edx)[i] = beta1 *(*Edx)[i] + (1.0 - beta1) * (*dx_)[i];
			assert(std::isfinite((*Edx)[i]));
		}

		// Update decaying sum of squared gradients - up to time t. NORMALIZER.
		for (size_t i=0; i<Edx2->size(); i++) {
			(*Edx2)[i] = beta2 *(*Edx2)[i] + (1.0 - beta2) * (*dx_)[i] * (*dx_)[i];
			assert(std::isfinite((*Edx2)[i]));
		}



		// Calculate update.
//		std::cout <<"Delta = ";
		for(size_t i=0; i< delta->size(); i++) {
			// update = integral + small derivative correction.
			// i.e. lr * I + lr^2 * D.
			eT delta_ID =  learning_rate * (*Edx)[i] + learning_rate*learning_rate * ((*dx_)[i] - (*dx_prev)[i]);
			(*delta)[i] = 1.0 / (sqrt( (*Edx2)[i] / (1 - beta2_powt)) + eps) * ( delta_ID  ) / (1 - beta1_powt);
//			std::cout<< (*delta)[i] << " | ";
			assert(std::isfinite((*delta)[i]));
		}
//		std::cout << std::endl;

		// Perform the update.
		for (size_t i=0; i<delta->size(); i++) {
			(*x_)[i] -= (*delta)[i];
		}

		// Store past gradients.
		// Perform the update.
		for (size_t i=0; i<dx_->size(); i++) {
			(*dx_prev)[i] = (*dx_)[i];
		}

//		std::cout << std::endl;

		// Update "powered" factors.
		beta1_powt *= beta1;
		beta2_powt *= beta2;

	}

protected:

	/// Learning rate.
	eT learning_rate;

	/// Decay rate 1 (momentum for past gradients).
	eT beta1;

	/// Decay rate 2 (momentum for past squared gradients).
	eT beta2;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Decay rate 1 to the power of t - bias correction.
	eT beta1_powt;

	/// Decay rate 2 to the power of t - bias correction.
	eT beta2_powt;

	/// Decaying average of gradients up to time t - E[g].
	mic::types2::MatrixPtr<eT> Edx;

	/// Decaying average of squared gradients up to time t - E[g^2].
	mic::types2::MatrixPtr<eT> Edx2;

	/// Previous value of gradients.
	mic::types2::MatrixPtr<eT> dx_prev;

	/// Calculated update.
	mic::types2::MatrixPtr<eT> delta;
};


} //: optimization
} //: neural_nets
} //: mic



#endif /* ADAMID_HPP_ */
