/*!
 * @file: AdaDelta.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef ADADELTA_HPP_
#define ADADELTA_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Update using AdaDelta - adaptive gradient descent with running average E[g^2] and  E[d^2].
 * \author tkornuta
 */
template <typename eT=float>
class AdaDelta : public OptimizationFunction<eT> {
public:

	/// Constructor.
	AdaDelta(size_t dims_, eT decay_ = 0.9, eT eps_ = 1e-8) : decay(decay_), eps(eps_) {
		EG = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*EG)[i] = 0.0;
		ED = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*ED)[i] = 0.0;
		delta = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*delta)[i] = 0.0;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == EG->size());

/*		for (size_t i=0; i<x_->size(); i++) {
			std::cout << "(*x)["<< i <<"] = " << (*x_)[i] <<std::endl;
		}
		for (size_t i=0; i<x_->size(); i++) {
			std::cout << "(*dx_)["<< i <<"] = " << (*dx_)[i] <<std::endl;
		}*/

		// Update decaying sum of squares of gradients - up to time t.
		for (size_t i=0; i<(size_t)x_->size(); i++) {
			(*EG)[i] = decay *(*EG)[i] + (1.0 - decay) * (*dx_)[i] * (*dx_)[i];
//			std::cout << "(*EG)["<< i <<"] = " << (*EG)[i] <<std::endl;
			assert(std::isfinite((*EG)[i]));
		}

		// Update decaying sum of squares of updates - up to time t-1.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*ED)[i] = decay *(*ED)[i] + (1 - decay) * (*delta)[i] * (*delta)[i];

		// Calculate updates - and store as previous (already) = - RMS(ED)/(RMS(G) * dx
		for (size_t i=0; i<(size_t)x_->size(); i++){
//			(*prev_d)[i] = - (0.1 / std::sqrt((*EG)[i] + eps)) * (*dx_)[i];
			(*delta)[i] = - (std::sqrt((*ED)[i] + eps) / std::sqrt((*EG)[i] + eps)) * (*dx_)[i];
//			std::cout << "(*prev_d)["<< i <<"] = " << (*prev_d)[i] <<std::endl;
			assert(std::isfinite((*delta)[i]));
		}

		// Perform the update.
		for (size_t i=0; i<(size_t)x_->size(); i++) {
			(*x_)[i] += (*delta)[i];
//			std::cout << "(*x_)["<< i <<"] = " << (*x_)[i] <<std::endl;
		}
//		std::cout << std::endl;

	}

protected:
	/// Decay ratio, similar to momentum.
	eT decay;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Decaying average of the squares of gradients up to time t ("diagonal matrix") - E[g^2].
	mic::types::MatrixPtr<eT> EG;

	/// Decaying average of the squares of updates up to time t ("diagonal matrix") - E[delta Theta^2].
	mic::types::MatrixPtr<eT> ED;

	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;
};


} //: optimization
} //: neural_nets
} //: mic

#endif /* ADADELTA_HPP_ */
