/*!
 * @file: AdamID.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef ADAMID_HPP_
#define ADAMID_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief AdamID - ADAM with integral and derivative coefficients.
 * \author tkornuta
 */
template <typename eT=float>
class AdamID : public OptimizationFunction<eT> {
public:

	/*!
	 * Constructor. Sets dimensions, values of decay (beta1=0.9 and beta2=0.999) and eps (default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	AdamID(size_t rows_, size_t cols_, eT beta1_ = 0.9, eT beta2_ = 0.999, eT eps_ = 1e-8)
		: beta1(beta1_), beta2(beta2_), eps(eps_)
{
		Edx = MAKE_MATRIX_PTR(eT, rows_, cols_);
		Edx->zeros();

		Edx2 = MAKE_MATRIX_PTR(eT, rows_, cols_);
		Edx2->zeros();

		dx_prev = MAKE_MATRIX_PTR(eT, rows_, cols_);
		dx_prev->zeros();

		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();

		beta1_powt = beta1;
		beta2_powt = beta2;
	}

	/*!
	 * Performs update according to the AdamID update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_ = 0.001) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == Edx->size());

		// Update decaying sum of gradients - up to time t. INTEGRAL.
		for (size_t i=0; i< (size_t)Edx->size(); i++) {
			(*Edx)[i] = beta1 *(*Edx)[i] + (1.0 - beta1) * (*dx_)[i];
			assert(std::isfinite((*Edx)[i]));
		}

		// Update decaying sum of squared gradients - up to time t. NORMALIZER.
		for (size_t i=0; i< (size_t)Edx2->size(); i++) {
			(*Edx2)[i] = beta2 *(*Edx2)[i] + (1.0 - beta2) * (*dx_)[i] * (*dx_)[i];
			assert(std::isfinite((*Edx2)[i]));
		}

		// Calculate update.
//		std::cout <<"Delta = ";
		for(size_t i=0; i< (size_t)delta->size(); i++) {
			// update = integral + small derivative correction.
			// i.e. lr * I + lr^2 * D.
			eT delta_ID =  learning_rate_ * (*Edx)[i] + learning_rate_*learning_rate_ * ((*dx_)[i] - (*dx_prev)[i]);
			(*delta)[i] = 1.0 / (sqrt( (*Edx2)[i] / (1 - beta2_powt)) + eps) * ( delta_ID  ) / (1 - beta1_powt);
//			std::cout<< (*delta)[i] << " | ";
			assert(std::isfinite((*delta)[i]));
		}
//		std::cout << std::endl;

		// Perform the update.
		for (size_t i=0; i< (size_t)delta->size(); i++) {
			(*x_)[i] -= (*delta)[i];
		}

		// Store past gradients.
		// Perform the update.
		for (size_t i=0; i< (size_t)dx_->size(); i++) {
			(*dx_prev)[i] = (*dx_)[i];
		}

//		std::cout << std::endl;

		// Update "powered" factors.
		beta1_powt *= beta1;
		beta2_powt *= beta2;

	}

protected:
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
	mic::types::MatrixPtr<eT> Edx;

	/// Decaying average of squared gradients up to time t - E[g^2].
	mic::types::MatrixPtr<eT> Edx2;

	/// Previous value of gradients.
	mic::types::MatrixPtr<eT> dx_prev;

	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;
};


} //: optimization
} //: neural_nets
} //: mic



#endif /* ADAMID_HPP_ */
