/*!
 * @file: Adam.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef ADAM_HPP_
#define ADAM_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Adam - adaptive moment estimation.
 * \author tkornuta
 */
template <typename eT=float>
class Adam : public OptimizationFunction<eT> {
public:

	/*!
	 * Constructor. Sets dimensions, momentum rates (beta1=0.9 and beta2=0.999) and eps(default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	Adam(size_t rows_, size_t cols_, eT beta1_ = 0.9, eT beta2_ = 0.999, eT eps_ = 1e-8)
			: beta1(beta1_), beta2(beta2_), eps(eps_)
{
		m = MAKE_MATRIX_PTR(eT, rows_, cols_);
		m->zeros();

		v = MAKE_MATRIX_PTR(eT, rows_, cols_);
		v->zeros();

		// Allocate and reset delta.
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();

		beta1_powt = beta1;
		beta2_powt = beta2;
	}

	/*!
	 * Calculates the update according to the ADAM update rule.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_  = 0.001) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == m->size());

		// Update the decaying average of past gradients.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*m)[i] = beta1 * (*m)[i] + (1-beta1) * (*dx_)[i];

		// Update the decaying average of past squared gradients.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*v)[i] = beta2 * (*v)[i] + (1-beta2) * (*dx_)[i] * (*dx_)[i];

		// Calculate the update.
		for (size_t i=0; i<(size_t)x_->size(); i++)
			(*delta)[i] = learning_rate_ / (sqrt( (*v)[i] / (1 - beta2_powt)) + eps) * (*m)[i] / (1 - beta1_powt);

		// Update "powered" factors.
		beta1_powt *= beta1;
		beta2_powt *= beta2;

		// Return the update.
		return delta;
	}

protected:
	/// Exponentially decaying average of past gradients.
	mic::types::MatrixPtr<eT> m;

	/// Exponentially decaying average of past squared gradients.
	mic::types::MatrixPtr<eT> v;

	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

	/// Decay rate 1 (momentum for past gradients).
	eT beta1;

	/// Decay rate 2 (momentum for past squared gradients).
	eT beta2;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Decay rate 1 to the power of t.
	eT beta1_powt;

	/// Decay rate 2 to the power of t.
	eT beta2_powt;

};

} //: optimization
} //: neural_nets
} //: mic

#endif /* ADAM_HPP_ */
