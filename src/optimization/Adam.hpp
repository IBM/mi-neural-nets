/*!
 * @file: Adam.hpp
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

#ifndef ADAM_HPP_
#define ADAM_HPP_

#include <optimization/OptimizationFunction.hpp>

/*!
 * \brief Adam - adaptive moment estimation.
 * \author tkornuta
 */
template <typename eT=float>
class Adam : public OptimizationFunction<eT> {
public:

	/// Constructor. Sets learning rate and momentum.
	Adam(size_t dims_, eT learning_rate_, eT beta1_ = 0.9, eT beta2_ = 0.999, eT eps_ = 1e-8)
			: learning_rate(learning_rate_), beta1(beta1_), beta2(beta2_), eps(eps_)
{
		m = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*m)[i] = 0.0;

		v = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*v)[i] = 0.0;

		beta1_powt = beta1;
		beta2_powt = beta2;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == m->size());

		// Update the decaying average of past gradients.
		for (size_t i=0; i<x_->size(); i++)
			(*m)[i] = beta1 * (*m)[i] + (1-beta1) * (*dx_)[i];

		// Update the decaying average of past squared gradients.
		for (size_t i=0; i<x_->size(); i++)
			(*v)[i] = beta2 * (*v)[i] + (1-beta2) * (*dx_)[i] * (*dx_)[i];

		// Theta = Theta - update.
		for (size_t i=0; i<x_->size(); i++)
			(*x_)[i] -= learning_rate / (sqrt( (*v)[i] / (1 - beta2_powt)) + eps) * (*m)[i] / (1 - beta1_powt);

		// Update "powered" factors.
		beta1_powt *= beta1;
		beta2_powt *= beta2;
	}

protected:
	/// Exponentially decaying average of past gradients.
	mic::types2::MatrixPtr<eT> m;

	/// Exponentially decaying average of past squared gradients.
	mic::types2::MatrixPtr<eT> v;

	/// Learning rate.
	eT learning_rate;

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






#endif /* ADAM_HPP_ */
