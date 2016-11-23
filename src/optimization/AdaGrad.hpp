/*!
 * @file: AdaGrad.hpp
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

#ifndef ADAGRAD_HPP_
#define ADAGRAD_HPP_

#include <optimization/OptimizationFunction.hpp>


/*!
 * \brief Update using AdaGrad - adaptive gradient descent.
 * \author tkornuta
 */
template <typename eT=float>
class AdaGrad : public OptimizationFunction<eT> {
public:

	/// Constructor.
	AdaGrad(size_t dims_, eT learning_rate_ = 0.01, eT eps_ = 1e-8) : learning_rate(learning_rate_), eps(eps_) {
		G = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*G)[i] = 0.0;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == G->size());

		// Update G - add square of the gradients.
		for (size_t i=0; i<x_->size(); i++)
				(*G)[i] += (*dx_)[i] * (*dx_)[i];

		// W = W - alpha * dW.
		for (size_t i=0; i<x_->size(); i++)
			(*x_)[i] -= learning_rate * (*dx_)[i] / (std::sqrt((*G)[i] + eps));
	}

protected:
	/// Learning rate.
	eT learning_rate;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Sum of all of the squares of the gradients up to time t ("diagonal matrix").
	mic::types2::MatrixPtr<eT> G;
};




#endif /* ADAGRAD_HPP_ */
