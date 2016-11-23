/*!
 * @file: Momentum.hpp
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

#ifndef MOMENTUM_HPP_
#define MOMENTUM_HPP_

#include <optimization/OptimizationFunction.hpp>

/*!
 * \brief Update in the direction of gradient descent - with momentum.
 * \author tkornuta
 */
template <typename eT=float>
class Momentum : public OptimizationFunction<eT> {
public:

	/// Constructor. Sets learning rate and momentum.
	Momentum(size_t dims_, eT learning_rate_, eT momentum_ = 0.9) : learning_rate(learning_rate_), momentum(momentum_){
		v = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*v)[i] = 0.0;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == v->size());

		// Update the update vector.
		for (size_t i=0; i<x_->size(); i++)
			(*v)[i] = momentum * (*v)[i] + learning_rate * (*dx_)[i];

		// Theta = Theta - update.
		for (size_t i=0; i<x_->size(); i++)
			(*x_)[i] -= (*v)[i];
	}

protected:
	/// Update vector.
	mic::types2::MatrixPtr<eT> v;

	/// Learning rate.
	eT learning_rate;

	/// Momentum rate.
	eT momentum;
};





#endif /* MOMENTUM_HPP_ */
