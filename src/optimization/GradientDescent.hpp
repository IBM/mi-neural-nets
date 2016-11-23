/*!
 * @file: GradientDescent.hpp
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

#ifndef GRADIENTDESCENT_HPP_
#define GRADIENTDESCENT_HPP_

#include <optimization/OptimizationFunction.hpp>

/*!
 * \brief Update in the direction of gradient descent.
 * \author tkornuta
 */
template <typename eT=float>
class GradientDescent : public OptimizationFunction<eT> {
public:

	/// Constructor.
	GradientDescent(eT learning_rate_) : learning_rate(learning_rate_){
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		// W = W - alpha * dW.
		for (size_t i=0; i<x_->size(); i++)
			(*x_)[i] -= learning_rate * (*dx_)[i];
	}

protected:
	/// Learning rate.
	eT learning_rate;
};




#endif /* GRADIENTDESCENT_HPP_ */
