/*!
 * @file: FullyConnectedLayer.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 4, 2016
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

#include <random>

#include "loss/Loss.hpp"

namespace mic {
namespace neural_nets {
namespace feed_forward {

/*!
 * \brief Class representing a fully connected layer.
 * All weights/biases/inputs/outputs are stored in std::vectors.
 * Layer incapable of batch processing (batch size = 1).
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class FullyConnectedLayer {
public:

	/*!
	 * Constructor. Sets the dimensions and shapes matrices accordingly.
	 */
	FullyConnectedLayer(std::size_t input_size_=1, std::size_t output_size_=1) {
		// Store sizes.
		input_size = input_size_;
		output_size = output_size_;
		// Initialize variables for forward pass.
		x = MAKE_MATRIX_PTR (eT, input_size, 1);
		y = MAKE_MATRIX_PTR (eT, output_size, 1);
		W = MAKE_MATRIX_PTR (eT, output_size * input_size, 1);
		b = MAKE_MATRIX_PTR (eT, output_size, 1);

		// Initialize variables for backward pass - gradients.
		dx = MAKE_MATRIX_PTR (eT, input_size, 1);
		dy = MAKE_MATRIX_PTR (eT, output_size, 1);
		dW = MAKE_MATRIX_PTR (eT, output_size * input_size, 1);
		db = MAKE_MATRIX_PTR (eT, output_size, 1);

		// Initialize weights.
		eT range = sqrt(6.0 / eT(output_size + input_size));

		// Initialize random number generator with normal distribution.
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<eT> dist(-range, range);

		// Initialize W.
		for (size_t i = 0; i < output_size * input_size; i++) {
			(*W)[i] = (eT)dist(rd);
		}

		// Initialize b.
		for(size_t row=0; row < output_size; row++)
			(*b)[row] = 0.0;
	}

	/*!
	 * Forwards the activations of the neural network.
	 */
	mic::types2::MatrixPtr<eT> forward(mic::types2::MatrixPtr<eT> x_) {
		// Memorize input.
		for(size_t col=0; col < input_size; col++)
			(*x)[col] = (*x_)[col];

		// Calculate output: y = W^T *x + b.
		for(size_t row=0; row < output_size; row++) {
			(*y)[row] = (*b)[row];
			for(size_t col=0; col < input_size; col++)
				(*y)[row] += (*W)[row*input_size + col] * (*x)[col];
		}//: for row

		// Return result.
		return y;
	}

	/*!
	 * Backward pass - backpropagation.
	 */
	mic::types2::MatrixPtr<eT> backward(mic::types2::MatrixPtr<eT> dy_) {
		// Memorize input - gradient.
		for(size_t col=0; col < output_size; col++)
			(*dy)[col] = (*dy_)[col];

		// Calculate dx gradient: dx = W * dy;
		for(size_t col=0; col < input_size; col++) {
			(*dx)[col] = 0;
			for(size_t row=0; row < output_size; row++) {
				(*dx)[col] += (*W)[row*input_size + col] * (*dy)[row];
			}
		}//: for col

		// Calculate db gradient: db = dy;
		for(size_t row=0; row < output_size; row++)
			(*db)[row] =  (*dy)[row];

		// Calculate dW gradient: dW = dy * x^T;
		for(size_t row=0; row < output_size; row++)
			for(size_t col=0; col < input_size; col++)
				(*dW)[row*input_size + col] = (*dy)[row] * (*x)[col];

		// Return result.
		return dx;
	}

	template<typename loss>
	mic::types2::MatrixPtr<eT> calculateNumericalGradient(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> target_y_, mic::types2::MatrixPtr<eT> param_, loss loss_, eT delta_) {
		// Allocate memory.
		mic::types2::MatrixPtr<eT> nGrad = MAKE_MATRIX_PTR(eT, param_->size(), 1);
		for (size_t i=0; i<param_->size(); i++) {
			// Add delta.
			(*param_)[i] += delta_;
			// Calculate loss.
			eT p = loss_.calculateLoss(forward(x_), target_y_);
			// Substract delta.
			(*param_)[i] -= 2*delta_;
			// Calculate loss.
			eT m = loss_.calculateLoss(forward(x_), target_y_);

			// Store numerical gradient.
			(*nGrad)[i] = (p-m)/(2*delta_);
			// Set original value.
			(*param_)[i] += delta_;

		}//: for
		return nGrad;
	}

private:

	/// Size of input vector.
	std::size_t input_size;

	/// Size of output vector.
	std::size_t output_size;

	/// Input vector of size [input_size x 1] (no batch).
	mic::types2::MatrixPtr<eT> x;

	/// Input vector of size [output_size x 1] (no batch)
	mic::types2::MatrixPtr<eT> y;

	// Weights of layer [output_size * input_size]
	mic::types2::MatrixPtr<eT> W;

	// Biases of layer of size [output_size * 1]
	mic::types2::MatrixPtr<eT> b;

	// Gradient dx of size [input_size x 1]
	mic::types2::MatrixPtr<eT> dx;

	// dy of size [output_size x 1].
	mic::types2::MatrixPtr<eT> dy;

	// Gradient dW of size [output_size * input_size]
	mic::types2::MatrixPtr<eT> dW;

	// Gradient db of size [output_size x 1]
	mic::types2::MatrixPtr<eT> db;

};

} //: feed_forward
} //: neural_nets
} //: mic


