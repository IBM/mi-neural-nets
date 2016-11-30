/*!
 * @file: ArtificialLandscapes.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 11, 2016
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

#ifndef ARTIFICIALLANDSCAPES_HPP_
#define ARTIFICIALLANDSCAPES_HPP_

#include <types2/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

/*!
 * \brief Abstract class representing interface to a differentiable function.
 * \author tkornuta
 */
template <typename eT=float>
class DifferentiableFunction {
public:
	/// Constructor. Asserts whether dimensions must be > 0.
	DifferentiableFunction (size_t dims_) : dims(dims_) {
		assert (dims > 0);
	}

	/// Virtual destructor - empty.
	virtual ~DifferentiableFunction () { }

	// Abstract method responsible for calculation of the function value.
	virtual eT calculateValue(mic::types2::MatrixPtr<eT> x_) = 0;

	// Abstract method responsible for calculation of the a gradient in a given point.
	virtual mic::types2::MatrixPtr<eT> calculateGradient(mic::types2::MatrixPtr<eT> x_) = 0;

	/// Returns the vector of arguments being the function minimum.
	mic::types2::MatrixPtr<eT> minArguments () { return min_arguments; }

	/// Returns min value of the function.
	eT minValue() { return min_value; }

protected:
	/// Number of function dimensions (input variables).
	size_t dims;

	/// vector of arguments for which the function has a minimum.
	mic::types2::MatrixPtr<eT> min_arguments;

	/// Minimal value.
	eT min_value;
};



/*!
 * \brief A sphere function - square function generalized to n dimensions.
 * \author tkornuta
 */
template <typename eT=float>
class SphereFunction : public DifferentiableFunction<eT> {
public:

	/// Constructor.
	SphereFunction(size_t dims_) : DifferentiableFunction<eT>(dims_) {
		// Set minimum.
		this->min_arguments = MAKE_MATRIX_PTR(eT, this->dims, 1);
		this->min_arguments->zeros();
		this->min_value = 0;
	}

	/*!
	 * Calculates value of a function for a given point.
	 */
	eT calculateValue(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);
		// Calculate sum of x^2.
		eT val = 0;
		for (size_t i=0; i<this->dims; i++)
			val += (*x_)[i] * (*x_)[i];
		return val;
	}

	/*!
	 * Calculates gradient of a function in a given point.
	 */
	mic::types2::MatrixPtr<eT> calculateGradient(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);

		// Calculate gradients.
		mic::types2::MatrixPtr<eT> dx = MAKE_MATRIX_PTR(eT, this->dims, 1);
		for (size_t i=0; i<this->dims; i++)
			(*dx)[i] = 2 * (*x_)[i];

		return dx;
	}
};


/*!
 * \brief 2D Beale's function.
 * \author tkornuta
 */
template <typename eT=float>
class Beale2DFunction  : public DifferentiableFunction<eT> {
public:

	/// Constructor.
	Beale2DFunction() : DifferentiableFunction<eT>(2) {
		// Set minimum.
		this->min_arguments = MAKE_MATRIX_PTR(eT, this->dims, 1);
		(*this->min_arguments)[0] = 3;
		(*this->min_arguments)[1] = 0.5;
		this->min_value = 0.0;
	}

	/*!
	 * Calculates value of a function for a given point.
	 */
	eT calculateValue(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);

		// Calculate value.
		eT x = (*x_)[0];
		eT y = (*x_)[1];

		eT a = (1.5 - x + x*y);
		eT b = (2.25 - x + x * y * y);
		eT c = (2.625 - x  + x * y * y * y);
		return a*a + b*b + c*c;
	}

	/*!
	 * Calculates gradient of a function in a given point.
	 */
	mic::types2::MatrixPtr<eT> calculateGradient(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);

		mic::types2::MatrixPtr<double> dx = MAKE_MATRIX_PTR(eT, this->dims, 1);

		// Calculate gradients.
		eT x = (*x_)[0];
		eT y = (*x_)[1];

		eT ax = 2*(1.5 - x + x*y) * (-1 + y);
		eT bx = 2*(2.25 - x + x * y * y) * ( - 1 + y * y);
		eT cx = 2*(2.625 - x  + x * y * y * y) * ( - 1 + y * y * y);

		eT ay = 2*(1.5 - x + x*y) * (x);
		eT by = 2*(2.25 - x + x * y * y) * (2*x*y);
		eT cy = 2*(2.625 - x  + x * y * y * y) * (3*x*y*y);

		(*dx)[0] = ax + bx +cx;
		(*dx)[1] = ay + by +cy;

		return dx;
	}
};




/*!
 * \brief 2D Rosenbrock function.
 * \author tkornuta
 */
template <typename eT=float>
class Rosenbrock2DFunction  : public DifferentiableFunction<eT> {
public:

	/// Constructor.
	Rosenbrock2DFunction(eT a_ = 1, eT b_ = 100) : DifferentiableFunction<eT>(2), a(a_), b(b_) {
		// Set minimum.
		this->min_arguments = MAKE_MATRIX_PTR(eT, this->dims, 1);
		(*this->min_arguments)[0] = a;
		(*this->min_arguments)[1] = a*a;
		this->min_value = 0.0;
	}

	/*!
	 * Calculates value of a function for a given point.
	 */
	eT calculateValue(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);

		// Calculate value.
		eT x = (*x_)[0];
		eT y = (*x_)[1];

		eT p1 = (a - x) * (a - x);
		eT p2 = b * (y - x * x) * (y - x * x);
//		std::cout << "p1 = " << p1 << " p2 = " << p2 << std::endl;
		return p1 + p2;
	}

	/*!
	 * Calculates gradient of a function in a given point.
	 */
	mic::types2::MatrixPtr<eT> calculateGradient(mic::types2::MatrixPtr<eT> x_) {
		assert(x_->size() == this->dims);

		mic::types2::MatrixPtr<eT> dx = MAKE_MATRIX_PTR(eT, this->dims, 1);

		// Calculate gradients.
		eT x = (*x_)[0];
		eT y = (*x_)[1];

		eT p1x = -2 * (a - x);
		eT p2x = 2 * b * (y - x * x) * (-2 * x) ;
//		std::cout << "p1x = " << p1x << " p2x = " << p2x << std::endl;

		eT p1y = 0;
		eT p2y = 2 * b * (y - x * x) ;
//		std::cout << "p1y = " << p1y << " p2y = " << p2y << std::endl;

		(*dx)[0] = p1x + p2x;
		(*dx)[1] = p1y + p2y;

		return dx;
	}

private:
	/// Coefficients.
	eT a, b;
};


} //: optimization
} //: neural_nets
} //: mic


#endif /* ARTIFICIALLANDSCAPES_HPP_ */
