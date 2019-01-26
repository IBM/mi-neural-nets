/*!
 * @file: NormalizedHebbianRule.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef NORMALIZEDHEBBIANRULE_HPP_
#define NORMALIZEDHEBBIANRULE_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace learning {

/*!
 * \brief
 * \author tkornuta
 */
/*!
 * \brief Updates according to classical Hebbian rule (wij += ni * x * y) with additional normalization
 *
 * \author tkornuta
 */
template <typename eT=float>
class NormalizedHebbianRule : public mic::neural_nets::optimization::OptimizationFunction<eT> {
public:
	/*!
	 * Constructor. Sets dimensions, momentum rates (beta1=0.9 and beta2=0.999) and eps(default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	NormalizedHebbianRule(size_t rows_, size_t cols_) {
		delta = MAKE_MATRIX_PTR(eT, cols_, rows_);
		delta->zeros();

	}

	// Virtual destructor - empty.
	virtual ~NormalizedHebbianRule() { }


	/*!
	 * Updates the weight matrix according to the hebbian rule with normalization (l2 norm).
	 * @param p_ Pointer to the parameter (weight) matrix.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	virtual void update(mic::types::MatrixPtr<eT> p_, mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_ = 0.001) {
		assert(p_->rows() == y_->rows());
		assert(p_->cols() == x_->rows());
		assert(x_->cols() == y_->cols());

		// Calculate the update using hebbian "fire together, wire together".
		mic::types::MatrixPtr<eT> delta = calculateUpdate(x_, y_, learning_rate_);

		std::cout<<"x: " << (*x_).transpose() << std::endl;
		std::cout<<"y: " << (*y_).transpose() << std::endl;
		std::cout<<"przed update: " << (*p_)[0] << std::endl;
		// weight += delta;
		(*p_) += (*delta);
		std::cout<<"przed normalizacją: " << (*p_)[0] << std::endl;
		// Normalize.
		(*p_) /= p_->squaredNorm();
		std::cout<<"po normalizacji: " << (*p_)[0] << std::endl;
	}

	/*!
	 * Calculates the update according to the hebbian rule.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_) {
		// delta + alpha * x * y.
		(*delta) = learning_rate_ * (*y_) * ((*x_).transpose());

		// Return delta normalized by batch size.
		(*delta) /= x_->cols();
		return delta;
	}


protected:
	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

};

} //: namespace learning
} /* namespace neural_nets */
} /* namespace mic */

#endif /* NORMALIZEDHEBBIANRULE_HPP_ */
