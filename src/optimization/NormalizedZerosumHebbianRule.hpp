/*!
 * @file: NormalizedZerosumHebbianRule.hpp
 * @Author: Alexis Asseman <alexis.asseman@ibm.com>, Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 30, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef NORMALIZEDZEROSUMHEBBIANRULE_HPP_
#define NORMALIZEDZEROSUMHEBBIANRULE_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace learning {

/*!
 * \brief Updates according to a modified Hebbian rule (wij += ni * f(x, y)) with additional normalization and zero summing for optimal edge detection
 *
 * \author tkornuta/Alexis-Asseman
 */
template <typename eT=float>
class NormalizedZerosumHebbianRule : public mic::neural_nets::optimization::OptimizationFunction<eT> {
public:
	/*!
	 * Constructor. Sets dimensions, momentum rates (beta1=0.9 and beta2=0.999) and eps(default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
    NormalizedZerosumHebbianRule(size_t rows_, size_t cols_) {
        // BUFIX?! Inverted rows and cols
        delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();

	}

	// Virtual destructor - empty.
    virtual ~NormalizedZerosumHebbianRule() { }


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

		// weight += delta;
		(*p_) += (*delta);
        for(size_t row = 0 ; row < (size_t)p_->rows() ; row++){
            // Zero sum
            p_->row(row).array() -= p_->row(row).sum() / p_->cols();
            // Normalize.
            p_->row(row) /= p_->row(row).squaredNorm();
        }
	}

	/*!
	 * Calculates the update according to the hebbian rule.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param learning_rate_ Learning rate (default=0.001).
	 */
	virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_) {
        // delta based on winner take all: Best corresponding kernel gets to learn for each slice
        // delta + alpha * x * y.
//		(*delta) = learning_rate_ * (*y_) * ((*x_).transpose());

        // Winner take all happens in the columns of the output matrix
        Eigen::Array<eT, Eigen::Dynamic, Eigen::Dynamic> wta = (*y_).colwise().maxCoeff();
        // Iterate over the output columns
        for(size_t i = 0 ; i < (size_t)wta.rows() ; i++){
            // Pick the image slice and apply it to best matching filter (ie: row of p['W'])
            delta->row(i) = x_->col(wta(i));

            // Transform the image slice into a filter:
            // Make the vector zero-sum
            delta->row(i).array() -= delta->row(i).sum() / delta->cols();
            // Normalize it
            delta->row(i) /= delta->row(i).squaredNorm();
        }


		return delta;
	}


protected:
	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

};

} //: namespace learning
} /* namespace neural_nets */
} /* namespace mic */

#endif /* NORMALIZEDZEROSUMHEBBIANRULE_HPP_ */
