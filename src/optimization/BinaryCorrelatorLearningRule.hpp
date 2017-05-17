/*!
 * @file: BinaryCorrelatorLearningRule.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef BINARYCORRELATORLEARNINGRULE_HPP_
#define BINARYCORRELATORLEARNINGRULE_HPP_

#include <optimization/OptimizationFunction.hpp>

namespace mic {
namespace neural_nets {
namespace learning {

/*!
 * \brief
 * \author tkornuta
 */
/*!
 * \brief Updates according to classical Hebbian rule (wij += ni * x * y).
 * \author tkornuta
 */
template <typename eT=float>
class BinaryCorrelatorLearningRule : public mic::neural_nets::optimization::OptimizationFunction<eT> {
public:
	/*!
	 * Constructor. Sets dimensions, momentum rates (beta1=0.9 and beta2=0.999) and eps(default=1e-8).
	 * @param rows_ Number of rows of the updated matrix/its gradient.
	 * @param cols_ Number of columns of the updated matrix/its gradient.
	 */
	BinaryCorrelatorLearningRule(size_t rows_, size_t cols_) {
		delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
		delta->zeros();
	}

	// Virtual destructor - empty.
	virtual ~BinaryCorrelatorLearningRule() { }


	/*!
	 * Calculates the update according to the hebbian rule.
	 * @param x_ Pointer to the input data matrix.
	 * @param y_ Pointer to the output data matrix.
	 * @param ni_aa Learning rate for P({AA}) (default=0.1).
	 */
	virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT ni_aa = 0.1) {
		// Calculate N_on - sum active bits.
		size_t N_on = y_->sum();
		// Calculate ni_ia = ni_ai.
		eT ni_ia = ni_aa * N_on * (x_->rows() - N_on);

		delta->setZero();

		// Calculate deltas.
		for (size_t b=0; b< (size_t)x_->cols(); b++) { // for batch size.
			for (size_t i=0; i< (size_t)x_->rows(); i++) {
				for (size_t j=0; j< (size_t)y_->rows(); j++) {
					if ((*y_)(j,b) && (*x_)(i,b))
						(*delta)(j,i) += ni_aa;
					else if ((*y_)(j,b) || (*x_)(i,b))
						(*delta)(j,i) -= ni_ia;
				}//: for
			}//: for
		}//: for

		return delta;
	}


protected:
	/// Calculated update.
	mic::types::MatrixPtr<eT> delta;

};

} //: namespace learning
} /* namespace neural_nets */
} /* namespace mic */

#endif /* BINARYCORRELATORLEARNINGRULE_HPP_ */
