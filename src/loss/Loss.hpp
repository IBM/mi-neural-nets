/*!
 * @file: Loss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef LOSS_HPP_
#define LOSS_HPP_

#include <types/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace loss {

/*!
 * \brief Abstract class representing a loss function.
 * Defines interfaces.
 * \author tkornuta
 * \tparam dtype Template parameter denoting precision of variables.
 */
template <typename dtype=float>
class Loss {
public:
	/*!
	 * \brief Function calculating loss - abstract.
	 */
	virtual dtype calculateLoss (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) = 0;

	/*!
	 * \brief Function calculating gradient - abstract.
	 */
	virtual mic::types::MatrixPtr<dtype> calculateGradient (mic::types::MatrixPtr<dtype> target_y_, mic::types::MatrixPtr<dtype> predicted_y_) = 0;

};

} //: loss
} //: neural_nets
} //: mic

#endif /* LOSS_HPP_ */
