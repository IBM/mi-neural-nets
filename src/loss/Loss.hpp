/*!
 * @file: Loss.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 8, 2016
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

#ifndef LOSS_HPP_
#define LOSS_HPP_

#include <types2/MatrixTypes.hpp>

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
	virtual dtype calculateLoss (mic::types2::MatrixPtr<dtype> predicted_y_, mic::types2::MatrixPtr<dtype> target_y_) = 0;

	/*!
	 * \brief Function calculating gradient - abstract.
	 */
	virtual mic::types2::MatrixPtr<dtype> calculateGradient (mic::types2::MatrixPtr<dtype> predicted_y_, mic::types2::MatrixPtr<dtype> target_y_) = 0;

};

} //: loss
} //: neural_nets
} //: mic

#endif /* LOSS_HPP_ */
