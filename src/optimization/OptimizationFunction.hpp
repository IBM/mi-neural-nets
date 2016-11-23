/*!
 * @file: OptimizationFunctions.hpp
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


#ifndef OPTIMIZATIONFUNCTIONS_HPP_
#define OPTIMIZATIONFUNCTIONS_HPP_

#include <types2/MatrixTypes.hpp>

using namespace mic::types2;

/*!
 * \brief Abstract class representing interface to optimization function.
 * \author tkornuta
 * \tparam eT Template type (single/double precision)
 */
template <typename eT=float>
class OptimizationFunction {
public:
	/// Constructor.
	OptimizationFunction () { }

	/// Virtual destructor - empty.
	virtual ~OptimizationFunction () { }

	/// Abstract method responsible for performing the update.
	virtual void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) = 0;

protected:
	/// A temporary variable used for storing the previous value of the input during update.
	mic::types2::MatrixPtr<eT> prev_x;
};



#endif /* OPTIMIZATIONFUNCTIONS_HPP_ */
