/*!
 * @file: OptimizationFunctions.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 11, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */


#ifndef OPTIMIZATIONFUNCTIONS_HPP_
#define OPTIMIZATIONFUNCTIONS_HPP_

#include <types/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace optimization {

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
	virtual void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_) = 0;

protected:
	/// A temporary variable used for storing the previous value of the input during update.
	mic::types::MatrixPtr<eT> prev_x;
};


} //: optimization
} //: neural_nets
} //: mic


#endif /* OPTIMIZATIONFUNCTIONS_HPP_ */
