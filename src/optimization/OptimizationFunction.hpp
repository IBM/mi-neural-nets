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
	/*!
	 * Constructor. Remembers dimensions.
	 */
	OptimizationFunction () { }

	/// Virtual destructor - empty.
	virtual ~OptimizationFunction () { }

	/*!
	 * Abstract method responsible for performing the update.
	 * @param x_ Pointer to the current matrix.
	 * @param dx_ Pointer to current gradient of that matrix.
	 * @param learning_rate_ Learning rate.
	 */
	virtual void update(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> dx_, eT learning_rate_) = 0;

};


} //: optimization
} //: neural_nets
} //: mic


#endif /* OPTIMIZATIONFUNCTIONS_HPP_ */
