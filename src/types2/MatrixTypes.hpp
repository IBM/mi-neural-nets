/*!
 * \file MatrixTypes.hpp
 * \brief Contains definition of basic matrix datatypes derived from Eigen.
 * \author tkornut
 * \date Feb 5, 2016
 */

#ifndef SRC_TYPES_MATRIXTYPES_HPP_
#define SRC_TYPES_MATRIXTYPES_HPP_

#include <types2/Matrix.hpp>


namespace mic {
namespace types2 {

/*!
 * \brief Typedef for a shared pointer to template-typed dynamic matrices.
 * \author tkornuta
 * \tparam eT template parameter indicating data type (int/float/double)
 */
template<typename eT>
using MatrixPtr = typename std::shared_ptr< mic::types2::Matrix<eT> >;

/*!
 * \brief Macro for initialization of values of vector pointer.
 * \author tkornuta
 */
#define MAKE_MATRIX_PTR(eT, ...) std::make_shared<mic::types2::Matrix<eT> >( mic::types2::Matrix<eT> (__VA_ARGS__))


}//: namespace types2
}//: namespace mic


#endif /* SRC_TYPES_MATRIXTYPES_HPP_ */
