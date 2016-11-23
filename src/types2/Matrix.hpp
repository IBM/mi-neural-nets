/*!
 * @file: Matrix.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 22, 2016
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

#ifndef MATRIX_HPP_
#define MATRIX_HPP_


#include <fstream>
// Include headers that implement a archive in simple text format
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#define ARMA_EXTRA_MAT_PROTO types2/Matrix_extra_bones.hpp
#define ARMA_EXTRA_MAT_MEAT types2/Matrix_extra_meat.hpp

#include <armadillo>
#include <memory> // shared_ptr


// Forward declaration of class boost::serialization::access
namespace boost {
namespace serialization {
class access;
}//: serialization
}//: access


namespace mic {
namespace types2 {

/*
 * \brief Typedef for a shared pointer to template-typed dynamic matrices.
 * \author tkornuta
 * \tparam eT template parameter indicating data type (int/float/double)
 */
template<typename eT>
using Matrix = typename arma::Mat<eT>;



}//: namespace types2
}//: namespace mic


// Set new version - mic::types2::Matrix is now based on Armadillo, not Eigen!
BOOST_CLASS_VERSION(mic::types2::Matrix<bool>, 2)
BOOST_CLASS_VERSION(mic::types2::Matrix<short>, 2)
BOOST_CLASS_VERSION(mic::types2::Matrix<int>, 2)
BOOST_CLASS_VERSION(mic::types2::Matrix<long>, 2)
BOOST_CLASS_VERSION(mic::types2::Matrix<float>, 2)
BOOST_CLASS_VERSION(mic::types2::Matrix<double>, 2)


#endif /* MATRIX_HPP_ */
