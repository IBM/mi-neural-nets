/*!
 * @file: Matrix_extra_meat.hpp
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

#ifndef MAT_EXTRA_MEAT_HPP_
#define MAT_EXTRA_MEAT_HPP_


/*!
 * \brief A serialization operator.
 */
template<typename eT>
template<typename Archive>
void Mat<eT>::serialize(Archive& ar, const unsigned int /* version */)
{
  using boost::serialization::make_nvp;
  using boost::serialization::make_array;

  const uword old_n_elem = n_elem;

  // This is accurate from Armadillo 3.6.0 onwards.
  // We can't use BOOST_SERIALIZATION_NVP() because of the access::rw() call.
  ar & make_nvp("n_rows", access::rw(n_rows));
  ar & make_nvp("n_cols", access::rw(n_cols));
  ar & make_nvp("n_elem", access::rw(n_elem));
  ar & make_nvp("vec_state", access::rw(vec_state));

  // mem_state will always be 0 on load, so we don't need to save it.
  if (Archive::is_loading::value)
  {
    // Don't free if local memory is being used.
    if (mem_state == 0 && mem != NULL && old_n_elem > arma_config::mat_prealloc)
    {
      memory::release(access::rw(mem));
    }

    access::rw(mem_state) = 0;

    // We also need to allocate the memory we're using.
    init_cold();
  }

  ar & make_array(access::rwp(mem), n_elem);
}

/*!
 * \brief Enumerates - sets values of elements to their indices.
 * \author tkornuta
 */
template<typename eT>
void Mat<eT>::enumerate() {
	eT * data_ptr = this->memptr();
#pragma omp parallel for
	for (size_t i = 0; i < n_elem; i++)
		data_ptr[i] = i;
}


#endif /* MAT_EXTRA_MEAT_HPP_ */
