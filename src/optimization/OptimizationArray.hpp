/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * @file: OptimizationArray.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 27, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef OPTIMIZATIONARRAY_HPP_
#define OPTIMIZATIONARRAY_HPP_

#include <string>
#include <map>
#include <stdio.h>


// Forward declaration of class boost::serialization::access
namespace boost {
namespace serialization {
class access;
}//: serialization
}//: access

namespace mic {
namespace neural_nets {
namespace optimization {


/*!
 * 	\brief A dynamic array of optimization functions (a hash-table).
 * 	\author tkornuta
 */
template<typename T>
class OptimizationArray {
public:

	/*!
	 * Default empty constructor.
	 */
	OptimizationArray() = default;

	/*!
	 * Adds an optimization function to the array.
	 * @param name_ Name of the matrix.
	 * @param input_ Input length.
	 * @param output_ Output length.
	 * @tparam opT Optimization function type.
	 */
	template <typename opT>
	void add( std::string name_, size_t rows_, size_t cols_){
		keys_map[name_] = functions.size();
		functions.push_back ( std::make_shared< opT > ( opT  ( rows_, cols_ ) ) );

	}

	template <typename opT>
	void add( std::string name_, std::shared_ptr<opT> opt_){
		keys_map[name_] = functions.size();
		functions.push_back ( opt_ );
	}

	void clear(){
		keys_map.clear();
		functions.clear();
	}


	/*!
	 * Returns the matrix with given number.
	 * @param number_ Number of the matrix.
	 * @return Pointer to a matrix.
	 */
	std::shared_ptr<mic::neural_nets::optimization::OptimizationFunction<T> > & operator[] ( size_t number_ ) {
		// TODO: throw exception when out of the scope?

		return functions[number_];

	}

	/*!
	 * Returns the matrix with given key (id).
	 * @param number_ Matrix key.
	 * @return Pointer to a matrix.
	 */
	std::shared_ptr<mic::neural_nets::optimization::OptimizationFunction<T> > & operator[] ( char key_ ) {
		// TODO: throw exception when out of the scope?

		return ( *this ) [std::string ( 1, key_ )];

	}

	/*!
	 * Returns the matrix with given key (id).
	 * @param number_ Matrix key.
	 * @return Pointer to a matrix.
	 */
	std::shared_ptr<mic::neural_nets::optimization::OptimizationFunction<T> > & operator[] ( std::string key ) {

		if ( keys_map.find ( key ) == keys_map.end() )
			std::cout << "Warning !!! " <<
					  "::[] - key not found:" << key << std::endl;

		// TODO: throw exception when out of the scope?

		return functions[keys_map[key]];

	}

	/*!
	 * Stream operator enabling to print the array.
	 * @param os_ Ostream object.
	 * @param obj_ Array object.
	 */
	friend std::ostream& operator<<(std::ostream& os_, const mic::neural_nets::optimization::OptimizationArray<T> & obj_) {
		// Display name
		for (auto& i: obj_.keys_map) {
			// Display elements.
			os_ << "[" << i.first << "] \n";
//			os_ << (*obj_.functions[i.second]) << std::endl;
		}
		return os_;
	}

	/*!
	 * Returns keys map.
	 */
	std::map<std::string, size_t> keys() {
		return keys_map;
	}

	/*!
	 * Returns the size of array.
	 */
	std::string size() {
		return functions.size();
	}

protected:
	/// Vector of pointers of optimization functions.
	std::vector<std::shared_ptr< mic::neural_nets::optimization::OptimizationFunction<T> > > functions;


	/// Vector of keys of consecutive functions in the array.
	std::map<std::string, size_t> keys_map;

private:
	// Friend class - required for using boost serialization.
    friend class boost::serialization::access;

};


} /* namespace optimzation */
} /* namespace neural_nets */
} /* namespace mic */


#endif /* OPTIMIZATIONARRAY_HPP_ */
