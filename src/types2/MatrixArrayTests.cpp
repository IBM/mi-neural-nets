/*!
 * @file: MatrixArrayTests.cpp
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


#include <gtest/gtest.h>

#include <types2/MatrixArray.hpp>

/*!
 * Tests whether matrices have proper dimensions.
 */
TEST(MatrixArray, Dimensions2x3x4) {
	// Default sizes of matrices.
	const size_t N = 2;
	const size_t M = 3;
	const size_t B = 4;

	// Test MatrixArray operations.
	mic::types2::MatrixArray<double> ma("test_array");

	// Two different methods of adding matrices to array.
	ma.add (
				{
					std::make_tuple ( "x", M, B ), 	// input
					std::make_tuple ( "y", N, B ) 		// output
				} );

	ma.add (std::make_tuple ( "w", N, M ));

	ASSERT_EQ(ma["x"]->n_rows, M);
	ASSERT_EQ(ma["x"]->n_cols, B);
	ASSERT_EQ(ma["y"]->n_rows, N);
	ASSERT_EQ(ma["y"]->n_cols, B);
	ASSERT_EQ(ma["w"]->n_rows, N);
	ASSERT_EQ(ma["w"]->n_cols, M);
}


/*!
 * Tests matrix array serialization.
 */
TEST(MatrixArray, Serialization) {
	// Default sizes of matrices.
	const size_t N = 2;
	const size_t M = 3;

	// Test MatrixArray operations.
	mic::types2::MatrixArray<double> ma1("test_array");

	//  Add single matrix.
	ma1.add (std::make_tuple ( "w", N, M ));
	ma1["w"]->randn();

//	std::cout<<"Saved MatrixArray = " << ma1;
	const char* fileName = "saved.txt";
	// Save data.
	{
		// Create an output archive.
		std::ofstream ofs(fileName);
		boost::archive::text_oarchive ar(ofs);
		// Write data
		ar & ma1;
	}

	// Restore data.
	mic::types2::MatrixArray<double> restored_ma;
	{
		// Create and input archive.
		std::ifstream ifs(fileName);
		boost::archive::text_iarchive ar(ifs);
		// Load data.
		ar & restored_ma;
//		std::cout << "Restored MatrixArray = " << restored_ma << std::endl;
	}

	for (size_t i =0; i< ma1["w"]->n_elem; i++)
		ASSERT_EQ((*ma1["w"])[i], (*restored_ma["w"])[i]);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


