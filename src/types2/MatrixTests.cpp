/*!
 * @file: MatrixTests.cpp
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

#include <types2/Matrix.hpp>

/*!
 * Tests whether matrix has proper dimensions (2x5).
 */
TEST(Matrix, Dimensions2x5) {
	// Default sizes of matrices.
	const size_t N = 2;
	const size_t M = 5;

	mic::types2::Matrix<float> nm(N, M);

	ASSERT_EQ(nm.n_rows, N);
	ASSERT_EQ(nm.n_cols, M);
}


/*!
 * Tests matrix serialization.
 */
TEST(Matrix, Serialization) {
	// Default sizes of matrices.
	const size_t N = 2;
	const size_t M = 5;

	mic::types2::Matrix<float> nm(N, M);
	nm.randn();

	const char* fileName = "saved.txt";
	// Save data
	{
		// Create an output archive
		std::ofstream ofs(fileName);
		boost::archive::text_oarchive ar(ofs);
		// Write data
		ar & nm;
//		std::cout << "Saved matrix = " << nm << std::endl;
	}

	// Restore data
	mic::types2::Matrix<float> restored_mat;
	restored_mat.randn();

	{
		// Create and input archive
		std::ifstream ifs(fileName);
		boost::archive::text_iarchive ar(ifs);
		// Load data
		ar & restored_mat;
//		std::cout << "Restored matrix = " << restored_mat << std::endl;
	}

	for (size_t i =0; i< nm.n_elem; i++)
		ASSERT_EQ(nm[i], restored_mat[i]);
}


/*!
 * Tests enumeration.
 */
TEST(Matrix, Enumeration4x5) {
	// Default sizes of matrices.
	const size_t N = 4;
	const size_t M = 5;

	mic::types2::Matrix<float> nm(N, M);
	nm.enumerate();

	for (size_t i =0; i< N*M; i++)
		ASSERT_EQ(nm[i], i);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


