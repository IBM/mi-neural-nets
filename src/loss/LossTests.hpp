/*!
 * @file: LossTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 9, 2016
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

#ifndef LOSSTESTS_HPP_
#define LOSSTESTS_HPP_


#include <gtest/gtest.h>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#include "Loss.hpp"


/*!
 * \brief Test Fixture - two vectors of size 4x1, floats.
 * \author tkornuta
 */
class Vectors4Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Vectors4Float () { }

protected:
	// Sets values
	virtual void SetUp() {
		predicted_y = MAKE_MATRIX_PTR(float, {1.0, 2.0, 3.0, 4.0});
		target_y = MAKE_MATRIX_PTR(float, {2.0, 3.0, 4.0, 5.0});
	}

private:
	// Target y values.
	mic::types2::MatrixPtr<float> target_y;

	// Predicted y values.
	mic::types2::MatrixPtr<float> predicted_y;
};

/*!
 * \brief Test Fixture - three vectors of size 4x1, floats.
 * \author tkornuta
 */
class Vectors4Float3 : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Vectors4Float3 () { }

protected:
	// Sets values
	virtual void SetUp() {
		target_y = MAKE_MATRIX_PTR(float, {.4, .1, .25, .25});
		predicted_y1 = MAKE_MATRIX_PTR(float, {.25, .25, .25, .25});
		predicted_y2 = MAKE_MATRIX_PTR(float, {.4, .1, .1, .4});
	}

private:
	// Target y values.
	mic::types2::MatrixPtr<float> target_y;

	// Predicted y values 1.
	mic::types2::MatrixPtr<float> predicted_y1;
	// Predicted y values 2.
	mic::types2::MatrixPtr<float> predicted_y2;
};


#endif /* LOSSTESTS_HPP_ */
