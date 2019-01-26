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
 * @file: LossTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 9, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
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
class Vectors4x1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Vectors4x1Float () {
		target_y = MAKE_MATRIX_PTR(float, 4, 1);
		predicted_y = MAKE_MATRIX_PTR(float, 4, 1);
	}

protected:
	// Sets values.
	virtual void SetUp() {
		(*target_y)  << 2.0, 3.0, 4.0, 5.0;
		(*predicted_y) << 1.0, 2.0, 3.0, 4.0;
	}

private:
	// Target y values.
	mic::types::MatrixPtr<float> target_y;

	// Predicted y values.
	mic::types::MatrixPtr<float> predicted_y;
};

/*!
 * \brief Test Fixture - two predictions of size 4x1, floats.
 * \author tkornuta
 */
class Vectors4x1Float2 : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Vectors4x1Float2 () {
		target_y = MAKE_MATRIX_PTR(float, 4,1);
		predicted_y1 = MAKE_MATRIX_PTR(float, 4, 1);
		predicted_y2 = MAKE_MATRIX_PTR(float, 4, 1);
	}

protected:
	// Sets values.
	virtual void SetUp() {
		(*target_y) << .4, .1, .25, .25;
		(*predicted_y1) << .25, .25, .25, .25;
		(*predicted_y2) << .4, .1, .1, .4;
	}

private:
	// Target y values.
	mic::types::MatrixPtr<float> target_y;

	// Predicted y values 1.
	mic::types::MatrixPtr<float> predicted_y1;
	// Predicted y values 2.
	mic::types::MatrixPtr<float> predicted_y2;
};


/*!
 * \brief Test Fixture - two vectors of size 3x2, floats.
 * \author tkornuta
 */
class Vectors3x2Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Vectors3x2Float () {
		target_y = MAKE_MATRIX_PTR(float, 3, 2);
		predicted_y = MAKE_MATRIX_PTR(float, 3, 2);
	}

protected:
	// Sets values.
	virtual void SetUp() {
		(*target_y)  << 0.1, 0.2, 0.3, 0.5, 0.6, 0.1;
		(*predicted_y) << 0.2, 0.2, 0.4, 0.4, 0.5, 0.6;
	}

private:
	// Target y values.
	mic::types::MatrixPtr<float> target_y;

	// Predicted y values.
	mic::types::MatrixPtr<float> predicted_y;
};


#endif /* LOSSTESTS_HPP_ */
