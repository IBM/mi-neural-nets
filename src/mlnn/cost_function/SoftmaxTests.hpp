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
 * @file: SoftmaxTests.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 12, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef SOFTMAXTESTS_HPP_
#define SOFTMAXTESTS_HPP_

#include <gtest/gtest.h>

// Redefine word "public" so every class field/method will be accessible for tests.
#define private public
#define protected public
#include <mlnn/cost_function/Softmax.hpp>
#include <loss/SquaredErrorLoss.hpp>

/*!
 * \brief Test Fixture - 4x1 softmax layer.
 * \author tkornuta
 */
class Softmax4x1Float : public ::testing::Test {
public:
	// Constructor. Sets layer size.
	Softmax4x1Float () : layer(4) {
		input_x = MAKE_MATRIX_PTR(float, 4, 1);
		output_y = MAKE_MATRIX_PTR(float, 4, 1);
		target_y = MAKE_MATRIX_PTR(float, 4, 1);
	}

protected:
	// Sets test values.
	virtual void SetUp() {
		(*input_x) << 1, -1, -5, 0.5;
		(*output_y) << 0.573281, 0.0775852, 0.00142102, 0.347713;
		(*target_y) << 0.5, 0.25, 0.15, 0.10;
	}

private:
	// Object to be tested.
	mic::mlnn::cost_function::Softmax<float> layer;

	// Test input x - used in forward pass.
	mic::types::MatrixPtr<float> input_x;

	// Output y values.
	mic::types::MatrixPtr<float> output_y;

	// Target y values.
	mic::types::MatrixPtr<float> target_y;
};



#endif /* SOFTMAXTESTS_HPP_ */
