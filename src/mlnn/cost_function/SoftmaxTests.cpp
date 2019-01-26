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
 * @file: SoftmaxTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Apr 12, 2017
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "SoftmaxTests.hpp"



/*!
 * Tests forward.
 */
TEST_F(Softmax4x1Float, Forward) {
	double eps = 1e-5;

	// Forward pass.
	mic::types::MatrixPtr<float> y = layer.forward(input_x);

	for (size_t i=0; i<4; i++)
		ASSERT_LE( fabs((*y)[i] - (*output_y)[i]), eps) << "Difference at position i=" << i << " where " << (*y)[i] << " and should be " << (*output_y)[i];
}



/*!
 * Numerical gradient test dW, size of layer is 2x3.
 */
/*TEST_F(Softmax4x1Float, NumericalGradientCheck_dx) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(input_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dx = MAKE_MATRIX_PTR(double, *layer.g["x"]);

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nx = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(const_x, target_y, --, loss, delta);

	// Compare gradients.
	double eps = 1e-8;
	for (size_t i=0; i<(size_t)dW->size(); i++){
//		std::cout << "i=" << i << " (*dW)[i]= " << (*dW)[i] << " (*nW)[i]= " << (*nW)[i] << std::endl;
		EXPECT_LE( fabs((*dW)[i] - (*nW)[i]), eps) << "Too big difference between dW and numerical dW at position i=" << i;
	}//: for
}*/


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
