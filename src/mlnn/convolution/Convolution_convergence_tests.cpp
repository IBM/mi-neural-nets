/*!
 * @file: Convolution_convergence_tests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 27, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */


#include "Convolution_tests.hpp"

namespace mic { namespace neural_nets { namespace unit_tests {

/*!
 * \brief Numerical gradient test of all parameters for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Double, NumericalGradientCheck) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dW00 = MAKE_MATRIX_PTR(double, *layer.g["W00"]);
	mic::types::MatrixPtr<double> dW01 = MAKE_MATRIX_PTR(double, *layer.g["W01"]);
	mic::types::MatrixPtr<double> dW10 = MAKE_MATRIX_PTR(double, *layer.g["W10"]);
	mic::types::MatrixPtr<double> dW11 = MAKE_MATRIX_PTR(double, *layer.g["W11"]);
	mic::types::MatrixPtr<double> db = MAKE_MATRIX_PTR(double, *layer.g["b"]);

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nW00 = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p["W00"], loss, delta);
	mic::types::MatrixPtr<double> nW01 = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p["W01"], loss, delta);
	mic::types::MatrixPtr<double> nW10 = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p["W10"], loss, delta);
	mic::types::MatrixPtr<double> nW11 = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p["W11"], loss, delta);
	mic::types::MatrixPtr<double> nb = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p["b"], loss, delta);

	// Compare gradients.
	double eps = 1e-8;
/*	std::cout << "(*dW00)= " << (*dW00) << " (*nW00)= " << (*nW00) << std::endl;
	std::cout << "(*dW01)= " << (*dW01) << " (*nW01)= " << (*nW01) << std::endl;
	std::cout << "(*dW10)= " << (*dW10) << " (*nW10)= " << (*nW10) << std::endl;
	std::cout << "(*dW11)= " << (*dW11) << " (*nW11)= " << (*nW11) << std::endl;
	std::cout << "(*db0)= " << (*db)(0) << " (*nb0)= " << (*nb)(0) << std::endl;
	std::cout << "(*db1)= " << (*db)(1) << " (*nb1)= " << (*nb)(1) << std::endl;*/
	EXPECT_LE( fabs((*dW00)(0) - (*nW00)(0)), eps) << "Too big difference between dW and numerical dW";
	EXPECT_LE( fabs((*dW10)(0) - (*nW10)(0)), eps) << "Too big difference between dW and numerical dW";
	EXPECT_LE( fabs((*dW01)(0) - (*nW01)(0)), eps) << "Too big difference between dW and numerical dW";
	EXPECT_LE( fabs((*dW11)(0) - (*nW11)(0)), eps) << "Too big difference between dW and numerical dW";
	EXPECT_LE( fabs((*db)(0) - (*nb)(0)), eps) << "Too big difference between dW and numerical dW";
	EXPECT_LE( fabs((*db)(1) - (*nb)(1)), eps) << "Too big difference between dW and numerical dW";
}



/*!
 * Checks whether the forward is working for layer of input size 2x2x2 and with filter bank of 2 filters of size 1x1 with stride 1.
 * \author tkornuta
 */
TEST_F(Conv2x2x2Filter2x1x1s1Double, DISABLED_Convergence) {

	std::cout<<" W00 = " << (*layer.p["W00"]) <<std::endl;
	std::cout<<" W01 = " << (*layer.p["W01"]) <<std::endl;
	std::cout<<" W10 = " << (*layer.p["W10"]) <<std::endl;
	std::cout<<" W11 = " << (*layer.p["W11"]) <<std::endl;
	for (size_t i=0; i<2; i++)
		std::cout<<" b"<<i<< " = "<< (*layer.p["b"])[i] << std::endl;

	// Training.
	size_t iteration = 0;
	while (iteration < 10) {
		std::cout<<"[" << iteration << "]" << std::endl;

		// Calculate gradients.
		mic::types::MatrixPtr<double> predicted_y = layer.forward(x);
		std::cout<< " ***  Loss = " << loss.calculateMeanLoss(target_y, predicted_y) <<std::endl;
		mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
		mic::types::MatrixPtr<double> dx = layer.backward(dy);

		// Apply the changes - according to the optimization function.
		layer.update(0.001, 0.0);

		std::cout<<" W00 = " << (*layer.p["W00"]) <<std::endl;
		std::cout<<" W01 = " << (*layer.p["W01"]) <<std::endl;
		std::cout<<" W10 = " << (*layer.p["W10"]) <<std::endl;
		std::cout<<" W11 = " << (*layer.p["W11"]) <<std::endl;
		for (size_t i=0; i<2; i++)
			std::cout<<" b"<<i<< " = "<< (*layer.p["b"])[i] << std::endl;

		iteration++;
	}//: while
}


} } } //: namespaces
