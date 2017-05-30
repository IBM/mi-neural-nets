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
TEST_F(Conv2x2x2Filter2x1x1s1Double, DISABLED_NumericalGradientCheck) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
	layer.backward(dy);

	// Get differentiable parameters.
	std::map<std::string, size_t> keys = layer.p.keys();

	// Store resulting gradients - make a copy!
	mic::types::MatrixArray<double> grads;
	for (auto& i: keys) {
		grads.add(i.first, MAKE_MATRIX_PTR(double, *layer.g[i.first]));
		//std::cout << "** d" << i.first << " = \n" << *(grads[i.first]) << std::endl;
	}//: for

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixArray<double> ngrads;
	for (auto& i: keys) {
		mic::types::MatrixPtr<double> ngrad = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p[i.first], loss, delta);
		// Store gradient - make a copy as well.
		ngrads.add(i.first, MAKE_MATRIX_PTR(double, *ngrad));
		//std::cout << "** n" << i.first << " = \n" << *(ngrads[i.first]) << std::endl;
	}//: for


	// Compare gradients.
	double eps = 1e-8;
	for (auto& i: keys) {
		// Get gradient and numerical gradient.
		mic::types::MatrixPtr<double> grad = grads[i.first];
		mic::types::MatrixPtr<double> ngrad = ngrads[i.first];
		// Iterate through params.
		for (size_t j=0; j<(size_t)grad->size(); j++){
			//std::cout << "param " << i.first << " j=" << j << " (*grad)[j]= " << (*grad)[j] << " (*ngrad)[j]= " << (*ngrad)[j] << std::endl;
			EXPECT_LE( fabs((*grad)[j] - (*ngrad)[j]), eps) << "Too big difference between grad and numerical grad of " << i.first << " at position j=" << j;
		}//: for

	}//: for

}



/*!
 * \brief Numerical gradient test of all parameters for layer of input size 28x28x1 and with filter bank of 2 filters of size 28x28 with stride 1, double.
 * \author tkornuta
 */
TEST_F(Conv28x28x1Filter2x28x28s1Double, DISABLED_NumericalGradientCheck) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
	layer.backward(dy);

	// Get differentiable parameters.
	std::map<std::string, size_t> keys = layer.p.keys();

	// Store resulting gradients - make a copy!
	mic::types::MatrixArray<double> grads;
	for (auto& i: keys) {
		grads.add(i.first, MAKE_MATRIX_PTR(double, *layer.g[i.first]));
		//std::cout << "** d" << i.first << " = \n" << *(grads[i.first]) << std::endl;
	}//: for

	// Calculate numerical gradients.
	double delta = 1e-7;
	mic::types::MatrixArray<double> ngrads;
	for (auto& i: keys) {
		mic::types::MatrixPtr<double> ngrad = layer.calculateNumericalGradient<mic::neural_nets::loss::SquaredErrorLoss<double> >(x, target_y, layer.p[i.first], loss, delta);
		// Store gradient - make a copy as well.
		ngrads.add(i.first, MAKE_MATRIX_PTR(double, *ngrad));
		//std::cout << "** n" << i.first << " = \n" << *(ngrads[i.first]) << std::endl;
	}//: for


	// Compare gradients.
	double eps = 1e-8;
	for (auto& i: keys) {
		// Get gradient and numerical gradient.
		mic::types::MatrixPtr<double> grad = grads[i.first];
		mic::types::MatrixPtr<double> ngrad = ngrads[i.first];
		// Iterate through params.
		for (size_t j=0; j<(size_t)grad->size(); j++){
			//std::cout << "param " << i.first << " j=" << j << " (*grad)[j]= " << (*grad)[j] << " (*ngrad)[j]= " << (*ngrad)[j] << std::endl;
			EXPECT_LE( fabs((*grad)[j] - (*ngrad)[j]), eps) << "Too big difference between grad and numerical grad of " << i.first << " at position j=" << j;
		}//: for

	}//: for

}



/*!
 * Checks convergence of layer of input size 28x28x1 and with filter bank of 2 filters of size 28x28 with stride 1, double.
 * \author tkornuta
 */
TEST_F(Conv28x28x1Filter2x28x28s1Double, DISABLED_Convergence) {
	double eps = 1e-6;
	double loss_value;
	size_t iteration = 0;
	// Change optimization function from default GradientDescent.
	//layer.setOptimization<mic::neural_nets::optimization::RMSProp<double> >();

	// Train for a number of iterations.
	while (iteration < 1000) {
		// Perform single learning step.
		mic::types::MatrixPtr<double> predicted_y = layer.forward(x);
		mic::types::MatrixPtr<double> dy = loss.calculateGradient(target_y, predicted_y);
		loss_value = loss.calculateMeanLoss(target_y, predicted_y);

		//std::cout<<"[" << iteration << "]\t Loss = " << loss_value <<std::endl;
		//std::cout << (*target_y).transpose() << " vs " << (*predicted_y).transpose() << std::endl;
		if (loss_value < eps)
			break;

		// Apply the changes - according to the optimization function.
		layer.backward(dy);
		layer.update(0.01, 0.0);
		// Next iteration.
		iteration++;
	}//: while
	// Check loss.
	EXPECT_LE(loss_value, eps);
}


} } } //: namespaces
