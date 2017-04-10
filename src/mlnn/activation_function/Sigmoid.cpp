/*!
 * \file Sigmoid.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/activation_function/Sigmoid.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

Sigmoid::Sigmoid(size_t inputs, size_t outputs, size_t batch_size, std::string name_) :
	Layer(inputs, outputs, batch_size, LayerTypes::Sigmoid, name_) {

};

void Sigmoid::forward(bool apply_dropout) {

	// y = logistic(x);

	// Access the data of both matrices.
	float* x = s['x']->data();
	float* y = s['y']->data();

	for (int i = 0; i < s['x']->rows() * s['x']->cols(); i++) {
		y[i] = 1.0f / (1.0f +::expf(-x[i]));
	}//: for

}

void Sigmoid::backward() {

	//dx.array() = dy.array() * y.array() * (1.0 - y.array()).array();

	// Access the data of matrices.
	float* gx = g['x']->data();
	float* gy = g['y']->data();
	float* y = s['y']->data();

	for (int i = 0; i < g['x']->rows() * g['x']->cols(); i++) {

		// Calculate the gradient.
		gx[i] = gy[i] * y[i] * (1.0 - y[i]);

	}//: for

}

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */
