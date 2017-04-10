/*!
 * \file ELU.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/activation_function/ELU.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

ELU::ELU(size_t inputs, size_t outputs, size_t batch_size, std::string name_) :
	Layer(inputs, outputs, batch_size, LayerTypes::ELU, name_) {

}

void ELU::forward(bool apply_dropout) {

	// y = activation_ELU(x);

	// Access the data of both matrices.
	float* x = s['x']->data();
	float* y = s['y']->data();

	for (int i = 0; i < s['x']->rows() * s['x']->cols(); i++) {
		y[i] = x[i] > 0.0f ? x[i] : (expf(x[i]) - 1.0f);
	}//: for
}

void ELU::backward() {

	// dx.array() = derivative_ELU(y).array() * dy.array();

	// Access the data of matrices.
	float* gx = g['x']->data();
	float* gy = g['y']->data();
	float* y = s['y']->data();

	for (int i = 0; i < g['x']->rows() * g['x']->cols(); i++) {

		// Calculate the ELU y derivative.
		float dy = y[i] > 0.0f ? 1.0f : expf(y[i]);
		// Calculate the gradient.
		gx[i] = dy * gy[i];

	}//: for

}

} /* namespace activation_function */
} /* namespace mlnn */
} /* namespace mic */
