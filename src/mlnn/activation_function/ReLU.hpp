/*!
 * \file ReLU.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_RELU_HPP_
#define SRC_MLNN_RELU_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

/*!
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class ReLU : public mic::mlnn::Layer<eT> {
public:

	ReLU<eT>(size_t inputs_, std::string name_ = "ReLU") :
		Layer<eT>(inputs_, inputs_, 1, LayerTypes::ReLU, name_) {

	}

	virtual ~ReLU() {};

	void forward(bool apply_dropout = false) {
		// Access the data of both matrices.
		eT* x = s['x']->data();
		eT* y = s['y']->data();

		// Iterate through elements.
		size_t size = s['x']->rows() * s['x']->cols();
		for (size_t i = 0; i < size;  i++) {
			y[i] = fmax(x[i], 0.0f); //: floats - fmax
		}//: for
	}

	void backward() {
		// Access the data of matrices.
		eT* gx = g['x']->data();
		eT* gy = g['y']->data();
		eT* y = s['y']->data();

		// Iterate through elements.
		size_t size = g['x']->rows() * g['x']->cols();
		for (size_t i = 0; i < size; i++) {
			// Calculate the ReLU y derivative.
			eT dy = (eT)(y[i] > 0.0);
			// Pass the gradient.
			gx[i] = dy * gy[i];

		}//: for
	}

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;

private:
	// Adds the nn class the access to protected fields of class layer.
	//friend class mic::mlnn::MultiLayerNeuralNetwork<eT>;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	ReLU<eT>() : Layer<eT> () { }

};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_RELU_HPP_ */
