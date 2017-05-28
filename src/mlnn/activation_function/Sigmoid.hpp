/*!
 * \file Sigmoid.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_SIGMOID_HPP_
#define SRC_MLNN_SIGMOID_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

/*!
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Sigmoid : public mic::mlnn::Layer<eT> {
public:
	Sigmoid<eT>(size_t inputs, std::string name_ = "Sigmoid") :
		Layer<eT>(inputs, inputs, 1, LayerTypes::Sigmoid, name_) {

	};

	virtual ~Sigmoid() {};

	void forward(bool test = false) {
		// Access the data of both matrices.
		eT* x = s['x']->data();
		eT* y = s['y']->data();

		for (size_t i = 0; i < (size_t)s['x']->rows() * s['x']->cols(); i++) {
			y[i] = 1.0f / (1.0f +::exp(-x[i])); //: float -> expf
		}//: for
	}

	void backward() {
		// Access the data of matrices.
		eT* gx = g['x']->data();
		eT* gy = g['y']->data();
		eT* y = s['y']->data();

		for (size_t i = 0; i < (size_t)g['x']->rows() * g['x']->cols(); i++) {
			// "Pass" the gradient multiplied by the sigmoid derivative.
			gx[i] = gy[i]* (y[i] * (1.0 - y[i]));
		}//: for
	}

	/*!
	 * Performs the update according to the calculated gradients and injected optimization method. Empty as this is a "const" layer.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	virtual void update(eT alpha_, eT decay_  = 0.0f) { };

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Sigmoid<eT>() : Layer<eT> () { }


};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SIGMOID_HPP_ */
