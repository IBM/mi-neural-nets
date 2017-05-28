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

/*		std::cout << "ReLU forward: s['x'] = \n" << (*s['x']) << std::endl;
		std::cout << "ReLU forward: s['y'] = \n" << (*s['y']) << std::endl;*/
	}

	void backward() {
		// Access the data of matrices.
		eT* gx = g['x']->data();
		eT* gy = g['y']->data();
		eT* y = s['y']->data();

		// Iterate through elements.
		size_t size = g['x']->rows() * g['x']->cols();
		for (size_t i = 0; i < size; i++) {
			// Calculate the ReLU "derivative".
			eT dy = (eT)(y[i] > 0.0);
			// Pass the gradient.
			gx[i] = dy * gy[i];

		}//: for

/*		std::cout << "ReLU backward: g['y'] = \n" << (*g['y']) << std::endl;
		std::cout << "ReLU backward: g['x'] = \n" << (*g['x']) << std::endl;*/
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
	ReLU<eT>() : Layer<eT> () { }

};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_RELU_HPP_ */
