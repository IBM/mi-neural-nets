/*!
 * \file ELU.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_ELU_HPP_
#define SRC_MLNN_ELU_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

/*!
 * \brief Class implementing the layer with Exponential Linear Unit (ELU).
 * http://arxiv.org/pdf/1511.07289v5.pdf
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class ELU : public mic::mlnn::Layer<eT> {
public:

	ELU<eT>(size_t inputs_, std::string name_ = "ELU") :
		Layer<eT>(inputs_, inputs_, 1, LayerTypes::ELU, name_) {

	}

	virtual ~ELU() {};

	void forward(bool test = false) {
		// Access the data of both matrices.
		eT* x = s['x']->data();
		eT* y = s['y']->data();

		// Iterate through elements.
		size_t size = (size_t) s['x']->rows() * s['x']->cols();
		for (size_t i = 0; i < size;  i++) {
			y[i] = x[i] > 0.0f ? x[i] : (expf(x[i]) - 1.0f);
		}//: for
	}

	void backward() {
		// Access the data of matrices.
		eT* gx = g['x']->data();
		eT* gy = g['y']->data();
		eT* y = s['y']->data();

		// Iterate through elements.
		size_t size = (size_t) g['x']->rows() * g['x']->cols();
		for (size_t i = 0; i < size;  i++) {
			// Calculate the ELU y derivative.
			eT dy = y[i] > 0.0f ? 1.0f : exp(y[i]);
			// Pass the gradient.
			gx[i] = dy * gy[i];

		}//: for
	}

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
	ELU<eT>() : Layer<eT> () { }

};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_ELU_HPP_ */
