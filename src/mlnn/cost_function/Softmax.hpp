/*!
 * \file Softmax.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_SOFTMAX_HPP_
#define SRC_MLNN_SOFTMAX_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace cost_function {


/*!
 * Softmax cost function.
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Softmax : public mic::mlnn::Layer<eT> {
public:

	Softmax<eT>(size_t inputs_, std::string name_ = "Softmax") :
		Layer<eT>(inputs_, inputs_, 1, LayerTypes::Softmax, name_) {

		// Add "temporary" parameters.
		m.add("e", inputs_, inputs_);
		m.add("sum", inputs_, inputs_);
	}

	virtual ~Softmax() {};

	void forward(bool test_ = false) {
		// Calculate the e matrix.
		mic::types::MatrixPtr<eT> e = m["e"];
		(*e) = ((*s['x']).unaryExpr(std::ptr_fun(::exp)));

		mic::types::MatrixPtr<eT> sum = m["sum"];
		(*sum) = e->colwise().sum();

		// Get output.
		mic::types::MatrixPtr<eT> y = s["y"];

		// Iterate through elements.
		for (size_t i = 0; i < (size_t)e->rows(); i++) {
			for (size_t j = 0; j < (size_t)e->cols(); j++) {
				(*y)(i, j) = (*e)(i, j) / (*sum)(j);
			}
		}

	}

	void backward() {
		// dx = dy - y;
		(*g['x']) = (*g['y']) - (*s['y']);
	}

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::m;


private:
	// Adds the nn class the access to protected fields of class layer.
	//friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Softmax<eT>() : Layer<eT> () { }


};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SOFTMAX_HPP_ */
