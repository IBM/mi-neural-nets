/*!
 * \file Regression.hpp
 * \brief 
 * \author tkornut
 * \date Apr 22, 2016
 */

#ifndef SRC_MLNN_REGRESSION_HPP_
#define SRC_MLNN_REGRESSION_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace cost_function {


/*!
 * \brief Regression layer
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Regression : public mic::mlnn::Layer<eT> {
public:

	Regression<eT>(size_t inputs_, std::string name_ = "Regression") :
		Layer<eT>(inputs_, inputs_, 1, LayerTypes::Regression, name_) {

	}

	virtual ~Regression() {};

	void forward(bool test_ = false) {
		// Pass inputs to outputs.
		(*s['y']) = (*s['x']);
	}

	void backward() {
		// dx = 2*(dy - y);
		(*g['x']) = 2 *((*g['y']) - (*s['y']));
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
	Regression<eT>() : Layer<eT> () { }

};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_REGRESSION_HPP_ */
