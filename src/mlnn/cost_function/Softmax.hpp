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

	}

	virtual ~Softmax() {};

	void forward(bool test_ = false) {
		// Calculate the e matrix.
		mic::types::Matrix<eT> e = (Eigen::Matrix<eT>)((*s['x']).unaryExpr(std::ptr_fun<eT>(::expf)));

		mic::types::Vector<eT> sum = e.colwise().sum();

		for (size_t i = 0; i < e.rows(); i++) {
			for (size_t j = 0; j < e.cols(); j++) {

				(*s['y'])(i, j) = e(i, j) / sum(j);
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
