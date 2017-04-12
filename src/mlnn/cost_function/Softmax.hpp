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
		m.add("e", inputs_, 1);
		m.add("sum", 1, 1);
	}

	virtual ~Softmax() {};

	void forward(bool test_ = false) {
		// Calculate the e matrix.
		mic::types::MatrixPtr<eT> e = m["e"];

		//(*e) = ((*s['x']).unaryExpr(std::ptr_fun<eT>(std::exp)));
		for (size_t i = 0; i < (size_t)e->size(); i++)
			(*e)[i] = std::exp((*s['x'])[i]);

		// Colwise sum - sum the values in columns, one by one.
		mic::types::MatrixPtr<eT> sum = m["sum"];
		(*sum) = e->colwise().sum();

		// Get output.
		mic::types::MatrixPtr<eT> y = s["y"];

		// Iterate through elements.
		for (size_t i = 0; i < (size_t)y->rows(); i++) {
			for (size_t j = 0; j < (size_t)y->cols(); j++) {
				(*y)(i, j) = (*e)(i, j) / (*sum)(j);
			}//: for
		}//: for

	}

	void backward() {
		// dx = dy - y;
		(*g['x']) = (*g['y']) - (*s['y']);
	}

	/*!
	 * Changes the size of the batch - resizes e and sum.
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_) {
		// Call parent resize.
		Layer<eT>::resizeBatch(batch_size_);

		// Reshape the temporary matrices.
		m["e"]->resize(m["e"]->rows(), batch_size_);
		m["sum"]->resize(m["sum"]->rows(), batch_size_);
	}

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::m;


private:
	/*!
	 * Private constructor, used only during the serialization.
	 */
	Softmax<eT>() : Layer<eT> () { }


};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SOFTMAX_HPP_ */
