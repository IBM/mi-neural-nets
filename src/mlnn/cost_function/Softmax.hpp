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
 * \author tkornuta
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
		m.add("max", 1, 1);
	}

	virtual ~Softmax() {};

	void forward(bool test_ = false) {
		mic::types::MatrixPtr<eT> x = s["x"];
		mic::types::MatrixPtr<eT> y = s["y"];
		mic::types::MatrixPtr<eT> e = m["e"];
		mic::types::MatrixPtr<eT> max = m["max"];
		mic::types::MatrixPtr<eT> sum = m["sum"];

//		std::cout << "Softmax forward: s['x'] = \n" << (*s['x']) << std::endl;

		// Prevent overflow according to: http://eric-yuan.me/softmax/
		(*max) = x->colwise().maxCoeff();

//		std::cout << "Softmax forward: max = \n" << (*max) << std::endl;

		// Calculate the e matrix - with overflow prevention.
		for (size_t i = 0; i < (size_t)y->rows(); i++)
			for (size_t j = 0; j < (size_t)y->cols(); j++)
				(*e)(i, j) = std::exp( (*x)(i, j) - (*max)(j) );

//		std::cout << "Softmax forward: e = \n" << (*e) << std::endl;

		// Sum the values in columns (single batch), one by one.
		(*sum) = e->colwise().sum();

//		std::cout << "Softmax forward: sum = \n" << (*sum) << std::endl;

		// Iterate through elements.
		for (size_t i = 0; i < (size_t)y->rows(); i++) {
			for (size_t j = 0; j < (size_t)y->cols(); j++) {
				(*y)(i, j) = (*e)(i, j) / (*sum)(j);
			}//: for
		}//: for

//		std::cout << "Softmax forward: s['y'] = \n" << (*s['y']) << std::endl;
	}

	void backward() {
		mic::types::MatrixPtr<eT> y = s["y"];
		mic::types::MatrixPtr<eT> dx = g["x"];
		mic::types::MatrixPtr<eT> dy = g["y"];

		// Pass the gradient.
		for (size_t i = 0; i < (size_t)y->size(); i++)
			// dx = dy *  derivatie of softmax, i.e. y * (1 - y);
			(*dx)[i] = (*dy)[i] * (*y)[i] * (1 - (*y)[i]);

/*		std::cout << "Softmax backward: g['y'] = \n" << (*g['y']) << std::endl;
		std::cout << "Softmax backward: g['x'] = \n" << (*g['x']) << std::endl;*/
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
		m["max"]->resize(m["max"]->rows(), batch_size_);
	}

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::m;


private:
	// Friend class - required for using boost serialization.
	template<typename tmp1, typename tmp2> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Softmax<eT>() : Layer<eT> () { }


};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SOFTMAX_HPP_ */
