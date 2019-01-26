/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
 * \brief Softmax activation function.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Softmax : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates a Softmax layer - reduced number of parameters.
	 * @param size_ Length of the input/output data.
	 * @param name_ Name of the layer.
	 */
	Softmax(size_t size_, std::string name_ = "Softmax") :
		Softmax(size_, 1, 1, name_)
	{

	}


	/*!
	 * Creates a Softmax layer.
	 * @param height_ Height of the input/output sample.
	 * @param width_ Width of the input/output sample.
	 * @param depth_ Depth of the input/output sample.
	 * @param name_ Name of the layer.
	 */
	Softmax(size_t height_, size_t width_, size_t depth_,
			std::string name_ = "Softmax") :
		Layer<eT>::Layer(height_, width_, depth_,
				height_, width_, depth_,
				LayerTypes::Softmax, name_)
	{
		// Add "temporary" parameters.
		m.add("e", Layer<eT>::inputSize(), 1);
		m.add("sum", 1, 1);
		m.add("max", 1, 1);
	}


	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~Softmax() {};

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



	void forward(bool test_ = false) {
		mic::types::MatrixPtr<eT> x = s["x"];
		mic::types::MatrixPtr<eT> y = s["y"];
		mic::types::MatrixPtr<eT> e = m["e"];
		mic::types::MatrixPtr<eT> max = m["max"];
		mic::types::MatrixPtr<eT> sum = m["sum"];

		//std::cout << "Softmax forward: s['x'] = \n" << (*s['x']) << std::endl;

		// Prevent overflow according to: http://eric-yuan.me/softmax/
		(*max) = x->colwise().maxCoeff();

		// Calculate the e matrix - with overflow prevention.
		for (size_t i = 0; i < (size_t)y->rows(); i++)
			for (size_t j = 0; j < (size_t)y->cols(); j++)
				(*e)(i, j) = std::exp( (*x)(i, j) - (*max)(j) );

		// Sum the values in columns (single batch), one by one.
		(*sum) = e->colwise().sum();

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
			// dx = dy *  derivative of softmax, i.e. y * (1 - y);
			(*dx)[i] = (*dy)[i] * (*y)[i] * (1 - (*y)[i]);

		/*std::cout << "Softmax backward: g['y'] = \n" << (*g['y']) << std::endl;
		std::cout << "Softmax backward: g['x'] = \n" << (*g['x']) << std::endl;*/
	}

	/*!
	 * Performs the update according to the calculated gradients and injected optimization method. Empty as this is a "const" layer.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	virtual void update(eT alpha_, eT decay_  = 0.0f) { };

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
	template<typename tmp> friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Softmax<eT>() : Layer<eT> () { }


};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SOFTMAX_HPP_ */
