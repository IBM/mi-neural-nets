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

	/*!
	 * Creates a ELU layer - reduced number of parameters.
	 * @param size_ Length of the input/output data.
	 * @param name_ Name of the layer.
	 */
	ELU(size_t size_,std::string name_ = "ELU") :
		ELU(size_, 1, 1, name_)
	{

	}


	/*!
	 * Creates a ELU layer.
	 * @param height_ Height of the input/output sample.
	 * @param width_ Width of the input/output sample.
	 * @param depth_ Depth of the input/output sample.
	 * @param name_ Name of the layer.
	 */
	ELU(size_t height_, size_t width_, size_t depth_, std::string name_ = "ELU") :
		Layer<eT>::Layer(height_, width_, depth_,
				height_, width_, depth_,
				LayerTypes::ELU, name_)
	{

	}

	/*!
	 * Virtual destructor - empty.
	 */
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
	template<typename tmp> friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	ELU<eT>() : Layer<eT> () { }

};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_ELU_HPP_ */
