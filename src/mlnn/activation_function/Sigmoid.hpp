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
	/*!
	 * Creates a Sigmoid layer - reduced number of parameters.
	 * @param size_ Length of the input/output data.
	 * @param name_ Name of the layer.
	 */
	Sigmoid(size_t size_,std::string name_ = "Linear") :
		Sigmoid(size_, 1, 1, size_, 1, 1, name_)
	{
		std::cout<<"constructor Sigmoid 1!\n";
	}


	/*!
	 * Creates a Sigmoid layer.
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param input_depth_ Depth of the input sample.
	 * @param output_height_ Width of the output sample.
	 * @param output_width_ Height of the output sample.
	 * @param output_depth_ Depth of the output sample.
	 * @param name_ Name of the layer.
	 */
	Sigmoid(size_t input_height_, size_t input_width_, size_t input_depth_, size_t output_height_, size_t output_width_, size_t output_depth, std::string name_ = "Sigmoid") :
		Layer<eT>::Layer(input_height_, input_width_, input_depth_,
				output_height_, output_width_, output_depth,
				LayerTypes::Sigmoid, name_)
	{
		std::cout<<"constructor Sigmoid!\n";
	}

	/*!
	 * Virtual destructor - empty.
	 */
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
