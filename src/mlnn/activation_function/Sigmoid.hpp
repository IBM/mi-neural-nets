/*!
 * \file Sigmoid.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_SIGMOID_HPP_
#define SRC_MLNN_SIGMOID_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

/*!
 * \author krocki
 */
class Sigmoid : public mic::mlnn::Layer {
public:
	Sigmoid(size_t inputs, size_t outputs, size_t batch_size, std::string name_ = "Sigmoid");

	virtual ~Sigmoid() {};

	void forward(bool test = false);

	void backward();

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Sigmoid() : Layer () { }


};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SIGMOID_HPP_ */
