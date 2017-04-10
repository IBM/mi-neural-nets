/*!
 * \file ELU.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_ELU_HPP_
#define SRC_MLNN_ELU_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace activation_function {

/*!
 * \brief Class implementing the layer with Exponential Linear Unit (ELU).
 * http://arxiv.org/pdf/1511.07289v5.pdf
 * \author krocki
 */
class ELU : public mic::mlnn::Layer {
public:

	ELU(size_t inputs, size_t outputs, size_t batch_size, std::string name_ = "ELU");

	virtual ~ELU() {};

	void forward(bool test = false);

	void backward();

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	ELU() : Layer () { }

};

} /* activation_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_ELU_HPP_ */
