/*!
 * \file Identity.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_IDENTITY_HPP_
#define SRC_MLNN_IDENTITY_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {


/*!
 * \brief Identity layer - passes inputs/gradients without any modifications.
 * Implemented for testing purposes.
 * \author krocki
 */
class Identity : public mic::mlnn::Layer {
public:

	Identity(size_t inputs, size_t outputs, size_t batch_size, std::string name_ = "Identity");

	virtual ~Identity() {};

	void forward(bool test = false);

	void backward();

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Identity() : Layer () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_IDENTITY_HPP_ */
