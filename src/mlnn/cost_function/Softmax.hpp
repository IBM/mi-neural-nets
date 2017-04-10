/*!
 * \file Softmax.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_SOFTMAX_HPP_
#define SRC_MLNN_SOFTMAX_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace cost_function {


/*!
 * Softmax cost function.
 * \author krocki
 */
class Softmax : public mic::mlnn::Layer {
public:

	Softmax(size_t inputs_, size_t outputs_, size_t batch_size_, std::string name_ = "Softmax");

	~Softmax() {};

	void forward(bool test_ = false);

	void backward();

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Softmax() : Layer () { }


};

} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_SOFTMAX_HPP_ */
