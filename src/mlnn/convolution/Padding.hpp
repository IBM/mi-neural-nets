/*!
 * \file Padding.hpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#ifndef SRC_MLNN_PADDING_HPP_
#define SRC_MLNN_PADDING_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

/*!
 * \brief Class implementing padding operation - expanding the size of image (matrix) by a margin of n pixels on every image side.
 * \author krocki
 */
class Padding : public Layer {
public:

	Padding(size_t inputs, size_t _channels, size_t batch_size, size_t _padding, std::string name_ = "Padding");

	~Padding() {};


	void forward(bool test = false);

	void backward();

protected:

	size_t channels;

	size_t padding;

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Padding() : Layer () { }



};


} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_PADDING_HPP_ */
