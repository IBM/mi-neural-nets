/*!
 * \file Dropout.hpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#ifndef SRC_MLNN_DROPOUT_HPP_
#define SRC_MLNN_DROPOUT_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace regularisation {


/*!
 * \brief Droput layer - a layer used for the regularization of neural network by randomly dropping neurons during training.
 * \author krocki
 */
class Dropout : public Layer {
public:

	Dropout(size_t inputs, size_t outputs, size_t batch_size, float _ratio, std::string name = "Dropout");

	virtual ~Dropout() {};

	void forward(bool test = false);

	void backward();

protected:

	float keep_ratio;

	/*!
	 * Dropout mask.
	 * Computed in forward() method, so it does not need to be moved to common parameters/memory arrays.
	 */
	mic::types::MatrixXf dropout_mask;

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Dropout() : Layer () { }


};


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_DROPOUT_HPP_ */
