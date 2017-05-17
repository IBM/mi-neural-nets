/*!
 * @file: BackpropagationNeuralNetwork.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef BACKPROPAGATIONNEURALNETWORK_H_
#define BACKPROPAGATIONNEURALNETWORK_H_

#include <mlnn/MultiLayerNeuralNetwork.hpp>

namespace mic {
namespace mlnn {

/*!
 * \brief Class representing a multi-layer neural network based on backpropagation/gradient descent.
 *
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT>
class BackpropagationNeuralNetwork : public MultiLayerNeuralNetwork<eT> {
public:

	/*!
	 * Constructor. Sets the neural network name.
	 * @param name_ Name of the network.
	 */
	BackpropagationNeuralNetwork(std::string name_ = "mlnn") : MultiLayerNeuralNetwork<eT> (name_)
	{
		// Set default cross entropy loss function.
		setLoss <mic::neural_nets::loss::CrossEntropyLoss<eT> >();

		// Set "classical" SDG as default optimization method.
		MultiLayerNeuralNetwork<eT>::template setOptimization<mic::neural_nets::optimization::GradientDescent<eT> > ();
	}

	/*!
	 * Sets the loss function.
	 * \tparam LossFunction Template parameter denoting the loss function type (e.g. mic::neural_nets::loss::CrossEntropyLoss<eT>).
	 */
	template<typename LossFunction>
	void setLoss () {
		loss = std::make_shared< LossFunction > (LossFunction());
	}


	/// Virtual descriptor - empty.
	virtual ~BackpropagationNeuralNetwork() { }


	/*!
	 * Performs the back propagation algorithm i.e. propagates the gradients from the last layer to the first.
	 * @param gradients_ The input gradient (i.e. result of the derivative of the loss function).
	 */
	void backward(mic::types::MatrixPtr<eT> gradients_) {
		// Make sure that there are some layers in the nn!
		assert(layers.size() != 0);

		LOG(LDEBUG) << "Last layer output gradient matrix size: " << layers.back()->g['y']->cols() << "x" << layers.back()->g['y']->rows();
		LOG(LDEBUG) << "Passed target matrix size: " <<  gradients_->cols() << "x" << gradients_->rows();

		// Make sure that the dimensions are ok.
		assert((layers.back()->g['y'])->cols() == gradients_->cols());
		assert((layers.back()->g['y'])->rows() == gradients_->rows());

		// Set gradient of the last layer - COPY data.
		(*(layers.back()->g['y'])) = (*gradients_);

		// Back-propagate the gradients.
		for (int i = layers.size() - 1; i >= 0; i--) {
			layers[i]->resetGrads();
			layers[i]->backward();
		}//: for

	}


	/*!
	 * Trains the neural network with a given batch.
	 * @param encoded_batch_ Batch encoded in the form of matrix of size [sample_size x batch_size].
	 * @param encoded_targets_ Targets (labels) encoded in the form of matrix of size [label_size x batch_size].
	 * @param learning_rate_ The learning rate.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 * @return Loss computed according to the selected loss function. If function not set - returns INF.
	 */
	eT train(mic::types::MatrixPtr<eT> encoded_batch_, mic::types::MatrixPtr<eT> encoded_targets_, eT learning_rate_, eT decay_ = 0.0f) {

		// Forward propagate the activations from first layer to the last.
		forward(encoded_batch_);

		// Get predictions.
		mic::types::MatrixPtr<eT> encoded_predictions = getPredictions();

		// Calculate gradient according to the loss function.
		mic::types::MatrixPtr<eT> dy = loss->calculateGradient(encoded_targets_, encoded_predictions);

		// Backpropagate the gradients from last layer to the first.
		backward(dy);

		// Apply the changes - according to the optimization function.
		update(learning_rate_, decay_);

		// Calculate mean value of the loss function (i.e. loss divided by the batch size).
		eT loss_value = loss->calculateMeanLoss(encoded_targets_, encoded_predictions);

		// Return loss.
		return loss_value;
	}


	/*!
	 * Tests the neural network with a given batch.
	 * @param encoded_batch_ Batch encoded in the form of matrix of size [sample_size x batch_size].
	 * @param encoded_targets_ Targets (labels) encoded in the form of matrix of size [label_size x batch_size].
	 * @return Loss computed according to the selected loss function. If function not set - returns INF.
	 */
	eT test(mic::types::MatrixPtr<eT> encoded_batch_, mic::types::MatrixPtr<eT> encoded_targets_) {
		// skip dropout layers at test time
		bool skip_dropout = true;

		forward(encoded_batch_, skip_dropout);

		// Get predictions.
		mic::types::MatrixPtr<eT> encoded_predictions = getPredictions();

		// Calculate the mean loss.
		return loss->calculateMeanLoss(encoded_targets_, encoded_predictions);
	}


	/*!
	 * Calculates the loss function according to the selected loss function.
	 * @param encoded_targets_ Targets (labels) encoded in the form of pointer to matrix of size [label_size x batch_size].
	 * @param encoded_predictions_ Predicted outputs of the network encoded in the form of pointer to matrix of size [label_size x batch_size].
	 * @return Loss computed according to the selected loss function.
	 */
	eT calculateMeanLoss(mic::types::MatrixPtr<eT> encoded_targets_, mic::types::MatrixPtr<eT> encoded_predictions_)  {

		return loss->calculateMeanLoss(encoded_targets_, encoded_predictions_);
	}

	// Unhide the overloaded public methods & fields inherited from the template class MultiLayerNeuralNetwork fields via "using" statement.
	using MultiLayerNeuralNetwork<eT>::getPredictions;
	using MultiLayerNeuralNetwork<eT>::forward;
	using MultiLayerNeuralNetwork<eT>::update;
	using MultiLayerNeuralNetwork<eT>::setOptimization;

protected:
	// Unhide the overloaded protected methods & fields inherited from the template class MultiLayerNeuralNetwork fields via "using" statement.
	using MultiLayerNeuralNetwork<eT>::layers;

	/*!
	 * Pointer to loss function.
	 */
	std::shared_ptr<mic::neural_nets::loss::Loss<eT> > loss;

};

} /* namespace mlnn */
} /* namespace mic */

#endif /* BACKPROPAGATIONNEURALNETWORK_H_ */
