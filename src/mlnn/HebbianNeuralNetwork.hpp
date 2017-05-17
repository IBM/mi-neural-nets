/*!
 * @file: HebbianNeuralNetwork.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef HEBBIANNEURALNETWORK_HPP_
#define HEBBIANNEURALNETWORK_HPP_

#include <mlnn/MultiLayerNeuralNetwork.hpp>


namespace mic {
namespace mlnn {

/*!
 * \brief Class representing a multi-layer neural network based on hebbian learning.
 *
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT>
class HebbianNeuralNetwork : public MultiLayerNeuralNetwork<eT> {
public:


	/*!
	 * Constructor. Sets the neural network name.
	 * @param name_ Name of the network.
	 */
	HebbianNeuralNetwork(std::string name_ = "hebbian_net") : MultiLayerNeuralNetwork<eT> (name_)
	{
		// Set classical hebbian rule as default learning rule.
		MultiLayerNeuralNetwork<eT>::template setOptimization<mic::neural_nets::learning::HebbianRule<eT> > ();
	}


	/// Virtual descriptor - empty.
	virtual ~HebbianNeuralNetwork() { }


	/*!
	 * Passes the data in a feed-forward manner through all consecutive layers, from the input to the output layer.
	 * @param input_data Input data - a matrix containing [sample_size x batch_size].
	 * @param skip_dropout Flag for skipping dropouts - which should be set to true during testing.
	 */
	void forward(mic::types::MatrixPtr<eT> input_data, bool skip_dropout = false)  {
		// Make sure that there are some layers in the nn!
		assert(layers.size() != 0);

		// Boost::Matrix is col major!
		LOG(LDEBUG) << "Inputs size: " << input_data->rows() << "x" << input_data->cols();
		LOG(LDEBUG) << "First layer input matrix size: " <<  layers[0]->s['x']->rows() << "x" << layers[0]->s['x']->cols();

		// Make sure that the dimensions are ok.
		// Check only rows, as cols determine the batch size - and we allow them to be dynamically changing!.
		assert((layers[0]->s['x'])->rows() == input_data->rows());
		//LOG(LDEBUG) <<" input_data: " << input_data.transpose();

		// Connect layers by setting the input matrices pointers to point the output matrices.
		// There will not need to be copy data between layers anymore.
		if (!connected) {
			// Set pointers - pass result to the next layer: x(next layer) = y(current layer).
			if (layers.size() > 1)
				for (size_t i = 0; i < layers.size()-1; i++) {
					layers[i+1]->s['x'] = layers[i]->s['y'];
				}//: for
			connected = true;
		}

		//assert((layers[0]->s['x'])->cols() == input_data->cols());
		// Change the size of batch - if required.
		resizeBatch(input_data->cols());

		// Copy inputs to the lowest point in the network.
		(*(layers[0]->s['x'])) = (*input_data);

		// Compute the forward activations.
		for (size_t i = 0; i < layers.size(); i++) {
			LOG(LDEBUG) << "Layer [" << i << "] " << layers[i]->name() << ": (" <<
					layers[i]->inputSize() << "x" << layers[i]->batchSize() << ") -> (" <<
					layers[i]->outputSize() << "x" << layers[i]->batchSize() << ")";

			// Perform the forward computation: y = f(x).
			layers[i]->forward(skip_dropout);

		}
		//LOG(LDEBUG) <<" predictions: " << getPredictions()->transpose();
	}

	/*!
	 * Trains the neural network with a given batch.
	 * @param encoded_batch_ Batch encoded in the form of matrix of size [sample_size x batch_size].
	 * @param learning_rate_ The learning rate.
	 * @return Loss computed according to the selected loss function. If function not set - returns INF.
	 */
	eT train(mic::types::MatrixPtr<eT> encoded_batch_, eT learning_rate_) {

		// Forward propagate the activations from first layer to the last.
		forward(encoded_batch_);

		// Apply the changes - according to the optimization function.
		update(learning_rate_);

		// Calculate mean value of the loss function (i.e. loss divided by the batch size).
		//eT loss_value = loss->calculateMeanLoss(encoded_targets_, encoded_predictions);

		// Return loss.
		return 0;//loss_value;
	}


	/*!
	 * Tests the neural network with a given batch.
	 * @param encoded_batch_ Batch encoded in the form of matrix of size [sample_size x batch_size].
	 * @param encoded_targets_ Targets (labels) encoded in the form of matrix of size [label_size x batch_size].
	 * @return Loss computed according to the selected loss function. If function not set - returns INF.
	 */
	eT test(mic::types::MatrixPtr<eT> encoded_batch_) {
		// skip dropout layers at test time
		bool skip_dropout = true;

		forward(encoded_batch_, skip_dropout);

		// Calculate the mean loss.
		return 0;//loss->calculateMeanLoss(encoded_targets_, encoded_predictions);
	}

	// Unhide the overloaded public methods & fields inherited from the template class MultiLayerNeuralNetwork fields via "using" statement.
	using MultiLayerNeuralNetwork<eT>::update;
	using MultiLayerNeuralNetwork<eT>::setOptimization;
	using MultiLayerNeuralNetwork<eT>::resizeBatch;

protected:
	// Unhide the overloaded protected methods & fields inherited from the template class MultiLayerNeuralNetwork fields via "using" statement.
	using MultiLayerNeuralNetwork<eT>::layers;
	using MultiLayerNeuralNetwork<eT>::connected;

};

} /* namespace mlnn */
} /* namespace mic */

#endif /* HEBBIANNEURALNETWORK_HPP_ */
