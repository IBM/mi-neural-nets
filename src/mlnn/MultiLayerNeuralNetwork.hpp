/*!
 * \file MultiLayerNeuralNetwork.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_MULTILAYERNEURALNETWORK_HPP_
#define SRC_MLNN_MULTILAYERNEURALNETWORK_HPP_

#include <types/MatrixTypes.hpp>
#include <mlnn/layer/LayerTypes.hpp>
#include <loss/LossTypes.hpp>

#include <fstream>
// Include headers that implement a archive in simple text format
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <logger/Log.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {
namespace serialization {
class access;
}//: serialization
}//: access


namespace mic {
namespace mlnn {

using namespace activation_function;
//using namespace convolution;
using namespace cost_function;
using namespace fully_connected;
using namespace regularisation;

/*!
 * \brief Class representing a multi-layer neural network.
 * \author tkornuta/kmrocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float, typename LossFunction=mic::neural_nets::loss::CrossEntropyLoss<float> >
class MultiLayerNeuralNetwork {
public:

	/*!
	 * Constructor. Sets the neural network name.
	 * @param name_ Name of the network.
	 */
	MultiLayerNeuralNetwork(std::string name_ = "mlnn") :
		name(name_),
		connected(false) // Initially the network is not connected.
	{

	}

	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~MultiLayerNeuralNetwork() { }

	/*!
	 * Adds layer to neural network.
	 * @param layer_ptr_ Pointer to the newly created layer.
	 * @tparam layer_ptr_ Layer type.
	 */
	template <typename LayerType>
	void pushLayer( LayerType* layer_ptr_){
		layers.push_back(std::shared_ptr <LayerType> (layer_ptr_));
		connected = false;
	}

	/*!
	 * Returns n-th layer of the neural network.
	 * @param layer_ptr_ Pointer to the newly created layer.
	 * @tparam layer_ptr_ Layer type.
	 */
	template <typename LayerType>
	std::shared_ptr<LayerType> getLayer(size_t index_){
		assert(index_ < layers.size());
		// Cast the pointer to LayerType.
		return std::dynamic_pointer_cast< LayerType >( layers[index_] );
	}


	/*!
	 * Removes several last layers of the neural network.
	 * @param number_of_layers_ Number of layers to be removed.
	 */
	void popLayer(size_t number_of_layers_ = 1){
		assert(number_of_layers_ <= layers.size());
		//layers.erase(layers.back() - number_of_layers_, layers.back());
		for (size_t i=0; i <number_of_layers_; i++)
			layers.pop_back();
		connected = false;
	}

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
			for (size_t i = 0; i < layers.size()-1; i++) {
				layers[i+1]->s['x'] = layers[i]->s['y'];
				layers[i]->g['y'] = layers[i+1]->g['x'];
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
	 * Performs the back propagation
	 * @param targets_ The targer matrix, containing target (desired) outputs of the network [encoded_label_size x batch_size]
	 */
	void backward(mic::types::MatrixPtr<eT> targets_) {
		// Make sure that there are some layers in the nn!
		assert(layers.size() != 0);

		LOG(LDEBUG) << "Last layer output gradient matrix size: " << layers.back()->g['y']->cols() << "x" << layers.back()->g['y']->rows();
		LOG(LDEBUG) << "Passed target matrix size: " <<  targets_->cols() << "x" << targets_->rows();

		// Make sure that the dimensions are ok.
		assert((layers.back()->g['y'])->cols() == targets_->cols());
		assert((layers.back()->g['y'])->rows() == targets_->rows());

		// Set targets at the top.
		(*(layers.back()->g['y'])) = (*targets_);

		// Back-propagate the error.
		for (int i = layers.size() - 1; i >= 0; i--) {
			layers[i]->resetGrads();
			layers[i]->backward();
		}//: for

	}

	/*!
	 * Performs the network training by update all layers parameters according to gradients computed by backprob.
	 * @param alpha Learning rate
	 * @param decay Weight decay - factor for
	 */
	void update(eT alpha, eT decay) {

		// update all layers according to gradients
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i]->applyGrads(alpha, decay);
		}//: for

	}

	/*!
	 * Trains the neural network with a given batch.
	 * @param encoded_batch_ Batch encoded in the form of matrix of size [sample_size x batch_size].
	 * @param encoded_targets_ Targets (labels) encoded in the form of matrix of size [label_size x batch_size].
	 * @param learning_rate_ The learning rate.
	 * @param weight_decay_ Weight decay.
	 * @return Loss computed according to the selected loss function. If function not set - returns INF.
	 */
	eT train(mic::types::MatrixPtr<eT> encoded_batch_, mic::types::MatrixPtr<eT> encoded_targets_, eT learning_rate_, eT weight_decay_) {

		// Forward propagate the activations from first layer to the last.
		forward(encoded_batch_);

		// Get predictions.
		mic::types::MatrixPtr<eT> encoded_predictions = getPredictions();

		// Calculate gradient according to the loss function.
		mic::types::MatrixPtr<eT> dy = loss.calculateGradient(encoded_targets_, encoded_predictions);

		// Backpropagate the gradients from last layer to the first.
		backward(dy);

		// Apply the changes - according to the optimization function.
		update(learning_rate_, weight_decay_);

		// Calculate mean value of the loss function (i.e. loss divided by the batch size).
		eT loss_value = loss.calculateMeanLoss(encoded_targets_, encoded_predictions);

		//eT correct = countCorrectPredictions(encoded_targets_, encoded_predictions);
		//LOG(LDEBUG) << " Loss = " << std::setprecision(2) << std::setw(6) << loss_value << " | " << std::setprecision(1) << std::setw(4) << std::fixed << 100.0 * (eT)correct / (eT)encoded_batch_->cols() << "% batch correct";

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
		return loss.calculateMeanLoss(encoded_targets_, encoded_predictions);
	}

	/*!
	 * Resets the gradients of all layers.
	 */
	void resetGrads() {
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->resetGrads();
	}

	/*!
	 * Changes the size of the batch.
	 * @param New size of the batch.
	 */
	void resizeBatch(size_t batch_size_) {
		// If current batch size is ok.
		if ((size_t)(layers[0]->s['x'])->cols() == batch_size_)
			return;

		// Else - resize.
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i]->resizeBatch(batch_size_);
		}//: for
	}

	/*!
	 * Calculates the loss function according to the selected loss function.
	 * @param encoded_targets_ Targets (labels) encoded in the form of pointer to matrix of size [label_size x batch_size].
	 * @param encoded_predictions_ Predicted outputs of the network encoded in the form of pointer to matrix of size [label_size x batch_size].
	 * @return Loss computed according to the selected loss function.
	 */
	eT calculateMeanLossFunction(mic::types::MatrixPtr<eT> encoded_targets_, mic::types::MatrixPtr<eT> encoded_predictions_)  {

		return loss.calculateMeanLoss(encoded_targets_, encoded_predictions_);
	}

	/*!
	 * Returns the predictions (output of the forward processing).
	 * @param predictions_ Predictions in the form of a matrix of size [label_size x batch_size].
	 */
	mic::types::MatrixPtr<eT> getPredictions() {
		return layers.back()->s['y'];
	}

	/*!
	 * Calculated difference between the predicted and target classes.
	 * Assumes 1-ouf-of-k encoding of classes.
	 *
	 * @param predictions_ Predictions in the form of a matrix of answers, each encoded as SDR.
	 * @param targets_ Desired results (targets) in the form of a matrix of answers, each encoded as SDR.
	 * @return
	 */
	size_t countCorrectPredictions(mic::types::MatrixPtr<eT> targets_, mic::types::MatrixPtr<eT> predictions_)  {

		// Get vectors of indices denoting classes (type of 1-ouf-of-k dencoding).
		mic::types::Matrix<eT> predicted_classes = predictions_->colwiseReturnMaxIndices();
		mic::types::Matrix<eT> target_classes = targets_->colwiseReturnMaxIndices();

		// Get direct pointers to data.
		eT *p = predicted_classes.data();
		eT *t = target_classes.data();

		size_t correct=0;
		size_t i;
		for(i=0; i< (size_t) predicted_classes.size(); i++) {
			if (p[i] == t[i])
				correct++;
		}//: for

		return correct;
	}


	size_t lastLayerInputsSize() {
		return layers.back()->inputsSize();
	}

	size_t lastLayerOutputsSize() {
		return layers.back()->outputsSize();
	}

	size_t lastLayerBatchSize() {
		return layers.back()->batchSize();
	}

	/*!
	 * Stream operator enabling to print neural network.
	 * @param os_ Ostream object.
	 * @param obj_ Tensor object.
	 */
	friend std::ostream& operator<<(std::ostream& os_, const MultiLayerNeuralNetwork& obj_) {
		// Display dimensions.
		os_ << "[" << obj_.name << "]:\n";
		// Display layers one by one.
		for (size_t i = 0; i < obj_.layers.size(); i++)
			os_ << (*obj_.layers[i]) << std::endl;

		return os_;
	}


	/*!
	 * Saves network to file using serialization.
	 * @param filename_ Name of the file.
	 */
	bool save(std::string filename_)
	{
		try {
			// Create an output archive
			std::ofstream ofs(filename_);
			boost::archive::text_oarchive ar(ofs);

			// Change batch size to 1 - fastening the save/load procedures.
			//setBatchSize(1);

			// Write data
			ar & (*this);
			LOG(LINFO) << "Network " << name << " properly saved to file " << filename_;
			LOG(LDEBUG) << "Saved network: \n" << (*this);
		} catch(...) {
			LOG(LERROR) << "Could not write neural network " << name << " to file " << filename_ << "!";
			// Clear layers - just in case.
			layers.clear();
			return false;
		}
		return true;
	}

	/*!
	 * Loads network from the file using serialization.
	 * @param filename_ Name of the file.
	 */
	bool load(std::string filename_)
	{
		try {
			// Create and input archive
			std::ifstream ifs(filename_);
			boost::archive::text_iarchive ar(ifs);
			// Load data
			ar & (*this);
			LOG(LINFO) << "Network " << name << " properly loaded from file " << filename_;
			LOG(LDEBUG) << "Loaded network: \n" << (*this);
		} catch(...) {
			LOG(LERROR) << "Could not load neural network from file " << filename_ << "!";
			// Clear layers - just in case.
			layers.clear();
			return false;
		}
		return true;
	}



protected:
	/*!
	 * Contains a list of consecutive layers.
	 */
	std::vector<std::shared_ptr <mic::mlnn::Layer<eT> > > layers;

	/*!
	 * Name of the neural network.
	 */
	std::string name;

	/*!
	 * Loss function.
	 */
	LossFunction loss;

private:
	// Friend class - required for using boost serialization.
    friend class boost::serialization::access;

    /// Flag denoting whether the layers are interconnected, thus no copying between inputs and outputs of the neighboring layers will be required.
    bool connected;

    /*!
     * Serialization save - saves the neural net object to archive.
     * @param ar Used archive.
     * @param version Version of the neural net class (not used currently).
     */
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const {
    	// Serialize name.
        ar & name;
		//ar & loss_type;

		// Serialize number of layers.
        size_t size = layers.size();
        ar & size;

		// Serialize layers one by one.
		for (size_t i = 0; i < layers.size(); i++) {
			// Serialize type first - so we can use it in load.
			ar & layers[i]->layer_type;

			// Serialize the layer.
			ar & (*layers[i]);
		}//: for

    }

    /*!
     * Serialization load - loads the neural net object to archive.
     * @param ar Used archive.
     * @param version Version of the neural net class (not used currently).
     */
    template<class Archive>
    void load(Archive & ar, const unsigned int version) {
    	// Clear the layers vector - just in case.
    	layers.clear();
    	connected = false;

    	// Deserialize name and loss function type.
		ar & name;
		//ar & loss;

		// Deserialize number of layers.
		size_t size;
		ar & size;

		// Serialize layers one by one.
		for (size_t i = 0; i < size; i++) {
			LayerTypes lt;
			// Get layer type
			ar & lt;

			std::shared_ptr<Layer<eT> > layer_ptr;
			switch(lt) {
			// activation_function
			case(LayerTypes::ELU):
				layer_ptr = std::make_shared<ELU<eT> >(ELU<eT>());
				LOG(LDEBUG) <<  "ELU";
				break;
			case(LayerTypes::ReLU):
				layer_ptr = std::make_shared<ReLU<eT> >(ReLU<eT>());
				LOG(LDEBUG) <<  "ReLU";
				break;
			case(LayerTypes::Sigmoid):
				layer_ptr = std::make_shared<Sigmoid<eT> >(Sigmoid<eT>());
				LOG(LDEBUG) <<  "Sigmoid";
				break;

			// convolution
/*			case(LayerTypes::Padding):
				layer_ptr = std::make_shared<Padding<eT> >(Padding<eT>());
				LOG(LERROR) <<  "Padding Layer serialization not implemented (some params are not serialized)!";
				break;
			case(LayerTypes::Pooling):
				layer_ptr = std::make_shared<Pooling<eT> >(Pooling<eT>());
				LOG(LERROR) <<  "Pooling Layer serialization not implemented (some params are not serialized)!";
				break;
			case(LayerTypes::Convolution):
				layer_ptr = std::make_shared<Convolution<eT> >(Convolution<eT>());
				LOG(LERROR) <<  "Convolution Layer serialization not implemented (some params are not serialized)!";
				break;*/

			// cost_function
			case(LayerTypes::Softmax):
				layer_ptr = std::make_shared<Softmax<eT> >(Softmax<eT>());
				LOG(LDEBUG) <<  "Softmax";
				break;

			// fully_connected
			case(LayerTypes::Linear):
				//ar.template register_type<mic::mlnn::Linear>();
				layer_ptr = std::make_shared<Linear<eT> >(Linear<eT>());
				LOG(LDEBUG) <<  "Linear";
				break;
			case(LayerTypes::SparseLinear):
				layer_ptr = std::make_shared<SparseLinear<eT> >(SparseLinear<eT>());
				LOG(LDEBUG) <<  "SparseLinear";
				break;
			case(LayerTypes::Identity):
				layer_ptr = std::make_shared<Identity<eT> >(Identity<eT>());
				LOG(LDEBUG) <<  "Identity";
				break;

			// regularisation
			case(LayerTypes::Dropout):
				layer_ptr = std::make_shared<Dropout<eT> >(Dropout<eT>());
				LOG(LERROR) <<  "Dropout Layer serialization not implemented (some params are not serialized)!";
				break;

			default:
				LOG(LERROR) <<  "Undefined Layer type detected during deserialization!";
			}//: switch

			ar & (*layer_ptr);
			layers.push_back(layer_ptr);
		}//: for

    }

     // The serialization must be splited as load requires to allocate the memory.
     BOOST_SERIALIZATION_SPLIT_MEMBER()

};


/*!
 * Adds layer to neural network - template method specialization for the Regression layer - sets the adequate loss function.
 * @param layer_ptr_ Pointer to the newly created layer.
 * @tparam layer_ptr_ Pointer to the newly created layer.
 */
/*template <>
void MultiLayerNeuralNetwork::pushLayer<mic::mlnn::cost_function::Regression<float> >( mic::mlnn::cost_function::Regression<float>* layer_ptr_){
	layers.push_back(std::shared_ptr <mic::mlnn::cost_function::Regression<float> > (layer_ptr_));
	loss_type = LossFunctionType::RegressionQuadratic;
}*/

/*!
 * Adds layer to neural network - template method specialization for the Softmax layer - sets the adequate loss function.
 * @param layer_ptr_ Pointer to the newly created layer.
 * @tparam layer_ptr_ Pointer to the newly created layer.
 */
/*template <>
void MultiLayerNeuralNetwork::pushLayer<mic::mlnn::cost_function::Softmax<float> >( mic::mlnn::cost_function::Softmax<float>* layer_ptr_){
	layers.push_back(std::shared_ptr <mic::mlnn::cost_function::Softmax<float> > (layer_ptr_));
	loss_type = LossFunctionType::ClassificationEntropy;
}*/


} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_MULTILAYERNEURALNETWORK_HPP_ */
