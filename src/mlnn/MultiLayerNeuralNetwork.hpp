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

// Forward declaration of class boost::serialization::access
namespace boost {
namespace serialization {
class access;
}//: serialization
}//: access


namespace mic {
namespace mlnn {

using namespace activation_function;
using namespace cost_function;
using namespace fully_connected;
using namespace convolution;
using namespace regularisation;

/*!
 * \brief Class representing a multi-layer neural network.
 * \author tkornuta/kmrocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT>
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
	 * @param layer_ptr_ Pointer to the layer.
	 * @tparam layer_ptr_ Layer type.
	 */
	template <typename LayerType>
	std::shared_ptr<LayerType> getLayer(size_t index_){
		assert(index_ < layers.size());
		// Cast the pointer to LayerType.
		return std::dynamic_pointer_cast< LayerType >( layers[index_] );
	}

	/*!
	 * Returns n-th layer of the neural network.
	 * @param layer_ptr_ Pointer to the layer.
	 * @tparam layer_ptr_ Layer type.
	 */
	std::shared_ptr<Layer<eT> > getLayer(size_t index_){
		assert(index_ < layers.size());
		// Cast the pointer to LayerType.
		return layers[index_];
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
	 * Returns the size of the input od a given layer.
	 * \param layer_number Number of layer. As default(= -1) returns the size of the input of LAST layer.
	 */
	size_t layerInputsSize(size_t layer_number_ = -1) {
		assert (layer_number_ < layers.size());
		// Last layer.
		if (layer_number_ < 0)
			layer_number_ = layers.size() -1;
		// Return input size.
		return layers[layer_number_]->inputsSize();
	}

	/*!
	 * Returns the size of the output of a given layer.
	 * \param layer_number Number of layer. As default(= -1) returns the size of the output of LAST layer.
	 */
	size_t lastLayerOutputsSize(size_t layer_number_ = -1) {
		assert (layer_number_ < layers.size());
		// Last layer.
		if (layer_number_ < 0)
			layer_number_ = layers.size() -1;
		// Return input size.
		return layers[layer_number_]->outputsSize();
	}

	/*!
	 * Returns the size of the batch of a given layer.
	 * \param layer_number Number of layer. As default(= -1) returns the size of the batch of LAST layer.
	 */
	size_t lastLayerBatchSize(size_t layer_number_ = -1) {
		assert (layer_number_ < layers.size());
		// Last layer.
		if (layer_number_ < 0)
			layer_number_ = layers.size() -1;
		// Return input size.
		return layers[layer_number_]->batchSize();
	}


	/*!
	 * Sets the optimization method.
	 * @tparam omT Optimization method type
	 */
	template<typename omT>
	void setOptimization () {
		// Iterate through layers and set optimization function for each one.
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->setOptimization<omT> ();
	}


	/*!
	 * Performs the network training by updating parameters of all layers according to gradients computed by back-propagation.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_ = 0.0f) {
		// The updates are cumulated for a batch, reduce the alpha rate.
		eT alpha_batch = alpha_/layers[0]->batch_size;

		for (size_t i = 0; i < layers.size(); i++) {
			layers[i]->update(alpha_batch, decay_);
		}//: for
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
	 * Returns the predictions (output of the forward processing) of the last layer in the form of a matrix of size [output_size x batch_size].
	 */
	mic::types::MatrixPtr<eT> getPredictions() {
		return layers.back()->s['y'];
	}

	/*!
	 * Returns the predictions (output of the forward processing) of a given layer in the form of a matrix of size [output_size x batch_size].
	 * @param layer_nr_ Layer number.
	 */
	mic::types::MatrixPtr<eT> getPredictions(size_t layer_nr_) {
		assert(layer_nr_ < layers.size());
		return layers[layer_nr_]->s['y'];
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

    /// Flag denoting whether the layers are interconnected, thus no copying between inputs and outputs of the neighboring layers will be required.
    bool connected;


private:
	// Friend class - required for using boost serialization.
    friend class boost::serialization::access;

    /*!
     * Serialization save - saves the neural net object to archive.
     * @param ar Used archive.
     * @param version Version of the neural net class (not used currently).
     */
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const {
    	// Serialize name.
        ar & name;

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

    	// Deserialize name.
		ar & name;

		// Deserialize the number of layers.
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
			case(LayerTypes::Convolution):
				layer_ptr = std::make_shared<Convolution<eT> >(Convolution<eT>());
				LOG(LERROR) <<  "Convolution Layer serialization not implemented (some params are not serialized)!";
				break;
			case(LayerTypes::Cropping):
				layer_ptr = std::make_shared<Cropping<eT> >(Cropping<eT>());
				LOG(LERROR) <<  "Cropping Layer serialization not implemented (some params are not serialized)!";
				break;
			case(LayerTypes::MaxPooling):
				layer_ptr = std::make_shared<MaxPooling<eT> >(MaxPooling<eT>());
				LOG(LERROR) <<  "MaxPooling Layer serialization not implemented (some params are not serialized)!";
				break;
			case(LayerTypes::Padding):
				layer_ptr = std::make_shared<Padding<eT> >(Padding<eT>());
				LOG(LERROR) <<  "Padding Layer serialization not implemented (some params are not serialized)!";
				break;

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
			case(LayerTypes::HebbianLinear):
				layer_ptr = std::make_shared<HebbianLinear<eT> >(HebbianLinear<eT>());
				LOG(LDEBUG) <<  "HebbianLinear";
				break;

			case(LayerTypes::BinaryCorrelator):
				layer_ptr = std::make_shared<BinaryCorrelator<eT> >(BinaryCorrelator<eT>());
				LOG(LDEBUG) <<  "BinaryCorrelator";
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


} /* namespace mlnn */
} /* namespace mic */

// Just in the case that something important will change in the MLNN class - set version.
BOOST_CLASS_VERSION(mic::mlnn::MultiLayerNeuralNetwork<float>, 2)
BOOST_CLASS_VERSION(mic::mlnn::MultiLayerNeuralNetwork<double>, 2)


#endif /* SRC_MLNN_MULTILAYERNEURALNETWORK_HPP_ */
