/*!
 * \file Layer.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_LAYER_HPP_
#define SRC_MLNN_LAYER_HPP_

#include <iostream>
#include <string>

#include<types/MatrixTypes.hpp>
#include<types/MatrixArray.hpp>

#include <boost/serialization/serialization.hpp>
// include this header to serialize vectors
#include <boost/serialization/vector.hpp>
// include this header to serialize arrays
#include <boost/serialization/array.hpp>
#include <boost/serialization/version.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {
namespace serialization {
class access;
}//: serialization
}//: access


namespace mic {
namespace mlnn {

/*!
 * \brief Enumeration of possible layer types.
 * \author tkornuta
 */
enum class LayerTypes : short
{
	Linear = 0,
	Pooling,
	Convolution,
	Sigmoid,
	Identity,
	ReLU,
	ELU,
	Softmax,
	Dropout,
	Padding,
	Regression,
	SparseLinear
};


/*class Linear;
class Pooling;
class Convolution;
class Sigmoid;
class Identity;
class ReLU;
class ELU;
class Softmax;
class Dropout;
class Padding;
class Regression;*/


// Forward declaration of MultiLayerNeuralNetwork
class MultiLayerNeuralNetwork;

inline float sqrt_eps(const float x) {
	return sqrtf(x + 1e-6);
}

/*!
 * abstract
 * \author krocki/tkornuta
 */
class Layer {
public:
	/*!
	 * Default constructor of the layer parent class. Sets the input-output dimensions, layer type and name.
	 * @param inputs_size_ Size of the input vector.
	 * @param outputs_size_ Size of the output vector.
	 * @param batch_size_ Size of the batch.
	 * @param layer_type_ Type of the layer.
	 * @param name_ Name of the layer.
	 */
	Layer(size_t inputs_size_, size_t outputs_size_, size_t batch_size_, LayerTypes layer_type_, std::string name_ = "layer");

	/*!
	 * Abstract method responsible for processing the data from the inputs to outputs. To be overridden in the derived classes.
	 * @param test Test mode - used for dropout-alike regularization techniques.
	 */
	virtual void forward(bool test = false) = 0;

	/*!
	 * Abstract method responsible for processing the gradients from outputs to inputs (i.e. in the opposite direction). To be overridden in the derived classes.
	 */
	virtual void backward() = 0;


	/*!
	 * Changes the size of the batch. By default it resizes
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_);

	/*!
	 * Reset gradients.
	 */
	virtual void resetGrads() {};


	virtual void applyGrads(double alpha_, double decay_) {};

	/*!
	 * Virtual destructor - required for the correct destruction of objects of derived classes.
	 */
	virtual ~Layer() {};

	/// Returns size (length) of inputs.
	size_t inputsSize();

	/// Returns size (length) of outputs.
	size_t outputsSize();

	/// Returns size (length) of (mini)batch.
	size_t batchSize();

	/// Returns name of the layer.
	const std::string name() const {
		return layer_name;
	}

	/*!
	 * Returns the pointer to a parameter (matrix of floats) (or throws an exception!)
	 */
	mic::types::MatrixXfPtr getParam(std::string name_) {
		return p[name_];
	}

	/*!
	 * Returns the type of layer.
	 */
	const std::string type() const {
		switch(layer_type) {
		case(LayerTypes::Linear):
			return "Linear";
		case(LayerTypes::Pooling):
			return "Pooling";
		case(LayerTypes::Convolution):
			return "Convolution";
		case(LayerTypes::Sigmoid):
			return "Sigmoid";
		case(LayerTypes::Identity):
			return "Identity";
		case(LayerTypes::ReLU):
			return "ReLU";
		case(LayerTypes::ELU):
			return "ELU";
		case(LayerTypes::Softmax):
			return "Softmax";
		case(LayerTypes::Dropout):
			return "Dropout";
		case(LayerTypes::Padding):
			return "Padding";
		case(LayerTypes::Regression):
			return "Regression";
		default:
			return "Undefined";
		}//: switch
	}

	/*!
	 * Stream operator enabling to print neural network.
	 * @param os_ Ostream object.
	 * @param obj_ Tensor object.
	 */
	friend std::ostream& operator<<(std::ostream& os_, Layer& obj_) {
		// Display dimensions.
		os_ << "  [" << obj_.type() << "]: " << obj_.layer_name << ": " << obj_.inputs_size << "x" << obj_.batch_size << " -> " << obj_.outputs_size << "x" << obj_.batch_size << "\n";
		// Display inputs.
		os_ << "    [" << obj_.s.name() << "]:\n";
		for (auto& i: obj_.s.keys()) {
			// Display elements.
			os_ << "      [" << i.first << "]: ";
			os_ << (obj_.s[i.second])->cols() << "x" << (obj_.s[i.second])->rows() << std::endl;
		}//: for keys

		// Display gradients.
		os_ << "    [" << obj_.g.name() << "]:\n";
		for (auto& i: obj_.g.keys()) {
			// Display elements.
			os_ << "      [" << i.first << "]: ";
			os_ << (obj_.g[i.second])->cols() << "x" << (obj_.g[i.second])->rows() << std::endl;
		}//: for keys

		// Display parameters.
		os_ << "    [" << obj_.p.name() << "]:\n";
		for (auto& i: obj_.p.keys()) {
			// Display elements.
			os_ << "      [" << i.first << "]: ";
			os_ << (obj_.p[i.second])->cols() << "x" << (obj_.p[i.second])->rows() << std::endl;
		}//: for keys

		// Display gradients.
		os_ << "    [" << obj_.m.name() << "]:\n";
		for (auto& i: obj_.m.keys()) {
			// Display elements.
			os_ << "      [" << i.first << "]: ";
			os_ << (obj_.m[i.second])->cols() << "x" << (obj_.m[i.second])->rows() << std::endl;
		}//: for keys

		return os_;
	}

protected:

	/// Size (length) of inputs.
	size_t inputs_size;

	/// Size (length) of outputs.
	size_t outputs_size;

	/// Size (length) of (mini)batch.
	size_t batch_size;

	/// Type of the layer.
	LayerTypes layer_type;

	/// Name (identifier of the type) of the layer.
	std::string layer_name;


	/// States - contains input [x] and output [y] matrices.
	mic::types::MatrixArray<float> s;

	/// Gradients - contains input [x] and output [y] matrices.
	mic::types::MatrixArray<float> g;

	/// Parameters - parameters of the layer, to be used by the derived classes.
	mic::types::MatrixArray<float> p;

	/// Memory - a list of temporal parameters, to be used by the derived classes.
	mic::types::MatrixArray<float> m;

	// Adds the nn class the access to protected fields of class layer.
	friend class MultiLayerNeuralNetwork;

	/*!
	 * Protected constructor, used only by the derived classes during the serialization. Empty!!
	 */
	Layer () { }

private:

	// Friend class - required for using boost serialization.
    friend class boost::serialization::access;

    /*!
     * Serializes the layer to and from archive.
     * @param ar Used archive.
     * @param version Version of the layer class (not used currently).
     */
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        // Archive parameters.
        ar & inputs_size;
        ar & outputs_size;
        ar & batch_size;
        ar & layer_type;
        ar & layer_name;
        // Archive four matrix arrays.
        ar & s;
        ar & g;
        ar & p;
        ar & m;
    }


};


} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_LAYER_HPP_ */
