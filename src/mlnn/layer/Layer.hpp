/*!
 * \file Layer.hpp
 * \brief Contains a template class representing a layer.
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_LAYER_HPP_
#define SRC_MLNN_LAYER_HPP_

#include <iostream>
#include <string>

#include<types/MatrixTypes.hpp>
#include<types/MatrixArray.hpp>
#include <optimization/OptimizationFunctionTypes.hpp>
#include <optimization/OptimizationArray.hpp>

#include <boost/serialization/serialization.hpp>
// include this header to serialize vectors
#include <boost/serialization/vector.hpp>
// include this header to serialize arrays
#include <boost/serialization/array.hpp>
#include <boost/serialization/version.hpp>

#include <logger/Log.hpp>

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
	// activation
	ELU = 0,
	ReLU,
	Sigmoid,
	// convolution
	Convolution,
	Padding,
	Pooling,
	// cost_function
	Softmax,
	// fully_connected
	Identity,
	Linear,
	SparseLinear,
	// regularization
	Dropout,
	HebbianLinear,
	BinaryCorrelator
};


/*!
 * Template inline function returning square root from (x + eps).
 * \author kmrocki/tkornuta
 */
template <typename eT=float>
inline eT sqrt_eps(const eT x) {
	return sqrt(x + 1e-6);
}

// Forward declaration of MultiLayerNeuralNetwork - required for "lazy connection".
template <typename eT>
class MultiLayerNeuralNetwork;

/*!
 * Template base (abstract) class representing a layer.
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Layer {
public:
	/*!
	 * Default constructor of the layer parent class. Sets the input-output dimensions, layer type and name.
	 * @param input_size_ Size of the input vector.
	 * @param output_size_ Size of the output vector.
	 * @param batch_size_ Size of the batch.
	 * @param layer_type_ Type of the layer.
	 * @param name_ Name of the layer.
	 */
	Layer(size_t input_size_, size_t output_size_, size_t batch_size_, LayerTypes layer_type_, std::string name_ = "layer") :
			input_size(input_size_),
			output_size(output_size_),
			batch_size(batch_size_),
			layer_type(layer_type_),
			layer_name(name_),
			s("state"),
			g("gradients"),
			p("parameters"),
			m("memory")

	{
		// State.
		s.add ( "x", input_size, batch_size ); 	// inputs
		s.add ( "y", output_size, batch_size); 	// outputs

		// Gradients.
		g.add ( "x", input_size, batch_size ); 	// inputs
		g.add ( "y", output_size, batch_size); 	// outputs
	};

	/*!
	 * Virtual destructor - required for the correct destruction of objects of derived classes.
	 */
	virtual ~Layer() {};

	/*!
	 * Abstract method responsible for processing the data from the inputs to outputs. To be overridden in the derived classes.
	 * @param test Test mode - used for dropout-alike regularization techniques.
	 */
	virtual void forward(bool test = false) = 0;

	/*!
	 * Forwards the activations of the neural network.
	 */
	mic::types::MatrixPtr<eT> forward(mic::types::MatrixPtr<eT> x_, bool test = false) {
		// Copy "input" sample/batch.
		(*s["x"]) = (*x_);

		// Call the (abstract, implemented by a given layer) forward pass.
		forward(test);

		// Return "output".
		return s["y"];
	}

	/*!
	 * Abstract method responsible for processing the gradients from outputs to inputs (i.e. in the opposite direction). To be overridden in the derived classes.
	 */
	virtual void backward() = 0;

	/*!
	 * Backward pass - backpropagation.
	 */
	mic::types::MatrixPtr<eT> backward(mic::types::MatrixPtr<eT> dy_) {
		// Copy "output" sample/batch gradient.
		(*g["y"]) = (*dy_);

		// Call the (abstract, implemented by a given layer) backward pass.
		backward();

		// Return "input" gradient.
		return g["x"];
	}

	/*!
	 * Changes the size of the batch. By default it resizes state (x,y) and gradients (x,y).
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_) {
		// Change the "value". (depricated)
		batch_size = batch_size_;
		// Reshape the inputs...
		s["x"]->resize(s["x"]->rows(), batch_size_);
		g["x"]->resize(g["x"]->rows(), batch_size_);
		// ... and outputs.
		s["y"]->resize(s["y"]->rows(), batch_size_);
		g["y"]->resize(g["y"]->rows(), batch_size_);
	}

	/*!
	 * Calculates the numerical gradient.
	 * @param x_ Input vector.
	 * @param target_y_ Target (desired) output.
	 * @param loss_ Loss function.
	 * @param delta_ Delta.
	 * @tparam loss Loss function type.
	 */
	template<typename loss>
	mic::types::MatrixPtr<eT> calculateNumericalGradient(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> target_y_, mic::types::MatrixPtr<eT> param_, loss loss_, eT delta_) {
		// Allocate memory.
		mic::types::MatrixPtr<eT> nGrad = MAKE_MATRIX_PTR(eT, param_->rows(), param_->cols());
		for (size_t i=0; i<(size_t)param_->size(); i++) {
			// Add delta.
			(*param_)[i] += delta_;
			// Calculate loss.
			eT p = loss_.calculateLoss(target_y_, forward(x_));
			// Substract delta.
			(*param_)[i] -= 2*delta_;
			// Calculate loss.
			eT m = loss_.calculateLoss(target_y_, forward(x_));

			// Store numerical gradient.
			(*nGrad)[i] = (p-m)/(2*delta_);
			// Set original value.
			(*param_)[i] += delta_;

		}//: for
		return nGrad;
	}


	/*!
	 * Reset gradients. Virtual empty method - to be implemented by the inherited classes.
	 */
	virtual void resetGrads() {};

	/*!
	 * Performs the update according to the calculated gradients and injected optimization method. Abstract.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	virtual void update(eT alpha_, eT decay_  = 0.0f) = 0;

	/// Returns size (length) of inputs.
	size_t inputSize() {
		return input_size;
	}

	/// Returns size (length) of outputs.
	size_t outputSize() {
		return output_size;
	}

	/// Returns size (length) of (mini)batch.
	size_t batchSize() {
		return batch_size;
	}

	/// Returns name of the layer.
	const std::string name() const {
		return layer_name;
	}

	/*!
	 * Returns the pointer to a parameter (matrix) (or throws an exception!)
	 */
	mic::types::MatrixPtr<eT> getParam(std::string name_) {
		return p[name_];
	}

	/*!
	 * Returns the pointer to state (matrix) (or throws an exception!)
	 */
	mic::types::MatrixPtr<eT> getState(std::string name_) {
		return s[name_];
	}

	/*!
	 * Returns the pointer to state (matrix) (or throws an exception!)
	 */
	void setState(std::string name_, mic::types::MatrixPtr<eT> mat_ptr_) {
		(*s[name_]) = (*mat_ptr_);
	}

	/*!
	 * Sets the optimization method.
	 * @tparam omT Optimization method type
	 */
	template<typename omT>
	void setOptimization () {
		// Remove all previous optimization functions.
		opt.clear();

		// Iterate through parameters and add a separate optimization function for each parameter.
		for (auto& i: p.keys()) {
			opt.add(
					i.first,
					std::make_shared< omT > (omT ( (p[i.second])->rows(), (p[i.second])->cols() ))
					);
		}//: for keys
	}

	/*!
	 * Returns the type of layer.
	 */
	const std::string type() const {
		switch(layer_type) {
		// activation
		case(LayerTypes::ELU):
			return "ELU";
		case(LayerTypes::ReLU):
			return "ReLU";
		case(LayerTypes::Sigmoid):
			return "Sigmoid";
		// convolution
		case(LayerTypes::Convolution):
			return "Convolution";
		case(LayerTypes::Padding):
			return "Padding";
		case(LayerTypes::Pooling):
			return "Pooling";
		// cost_function
		case(LayerTypes::Softmax):
			return "Softmax";
		// fully_connected
		case(LayerTypes::Identity):
			return "Identity";
		case(LayerTypes::Linear):
			return "Linear";
		case(LayerTypes::SparseLinear):
			return "SparseLinear";
		// regularization
		case(LayerTypes::Dropout):
			return "Dropout";
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
		os_ << "  [" << obj_.type() << "]: " << obj_.layer_name << ": " << obj_.input_size << "x" << obj_.batch_size << " -> " << obj_.output_size << "x" << obj_.batch_size << "\n";
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
	size_t input_size;

	/// Size (length) of outputs.
	size_t output_size;

	/// Size (length) of (mini)batch.
	size_t batch_size;

	/// Type of the layer.
	LayerTypes layer_type;

	/// Name (identifier of the type) of the layer.
	std::string layer_name;


	/// States - contains input [x] and output [y] matrices.
	mic::types::MatrixArray<eT> s;

	/// Gradients - contains input [x] and output [y] matrices.
	mic::types::MatrixArray<eT> g;

	/// Parameters - parameters of the layer, to be used by the derived classes.
	mic::types::MatrixArray<eT> p;

	/// Memory - a list of temporal parameters, to be used by the derived classes.
	mic::types::MatrixArray<eT> m;

	/// Array of optimization functions.
	mic::neural_nets::optimization::OptimizationArray<eT> opt;

	/*!
	 * Protected constructor, used only by the derived classes during the serialization. Empty!!
	 */
	Layer () { }

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;
	template<typename tmp> friend class BackpropagationNeuralNetwork;
	template<typename tmp> friend class HebbianNeuralNetwork;

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
        ar & input_size;
        ar & output_size;
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


// Just in the case that something important will change in the Layer class - set version.
BOOST_CLASS_VERSION(mic::mlnn::Layer<float>, 2)
BOOST_CLASS_VERSION(mic::mlnn::Layer<double>, 2)

#endif /* SRC_MLNN_LAYER_HPP_ */
