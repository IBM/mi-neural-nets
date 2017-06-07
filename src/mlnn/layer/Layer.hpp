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
	Cropping,
	Padding,
	MaxPooling,
	// cost_function
	Softmax,
	// fully_connected
	Linear,
	SparseLinear,
	HebbianLinear,
	BinaryCorrelator,
	// regularization
	Dropout
};



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
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param input_depth_ Depth of the input sample.
	 * @param output_height_ Width of the output sample.
	 * @param output_width_ Height of the output sample.
	 * @param output_depth_ Depth of the output sample.
	 * @param layer_type_ Type of the layer.
	 * @param name_ Name of the layer.
	 */
	Layer(size_t input_height_, size_t input_width_, size_t input_depth_,
			size_t output_height_, size_t output_width_, size_t output_depth_,
			LayerTypes layer_type_, std::string name_ = "layer") :
		// Set "reduced" input dimensions.
		input_height(input_height_),
		input_width(input_width_),
		input_depth(input_depth_),
		// Set "reduced" output dimensions.
		output_height(output_height_),
		output_width(output_width_),
		output_depth(output_depth_),
		// Set batch size.
		batch_size(1),
		// Set layer type and name.
		layer_type(layer_type_),
		layer_name(name_),
		// Initialize matrice arrays.
		s("state"),
		g("gradients"),
		p("parameters"),
		m("memory")
	{
		// State.
		s.add ( "x", input_depth*input_height*input_width, batch_size ); 	// inputs
		s.add ( "y", output_depth*output_height*output_width, batch_size); 	// outputs

		// Gradients.
		g.add ( "x", input_depth*input_height*input_width, batch_size ); 	// inputs
		g.add ( "y", output_depth*output_height*output_width, batch_size); 	// outputs

		// Allocate (temporary) memory for "input sample" - column vector.
		m.add ("xs", input_depth*input_height*input_width, 1);
		// Allocate (temporary) memory for "input channel" - column vector.
		m.add ("xc", input_height*input_width, 1);

		// Allocate (temporary) memory for "output sample" - a column vector of all channels of a given sample.
		m.add ("ys", output_depth*output_height*output_width, 1);

		// Allocate (temporary) memory for "output sample" - a column vector.
		m.add ("yc", output_height * output_width, 1);

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
	inline size_t inputSize() {
		return input_height*input_width*input_depth;
	}

	/// Returns size (length) of outputs.
	inline size_t outputSize() {
		return output_height*output_width*output_depth;
	}

	/// Returns size (length) of (mini)batch.
	inline size_t batchSize() {
		return batch_size;
	}

	/// Returns name of the layer.
	inline const std::string name() const {
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
	 * Returns the pointer to gradient (matrix) (or throws an exception!)
	 */
	mic::types::MatrixPtr<eT> getGradient(std::string name_) {
		return g[name_];
	}

	/*!
	 * Sets state (matrix) (or throws an exception!)
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
		case(LayerTypes::MaxPooling):
			return "MaxPooling";
		// cost_function
		case(LayerTypes::Softmax):
			return "Softmax";
		// fully_connected
		case(LayerTypes::Linear):
			return "Linear";
		case(LayerTypes::SparseLinear):
			return "SparseLinear";
		case(LayerTypes::HebbianLinear):
			return "HebbianLinear";
		case(LayerTypes::BinaryCorrelator):
			return "BinaryCorrelator";
		// regularization
		case(LayerTypes::Dropout):
			return "Dropout";
		default:
			return "Undefined";
		}//: switch
	}

	/*!
	 * Stream layer parameters.
	 * @return Ostream object.
	 */
	virtual std::string streamLayerParameters() {
		std::ostringstream os_;
		// Display id/type.
		os_ << "  [" << type() << "]: " << layer_name << ": " << inputSize() << "x" << batch_size << " -> " << outputSize() << "x" << batch_size << "\n";

		// Display dimensions.
		os_<<"    * input_height = " << input_height <<std::endl;
		os_<<"    * input_width = " << input_width <<std::endl;
		os_<<"    * input_channels = " << input_depth <<std::endl;
		os_<<"    * output_height = " << output_height <<std::endl;
		os_<<"    * output_width = " << output_width <<std::endl;
		os_<<"    * output_channels = " << output_depth;

		return os_.str();
	}


	/*!
	 * Stream operator enabling to print neural network.
	 * @param os_ Ostream object.
	 * @param obj_ Tensor object.
	 */
	friend std::ostream& operator<<(std::ostream& os_, Layer& obj_) {
		// Display dimensions.
		os_ << obj_.streamLayerParameters();

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

	/*!
	 * Returns sample from batch. If a given memory array does not contain such a memory pointer - it (lazy) allocates it.
	 * OMP secured - critical section inside.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param array_ Array of matrices where a given matrix (sample) is/will be stored.
	 * @param id_ Sample id (variable prefix).
	 * @param sample_number_ Number of the sample in batch.
	 * @param sample_size_ Size of the sample.
	 */
	mic::types::MatrixPtr<eT> lazyReturnSampleFromBatch (mic::types::MatrixPtr<eT> batch_ptr_, mic::types::MatrixArray<eT> & array_, std::string id_, size_t sample_number_, size_t sample_size_){
		// Generate "unique id" for a given sample.
		std::string sample_id = id_ + std::to_string(sample_number_);
		mic::types::MatrixPtr<eT> sample;

		#pragma omp critical
		{
			if (!array_.keyExists(sample_id)) {
				// Allocate memory.
				array_.add(sample_id, sample_size_, 1);
			}//: if

			// Get array.
			sample = m[sample_id];
			// Copy data.
			(*sample) = batch_ptr_->col(sample_number_);
		}//: end OMP critical section

		// Return it.
		return sample;
	}

	/*!
	 * Returns input sample, with lazy matrix ptr allocation.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param sample_number_ Number of the sample in batch.
	 */
	inline mic::types::MatrixPtr<eT> lazyReturnInputSample (mic::types::MatrixPtr<eT> batch_ptr_, size_t sample_number_){
		return lazyReturnSampleFromBatch(batch_ptr_, m, "xs", sample_number_, Layer<eT>::inputSize());
	}


	/*!
	 * Returns output sample, with lazy matrix ptr allocation.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param sample_number_ Number of the sample in batch.
	 */
	inline mic::types::MatrixPtr<eT> lazyReturnOutputSample (mic::types::MatrixPtr<eT> batch_ptr_, size_t sample_number_){
		return lazyReturnSampleFromBatch(batch_ptr_, m, "ys", sample_number_, Layer<eT>::outputSize());
	}


	/*!
	 * Returns channel from a sample. If a given memory array does not contain such a memory pointer - it (lazy) allocates it.
	 * Assumes that: a) a sample is a column vector and b) there is one "channel memory ptr" for each sample (so the whole batch can be processed in parallel).
	 * OMP secured - critical section inside.
	 * @param sample_ptr Pointer to a sample.
	 * @param array_ Array of matrices where a given matrix (channel) is/will be stored.
	 * @param id_ Channel id (variable prefix).
	 * @param sample_number_ Number of the sample in batch.
	 * @param channel_number_ Number of the channel.
	 * @param height_ Height of the channel.
	 * @param width_ Width of the channel.
	 */
	mic::types::MatrixPtr<eT> lazyReturnChannelFromSample (mic::types::MatrixPtr<eT> sample_ptr_, mic::types::MatrixArray<eT> & array_, std::string id_, size_t sample_number_, size_t channel_number_, size_t height_, size_t width_){
		// Generate "unique id" for a given sample.
		std::string channel_id = id_ + std::to_string(channel_number_);
		mic::types::MatrixPtr<eT> channel;

		#pragma omp critical
		{
			if (!array_.keyExists(channel_id)) {
				// Allocate memory.
				array_.add(channel_id, height_*width_, 1);
			}//: if

			// Get array.
			channel = m[channel_id];
			// Just in case - resize.
			sample_ptr_->resize(sample_ptr_->size(), 1);
			// Copy data.
			(*channel) = sample_ptr_->block(channel_number_*height_*width_, 0, height_*width_, 1);
			// Resize channel.
			channel-> resize(height_, width_);
		}//: end OMP critical section

		// Return it.
		return channel;
	}

	/*!
	 * Returns input channel, with lazy matrix ptr allocation.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param sample_number_ Number of the sample in batch.
	 * @param channel_number_ Number of the channel in sample.
	 */
	inline mic::types::MatrixPtr<eT> lazyReturnInputChannel (mic::types::MatrixPtr<eT> sample_ptr_, size_t sample_number_, size_t channel_number_){
		return lazyReturnChannelFromSample(sample_ptr_, m, "xc", sample_number_, channel_number_, input_height, input_width);
	}


	/*!
	 * Returns output sample, with lazy matrix ptr allocation.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param sample_number_ Number of the sample in batch.
	 * @param channel_number_ Number of the channel in sample.
	 */
	inline mic::types::MatrixPtr<eT> lazyReturnOutputChannel (mic::types::MatrixPtr<eT> sample_ptr_, size_t sample_number_, size_t channel_number_){
		return lazyReturnChannelFromSample(sample_ptr_, m, "yc", sample_number_, channel_number_, output_height, output_width);
	}



	/*!
	 * Allocates memory to a matrix vector (lazy).
	 * @param vector_ Vector that will store the matrices.
	 * @param vector_size Number of matrices to be added.
	 * @param matrix_height_ Height of matrices.
	 * @param matrix_width_ Width of matrices.
	 */
	void lazyAllocateMatrixVector(std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & vector_, size_t vector_size_, size_t matrix_height_, size_t matrix_width_) {
		// Check if memory for the activations was allocated.
		if (vector_.size() != vector_size_) {
			// Free memory.
			vector_.clear();
			// Allocate.
			for (size_t i=0; i < vector_size_; i++) {
				// Allocate memory for activation of every neuron.
				mic::types::MatrixPtr<eT> m = MAKE_MATRIX_PTR(eT, matrix_height_, matrix_width_);
				vector_.push_back(m);
			}//: for
		}//: if
	}


	/*!
	 * Normalizes the matrix. to the range <-1.0, 1.0>, e.g. for the visualization purposes.
	 * @param matrix_ Matrix to be normalized.
	 */
	void normalizeMatrixForVisualization(mic::types::MatrixPtr<eT> matrix_) {
		// Epsilon added for numerical stability.
		//eT eps = 1e-5;
		//eT l2 = matrix_->norm() + eps;

		// Calculate the norm.
		eT max = matrix_->maxCoeff();
		eT min = matrix_->minCoeff();
		eT diff =  0.5*(max - min);

		//std::cout << "before: min:" << (*matrix_).minCoeff() <<" max: " << (*matrix_).maxCoeff() << std::endl;
		// Normalize the inputs to range <0.0, 1.0>.
		// Check if we can normalize.
		if (diff != 0.0) {
			(*matrix_) = matrix_->unaryExpr ( [&] ( eT x ) { return ( (x- min)/diff  - 1.0); } );
			//std::cout << "after: min:" << (*matrix_).minCoeff() <<" max: " << (*matrix_).maxCoeff() << std::endl;
		}//: else: do nothing, all values are ~0 already.

	}


	/*!
	 * Returns activations of input neurons.
	 */
	virtual std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getInputActivations(bool normalize_ = true) {

 		// Allocate memory.
		lazyAllocateMatrixVector(x_activations, input_depth * batch_size, input_height*input_width, 1);

		// Get y batch.
		mic::types::MatrixPtr<eT> batch_x = s['x'];

		// Iterate through filters and generate "activation image" for each one.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get input sample from batch!
			mic::types::MatrixPtr<eT> sample_x = m["xs"];
			(*sample_x) = batch_x->col(ib);

			// Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// Get activation "row".
				mic::types::MatrixPtr<eT> row = x_activations[ib*input_depth + ic];

				// Copy "channel block" from given dx sample.
				(*row) = sample_x->block(ic*input_height*input_width, 0, input_height*input_width, 1);
				row->resize(input_height, input_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for channel
		}//: for batch

		// Return output activations.
		return x_activations;
	}


	/*!
	 * Returns activations of input gradients (dx).
	 */
	virtual std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getInputGradientActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(dx_activations, batch_size * input_depth, input_height*input_width, 1);

		// Get dx batch.
		mic::types::MatrixPtr<eT> batch_dx = g['x'];

		// Iterate through filters and generate "activation image" for each one.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get input sample from batch!
			mic::types::MatrixPtr<eT> sample_dx = m["xs"];
			(*sample_dx) = batch_dx->col(ib);

			// Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// Get dx "row".
				mic::types::MatrixPtr<eT> row = dx_activations[ib*input_depth + ic];

				// Copy "channel block" from given dx sample.
				(*row) = sample_dx->block(ic*input_height*input_width, 0, input_height*input_width, 1);
				row->resize(input_height, input_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for channel
		}//: for batch

		// Return dx activations.
		return dx_activations;
	}


	/*!
	 * Returns activations of output neurons.
	 */
	virtual std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getOutputActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(y_activations, batch_size*output_depth, output_height*output_width, 1);

		// Get y batch.
		mic::types::MatrixPtr<eT> batch_y = s['y'];

		// Iterate through filters and generate "activation image" for each one.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get input sample from batch!
			mic::types::MatrixPtr<eT> sample_y = m["ys"];
			(*sample_y) = batch_y->col(ib);

			// Iterate through output channels.
			for (size_t oc=0; oc< output_depth; oc++) {
				// Get y "row".
				mic::types::MatrixPtr<eT> row = y_activations[ib*output_depth + oc];

				// Copy "channel block" from given dx sample.
				(*row) = sample_y->block(oc*output_height*output_width, 0, output_height*output_width, 1);
				row->resize(output_height, output_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for channel
		}//: for batch

		// Return output activations.
		return y_activations;
	}


	/*!
	 * Returns activations of gradients of output neurons.
	 */
	virtual std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getOutputGradientActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(dy_activations, output_depth*batch_size, output_height*output_width, 1);

		// Get dy batch.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];

		// Iterate through filters and generate "activation image" for each one.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get input sample from batch!
			mic::types::MatrixPtr<eT> sample_dy = m["ys"];
			(*sample_dy) = batch_dy->col(ib);

			// Iterate through output channels.
			for (size_t oc=0; oc< output_depth; oc++) {
				// Get y "row".
				mic::types::MatrixPtr<eT> row = dy_activations[ib*output_depth + oc];

				// Copy "channel block" from given dx sample.
				(*row) = sample_dy->block(oc*output_height*output_width, 0, output_height*output_width, 1);
				row->resize(output_height, output_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for channel
		}//: for batch

		// Return output activations.
		return dy_activations;
	}


protected:

    /// Height of the input (e.g. 28 for MNIST).
    size_t input_height;

    /// Width of the input (e.g. 28 for MNIST).
    size_t input_width;

    /// Number of channels of the input (e.g. 3 for RGB images).
    size_t input_depth;

	/// Number of receptive fields in a single channel - vertical direction.
	size_t output_height;

	/// Number of receptive fields in a single channel - horizontal direction.
	size_t output_width;

	/// Number of filters = number of output channels.
	size_t output_depth;

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

	/// Vector containing activations of input neurons - used in visualization.
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > x_activations;

	/// Vector containing activations of gradients of inputs (dx) - used in visualization.
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > dx_activations;

	/// Vector containing activations of output neurons - used in visualization.
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > y_activations;

	/// Vector containing activations of gradients of outputs (dy) - used in visualization.
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > dy_activations;


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
        ar & input_height;
        ar & input_width;
        ar & input_depth;
        ar & output_height;
        ar & output_width;
        ar & output_depth;
        ar & batch_size;
        ar & layer_type;
        ar & layer_name;
        // Archive four matrix arrays.
        ar & s;
        ar & g;
        ar & p;
        ar & m;
        // TODO: serialize optimization function!
    }


};


} /* namespace mlnn */
} /* namespace mic */


// Just in the case that something important will change in the Layer class - set version.
BOOST_CLASS_VERSION(mic::mlnn::Layer<float>, 2)
BOOST_CLASS_VERSION(mic::mlnn::Layer<double>, 2)

#endif /* SRC_MLNN_LAYER_HPP_ */
