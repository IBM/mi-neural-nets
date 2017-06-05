/*!
 * \file Padding.hpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#ifndef SRC_MLNN_PADDING_HPP_
#define SRC_MLNN_PADDING_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

/*!
 * \brief Class implementing padding operation - expanding the size of image (matrix) by a margin of n pixels on every image side.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Padding : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates a Padding layer.
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param depth_ Depth of the input/output sample.
	 * @param padding_ Padding added to each channel (width and height).
	 * @param name_ Name of the layer.
	 */
	Padding(size_t input_height_, size_t input_width_, size_t depth_,
			size_t padding_,
			std::string name_ = "Padding") :
		Layer<eT>::Layer(input_height_, input_width_, depth_,
				(input_height_ + 2*padding_), (input_width_ + 2*padding_), depth_,
				LayerTypes::Padding, name_),
				padding(padding_)
	{

	}

	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~Padding() { }

	/*!
	 * Performs forward pass - add padding.
	 */
	void forward(bool test = false) {

		// Get pointer to input batch.
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "forward batch_x=\n" << (*batch) << std::endl;
		//std::cout << "forward input x activation: min:" << (*batch_x).minCoeff() <<" max: " << (*batch_x).maxCoeff() << std::endl;

		// Get pointer to output batch - so the results will be stored!
		mic::types::MatrixPtr<eT> batch_y = s['y'];
		batch_y->setZero();

		// TODO: should work for more channels - but requires testing!
		assert(input_depth == 1);

		// Iterate through batch.
		#pragma omp parallel for
		for (size_t bi = 0; bi < batch_size; bi++) {

			// Iterate through input/output channels.
			for (size_t ic=0; ic< input_depth; ic++) {

				// Iterate through "blocks" o in channels.
				for (size_t iw=0; iw< input_width; iw++) {
					// Calculate addresses.
					size_t ia = ic * (input_width) * (input_height) + iw*(input_height);
					size_t oa = ic * (input_width + 2*padding) * (input_height + 2*padding) + (iw+padding)*(input_height + 2*padding) + padding;
					//std::cout << " iw = " << iw << " ia = " << ia << " oa = " << oa << std::endl;

					// Copy "height" block from input to output.
					batch_y->block(oa, bi, input_height, 1) =
						batch_x->block(ia, bi, input_height, 1);

				}//: for width
			}//: for channels
		}//: for batch
	}

	/*!
	 * Backward pass.
	 */
	void backward() {

		// Get pointer to dy batch.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];

		// Get pointer to dx batch.
		mic::types::MatrixPtr<eT> batch_dx = g['x'];


		// Iterate through batch.
		#pragma omp parallel for
		for (size_t bi = 0; bi < batch_size; bi++) {

			// Iterate through input/output channels.
			for (size_t ic=0; ic< input_depth; ic++) {

				// Iterate through "blocks" o in channels.
				for (size_t iw=0; iw< input_width; iw++) {
					// Calculate addresses.
					size_t ia = ic * (input_width) * (input_height) + iw*(input_height);
					size_t oa = ic * (input_width + 2*padding) * (input_height + 2*padding) + (iw+padding)*(input_height + 2*padding) + padding;

					// Copy "height" block from input to output.
					batch_dx->block(ia, bi, input_height, 1) = batch_dy->block(oa, bi, input_height, 1);

				}//: for width

			}//: for channels
		}//: for batch

	}

	/*!
	 * Update - empty as this layer is not "plastic".
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_  = 0.0f) { }

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;

    // Uncover "sizes" for visualization.
    using Layer<eT>::input_height;
    using Layer<eT>::input_width;
    using Layer<eT>::input_depth;
	using Layer<eT>::output_height;
	using Layer<eT>::output_width;
	using Layer<eT>::output_depth;
    using Layer<eT>::batch_size;

    // Size of padding.
	size_t padding;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp1> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Padding<eT>() : Layer<eT> () { }



};


} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_PADDING_HPP_ */
