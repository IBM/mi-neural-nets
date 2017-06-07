/*!
 * \file Pooling.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_POOLING_HPP_
#define SRC_MLNN_POOLING_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace convolution {


/*!
 * \brief Layer performing max pooling.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class MaxPooling : public mic::mlnn::Layer<eT> {
public:


	/*!
	 * Creates a max pooling layer.
	 * @param input_height_ Height of the input sample.
	 * @param input_width_ Width of the input sample.
	 * @param depth_ Depth of the input/output sample.
	 * @param window_size_ Max pooling window in each channel (width and height).
	 * @param name_ Name of the layer.
	 */
	MaxPooling(size_t input_height_, size_t input_width_, size_t depth_,
			size_t window_size_,
			std::string name_ = "MaxPooling") :
		Layer<eT>::Layer(input_height_, input_width_, depth_,
				(input_height_ /window_size_), (input_width_ / window_size_), depth_,
				LayerTypes::MaxPooling, name_),
				window_size(window_size_)
	{
		// Mapping from input to output - every cell will contain address of input image.
		m.add("pooling_map", Layer<eT>::outputSize(), 1);
	};

	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~MaxPooling() {};

	/*!
	 * Changes the size of the batch - calls base Layer class resize and additionally resizes the cache size.
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_) {
		// Call base Layer resize.
		Layer<eT>::resizeBatch(batch_size_);

		// Reshape pooling mask and map.
		m["pooling_map"]->resize(Layer<eT>::outputSize(), batch_size_);

	}


	void forward(bool test_ = false) {
		LOG(LTRACE) << "MaxPooling::forward\n";

		// Get pointer to input batch.
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "forward batch_x=\n" << (*batch) << std::endl;
		//std::cout << "forward input x activation: min:" << (*batch_x).minCoeff() <<" max: " << (*batch_x).maxCoeff() << std::endl;

		// Get pointer to output batch - so the results will be stored!
		mic::types::MatrixPtr<eT> batch_y = s['y'];
		// Reset the whole batch.
		batch_y->setZero();

		// Get pointer to the mask.
		mic::types::MatrixPtr<eT> pooling_map = m["pooling_map"];
		pooling_map->setZero();

		// TODO: should work for more channels - but requires testing!
		//assert(input_depth == 1);

		// Iterate through batch - cannot be done in parallel:
		// * pooling mask is shared (ok)
		// * tmp variables storing input samples/channels are multiplied and OMP secured (ok)
		// * pooling mask and output batch are edited on different addresses - OMP secured (ok)
		#pragma omp parallel for
		for (size_t ib = 0; ib < batch_size; ib++) {
			// Get input sample from batch.
			mic::types::MatrixPtr<eT> xs = lazyReturnInputSample(batch_x, ib);

			// Iterate through input/output channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// Get input channel from input sample.
				mic::types::MatrixPtr<eT> xc = lazyReturnInputChannel(xs, ib, ic);

				// Iterate through "blocks" in a given channel.
				for (size_t ih=0, oh=0; ih< input_height; ih+=window_size, oh++) {
					for (size_t iw=0, ow=0; iw< input_width; iw+=window_size, ow++) {

					#pragma omp critical
					{
						// Get location of max element.
						size_t maxRow, maxCol;
						eT max_val = xc->block(ih, iw, window_size, window_size).maxCoeff(&maxRow, &maxCol);

						//std::cout << "xc->block(ih, iw, window_size, window_size) = " <<xc->block(ih, iw, window_size, window_size) <<std::endl;
						//std::cout << " maxRow = " << maxRow << " maxCol = "<< maxCol << " max_val = "<< max_val << std::endl;

						// Calculate "absolute addresses.
						size_t ia = (ib * Layer<eT>::inputSize()) + ic * input_height * input_width + (iw + maxCol) * input_height + (ih + maxRow);
						size_t oa = (ib * Layer<eT>::outputSize()) + ic * output_height * output_width + (ow) * output_height + (oh);
						/*std::cout << " ih = " << ih << " iw = " << iw << " ia = " << ia << std::endl;
						std::cout << " oh = " << oh << " ow = " << ow << " oa = " << oa << std::endl;*/

						// Map output to input.
						(*pooling_map)[oa] = ia;

						// Copy value to output.
						(*batch_y)[oa] = max_val;
					}

					}//: for width
				}//: for width
			}//: for channels
		}//: for batch
		LOG(LTRACE) << "MaxPooling::forward end\n";
	}

	/*!
	 * Backward pass.
	 */
	void backward() {
		LOG(LTRACE) << "MaxPooling::backward\n";

		// Get pointer to dy batch.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];

		// Get pointer to dx batch.
		mic::types::MatrixPtr<eT> batch_dx = g['x'];
		batch_dx->setZero();

		mic::types::MatrixPtr<eT> pooling_map = m["pooling_map"];

		// Iterate through batch.
		#pragma omp parallel for
		for (size_t oi = 0; oi < batch_size * Layer<eT>::outputSize(); oi++) {

			// Map outputs to inputs.
			//std::cout << " oi = " << oi << " (*pooling_map)[oi] = " << (*pooling_map)[oi] << std::endl;
			(*batch_dx)[(size_t)(*pooling_map)[oi]] = (*batch_dy)[oi];

		}//: for batch

		LOG(LTRACE) << "MaxPooling::backward end\n";
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

    using Layer<eT>::lazyReturnInputSample;
    using Layer<eT>::lazyReturnOutputSample;
    using Layer<eT>::lazyReturnInputChannel;
    using Layer<eT>::lazyReturnOutputChannel;

	/*!
	 * Size of the pooling window.
	 */
	size_t window_size;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp1> friend class MultiLayerNeuralNetwork;


	/*!
	 * Private constructor, used only during the serialization.
	 */
	MaxPooling<eT>() : Layer<eT> () { }

};

} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_POOLING_HPP_ */
