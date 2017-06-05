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
		// Allocate pooling mask - one matrix for the whole batch.
		m.add("pooling_mask", input_height*input_width*input_depth, 1);
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

		// Reshape pooling mask.
		m["pooling_mask"]->resize(Layer<eT>::inputSize(), batch_size_);

	}

	/*!
	 * Returns sample from batch. If a given memory array does not contain such a memory pointer - it (lazy) allocates it.
	 * @param batch_ptr_ Pointer to a batch.
	 * @param array_ Array of matrices where a given matrix (sample) is/will be stored.
	 * @param id_ Sample id (variable prefix).
	 * @param sample_number_ Number of the sample in batch.
	 * @param sample_size_ Size of the sample.
	 */
	mic::types::MatrixPtr<eT> lazyReturnSampleFromBatch (mic::types::MatrixPtr<eT> batch_ptr_, mic::types::MatrixArray<eT> & array_, std::string id_, size_t sample_number_, size_t sample_size_){
		// Generate "unique id" for a given sample.
		std::string sample_id = id_ + std::to_string(sample_number_);
		if (!array_.keyExists(sample_id)) {
			// Allocate memory.
			array_.add(sample_id, sample_size_, 1);
		}//: if

		// Get array.
		mic::types::MatrixPtr<eT> sample = m[sample_id];
		// Copy data.
		(*sample) = batch_ptr_->col(sample_number_);
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
		if (!array_.keyExists(channel_id)) {
			// Allocate memory.
			array_.add(channel_id, height_*width_, 1);
		}//: if

		// Get array.
		mic::types::MatrixPtr<eT> channel = m[channel_id];
		// Just in case - resize.
		sample_ptr_->resize(sample_ptr_->size(), 1);
		// Copy data.
		(*channel) = sample_ptr_->block(channel_number_*height_*width_, 0, height_*width_, 1);
		// Resize channel.
		channel-> resize(height_, width_);
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


	void forward(bool test_ = false) {

		// Get pointer to input batch.
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "forward batch_x=\n" << (*batch) << std::endl;
		//std::cout << "forward input x activation: min:" << (*batch_x).minCoeff() <<" max: " << (*batch_x).maxCoeff() << std::endl;

		// Get pointer to output batch - so the results will be stored!
		mic::types::MatrixPtr<eT> batch_y = s['y'];
		// Reset the whole batch.
		batch_y->setZero();

		// Get pointer to the mask.
		mic::types::MatrixPtr<eT> pooling_mask = m["pooling_mask"];
		// Reset mask.
		pooling_mask->setZero();

		// TODO: should work for more channels - but requires testing!
		assert(input_depth == 1);

		// Iterate through batch - cannot be done in parallel:
		// * pooling mask is shared (ok)
		// * tmp variables storing input/outpu samples/channels must be multiplied (not ok)
		//#pragma omp parallel for
		for (size_t ib = 0; ib < batch_size; ib++) {

			// Get input sample from batch.
			mic::types::MatrixPtr<eT> xs = lazyReturnInputSample(batch_x, ib);

			// Get output sample from batch.
			mic::types::MatrixPtr<eT> ys = lazyReturnOutputSample(batch_y, ib);

			// Iterate through input/output channels.
			for (size_t ic=0; ic< input_depth; ic++) {

				// Get input channel from input sample.
				mic::types::MatrixPtr<eT> xc  = lazyReturnInputChannel(xs, ib, ic);

				// Get output channel from output sample.
				mic::types::MatrixPtr<eT> yc  = lazyReturnOutputChannel(ys, ib, ic);
				yc->setZero();

				// Iterate through "blocks" in a given channel.
				for (size_t ih=0, oh=0; ih< input_height; ih+=window_size, oh++) {
					for (size_t iw=0, ow=0; iw< input_width; iw+=window_size, ow++) {

					// Get location of max element.
					size_t maxRow, maxCol;
					eT max_val = xc->block(ih, iw, window_size, window_size).maxCoeff(&maxRow, &maxCol);

					//std::cout << "xc->block(ih, iw, window_size, window_size) = " <<xc->block(ih, iw, window_size, window_size) <<std::endl;
					std::cout << " maxRow = " << maxRow << " maxCol = "<< maxCol << " max_val = "<< max_val << std::endl;

					// Calculate "absolute addresses.
					size_t ia = (ib * Layer<eT>::inputSize()) + ic * input_height * input_width + (iw + maxCol) * input_height + (ih + maxRow);
					size_t oa = (ib * Layer<eT>::outputSize()) + ic * output_height * output_width + (ow) * output_height + (oh);
					std::cout << " ih = " << ih << " iw = " << iw << " ia = " << ia << std::endl;
					std::cout << " oh = " << oh << " ow = " << ow << " oa = " << oa << std::endl;


					// Set mask in a given address to 1.
					(*pooling_mask)[ia] = 1;

					// Copy value to output.
					(*batch_y)[oa] = max_val;

					}//: for width
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
		batch_dx->setZero();

		/*// Iterate through batch.
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
		}//: for batch*/

	}


	/*!
	 * Processes forwards data for a single channel.
	 * @param x Channel
	 * @param cache
	 * @param window_size
	 * @return
	 */
/*	mic::types::MatrixXf forwardChannel(mic::types::MatrixXf& x_channel_, mic::types::MatrixXf& cache_) {

		size_t image_size = sqrt(x_channel_.rows());
		// size_t y_width = image_size - window_size + 1;
		//disjoint
		size_t y_width = image_size / window_size;
		mic::types::MatrixXf y = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(y_width * y_width, x_channel_.cols());

		for (size_t i = 0; i < (size_t) y.cols(); i++) { //images in a batch

			mic::types::MatrixXf image = (Eigen::MatrixXf)x_channel_.col(i);
			mic::types::MatrixXf local_cache = (Eigen::MatrixXf)cache_.col(i);

			image.resize(image_size, image_size);
			local_cache.resize(image_size, image_size);

			mic::types::MatrixXf out = (Eigen::MatrixXf)y.col(i);
			out.resize(sqrt(out.size()), sqrt(out.size()));

			out.setZero();
			local_cache.setZero();

			// __pooling_2D(out, image, window_size);
			poolDisjoint2D(out, local_cache, image);

			out.resize(out.size(), 1);
			local_cache.resize(local_cache.size(), 1);

			y.col(i) = out;
			cache_.col(i) = local_cache;
		}

		return y;

	}*/

	/*!
	 *
	 * @param out
	 * @param cache
	 * @param image
	 */
/*	void poolDisjoint2D(mic::types::MatrixXf& out, mic::types::MatrixXf& cache, mic::types::MatrixXf& image)  {

		for (size_t yi = 0; yi < (size_t)out.rows(); yi++) {

			for (size_t yj = 0; yj < (size_t)out.cols(); yj++) {

				float value = -INFINITY;
				size_t idx_j;
				size_t idx_i;

				for (size_t ki = 0; ki < window_size; ki++) {

					for (size_t kj = 0; kj < window_size; kj++) {

						float pix = image(yj * window_size + kj, yi * window_size + ki);

						if (value < pix) {

							value = pix;
							idx_j = kj;
							idx_i = ki;

						}

					}
				}

				out(yj, yi) = (float)value;
				cache(yj * window_size + idx_j, yi * window_size + idx_i) = 1;
			}
		}
	}*/

/*	void backward() {

		//	pooling_backward(channels, *(g['x']), *(g['y']), cache, window_size);

		#pragma omp parallel for
		for (size_t k = 0; k < channels; k++) {
			// Prepare submatrices.
			mic::types::MatrixXf dy_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['y']->rows() / channels, g['y']->cols());
			mic::types::MatrixXf dx_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['x']->rows() / channels, g['x']->cols());
			mic::types::MatrixXf cache_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['x']->rows() / channels, g['x']->cols());

			// Get block and cache related to a given channel.
			dy_map = g['y']->block(dy_map.rows() * k, 0, dy_map.rows(), dy_map.cols());
			cache_map = pooling_mask.block(cache_map.rows() * k, 0, cache_map.rows(), cache_map.cols());

			// Perform backward channel pooling.
			dx_map = backwardChannel(dy_map, cache_map).array();

			// Map results to inputs.
			g['x']->block(dx_map.rows() * k, 0, dx_map.rows(), dx_map.cols()) = dx_map;
			}//: for

		}*/

	/*!
	 * Processes backwards data for a single channel.
	 * @param dy
	 * @param cache_
	 * @return
	 */
/*	mic::types::MatrixXf backwardChannel(mic::types::MatrixXf& dy, mic::types::MatrixXf& cache_) {

		mic::types::MatrixXf dx = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(cache_.rows(), cache_.cols());

		for (size_t i = 0; i < (size_t)dy.cols(); i++) {

			mic::types::MatrixXf dy_local = (Eigen::MatrixXf)dy.col(i);
			mic::types::MatrixXf cache_local = (Eigen::MatrixXf)cache_.col(i);
			mic::types::MatrixXf dx_local = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(cache_local.rows(), cache_local.cols());
			dy_local.resize(sqrt(dy_local.size()), sqrt(dy_local.size()));
			cache_local.resize(sqrt(cache_local.size()), sqrt(cache_local.size()));
			dx_local.resize(sqrt(cache_local.size()), sqrt(cache_local.size()));
			dx_local.setZero();

			unpoolDisjoint2D(dx_local, cache_local, dy_local);

			dx_local.resize(dx_local.size(), 1);
			dx.col(i) = dx_local;
		}

		return dx;
	}*/

	/*!
	 *
	 * @param dx
	 * @param cache
	 * @param dy
	 */
/*	void unpoolDisjoint2D(mic::types::MatrixXf& dx, mic::types::MatrixXf& cache, mic::types::MatrixXf& dy)  {

		for (size_t xi = 0; xi < (size_t)dx.rows(); xi++) {

			for (size_t xj = 0; xj < (size_t)dx.cols(); xj++) {

				dx(xj, xi) = cache(xj, xi) * dy(xj / window_size, xi / window_size);

			}
		}

	}*/

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
