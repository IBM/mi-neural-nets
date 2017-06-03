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
 * \author krocki/tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Pooling : public mic::mlnn::Layer<eT> {
public:

	Pooling<eT>(size_t inputs_, size_t window_size_, size_t channels_, std::string name_ = "Pooling") :
		Layer<eT>(inputs_, channels_ * (sqrt(inputs_ / channels_) / window_size_) * (sqrt(inputs_ / channels_) / window_size_), 1, LayerTypes::Pooling, name_),
		channels(channels_), window_size(window_size_) {

		// Alloc cache.
		cache = mic::types::Matrix<eT>::Zero(inputs_, 1);
	};

	virtual ~Pooling() {};

	/*!
	 * Changes the size of the batch - calls base Layer class resize and additionally resizes the cache size.
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_) {
		// Call base Layer resize.
		Layer<eT>::resizeBatch(batch_size_);

		// Reshape cache.
		cache.resize(cache.rows(), batch_size_);

	}

	void forward(bool test_ = false) {

		//	pooling_forward(channels, *(s['x']), *(s['y']), cache, window_size);

		#pragma omp parallel for
		for (size_t k = 0; k < channels; k++) {
			// Prepare submatrices.
			mic::types::MatrixXf y_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(s['y']->rows() / channels, s['y']->cols());
			mic::types::MatrixXf inputs = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(s['x']->rows() / channels, s['x']->cols());
			mic::types::MatrixXf cache_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(s['x']->rows() / channels, s['x']->cols());

			// Get block related to a given channel.
			inputs = s['x']->block(inputs.rows() * k, 0, inputs.rows(), inputs.cols());

			// Perform forward channel pooling.
			y_map = forwardChannel(inputs, cache_map).array();

			// Map results to output.
			s['y']->block(y_map.rows() * k, 0, y_map.rows(), y_map.cols()) = y_map;

			// Save to cache.
			cache.block(cache_map.rows() * k, 0, cache_map.rows(), cache_map.cols()) = cache_map;

			}//: for
		}

	/*!
	 * Processes forwards data for a single channel.
	 * @param x Channel
	 * @param cache
	 * @param window_size
	 * @return
	 */
	mic::types::MatrixXf forwardChannel(mic::types::MatrixXf& x_channel_, mic::types::MatrixXf& cache_) {

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

	}

	/*!
	 *
	 * @param out
	 * @param cache
	 * @param image
	 */
	void poolDisjoint2D(mic::types::MatrixXf& out, mic::types::MatrixXf& cache, mic::types::MatrixXf& image)  {

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
	}

	void backward() {

		//	pooling_backward(channels, *(g['x']), *(g['y']), cache, window_size);

		#pragma omp parallel for
		for (size_t k = 0; k < channels; k++) {
			// Prepare submatrices.
			mic::types::MatrixXf dy_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['y']->rows() / channels, g['y']->cols());
			mic::types::MatrixXf dx_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['x']->rows() / channels, g['x']->cols());
			mic::types::MatrixXf cache_map = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(g['x']->rows() / channels, g['x']->cols());

			// Get block and cache related to a given channel.
			dy_map = g['y']->block(dy_map.rows() * k, 0, dy_map.rows(), dy_map.cols());
			cache_map = cache.block(cache_map.rows() * k, 0, cache_map.rows(), cache_map.cols());

			// Perform backward channel pooling.
			dx_map = backwardChannel(dy_map, cache_map).array();

			// Map results to inputs.
			g['x']->block(dx_map.rows() * k, 0, dx_map.rows(), dx_map.cols()) = dx_map;
			}//: for

		}

	/*!
	 * Processes backwards data for a single channel.
	 * @param dy
	 * @param cache_
	 * @return
	 */
	mic::types::MatrixXf backwardChannel(mic::types::MatrixXf& dy, mic::types::MatrixXf& cache_) {

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
	}

	/*!
	 *
	 * @param dx
	 * @param cache
	 * @param dy
	 */
	void unpoolDisjoint2D(mic::types::MatrixXf& dx, mic::types::MatrixXf& cache, mic::types::MatrixXf& dy)  {

		for (size_t xi = 0; xi < (size_t)dx.rows(); xi++) {

			for (size_t xj = 0; xj < (size_t)dx.cols(); xj++) {

				dx(xj, xi) = cache(xj, xi) * dy(xj / window_size, xi / window_size);

			}
		}

	}


protected:

	mic::types::MatrixXf cache;

	/*!
	 * Number of image (matrix) channels.
	 */
	size_t channels;

	/*!
	 * Size of the pooling window.
	 */
	size_t window_size;

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::inputs_size;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp1> friend class MultiLayerNeuralNetwork;


	/*!
	 * Private constructor, used only during the serialization.
	 */
	Pooling<eT>() : Layer<eT> () { }

};

} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_POOLING_HPP_ */
