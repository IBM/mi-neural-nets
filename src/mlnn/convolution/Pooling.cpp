/*!
 * \file Pooling.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/convolution/Pooling.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

Pooling::Pooling(size_t inputs_, size_t window_size_, size_t channels_, size_t batch_size_, std::string name_) :
	Layer(inputs_, channels_ * (sqrt(inputs_ / channels_) / window_size_) * (sqrt(inputs_ / channels_) / window_size_), batch_size_, LayerTypes::Pooling, name_),
	channels(channels_), window_size(window_size_) {

	// cache = mic::types::MatrixXf::Zero(x.rows(), x.cols());
	cache = mic::types::MatrixXf::Zero(inputs_size, batch_size);

};

void Pooling::forward(bool test_) {

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

mic::types::MatrixXf Pooling::forwardChannel(mic::types::MatrixXf& x_channel_, mic::types::MatrixXf& cache_) {

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

void Pooling::poolDisjoint2D(mic::types::MatrixXf& out, mic::types::MatrixXf& cache, mic::types::MatrixXf& image) {

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


void Pooling::backward() {

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

mic::types::MatrixXf Pooling::backwardChannel(mic::types::MatrixXf& dy, mic::types::MatrixXf& cache_) {

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

void Pooling::unpoolDisjoint2D(mic::types::MatrixXf& dx, mic::types::MatrixXf& cache, mic::types::MatrixXf& dy) {

	for (size_t xi = 0; xi < (size_t)dx.rows(); xi++) {

		for (size_t xj = 0; xj < (size_t)dx.cols(); xj++) {

			dx(xj, xi) = cache(xj, xi) * dy(xj / window_size, xi / window_size);

		}
	}

}


void Pooling::save_to_files(std::string prefix) {

	Layer::save_to_files(prefix);
	save_matrix_to_file(cache, prefix + "_cache.txt");

}

} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */
