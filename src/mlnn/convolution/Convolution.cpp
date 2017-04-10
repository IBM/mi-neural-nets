/*!
 * \file Convolution.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/convolution/Convolution.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

#define ADDRESS_3D_TO_1D(i, j, k, cols, channel_size) ((i) + (j) * (cols) + (k) * (channel_size))

Convolution::Convolution(size_t inputs, size_t channels, size_t filter_size, size_t filters, size_t batch_size, std::string name_) :
	Layer(inputs * channels, filters * (sqrt(inputs) - filter_size + 1) * (sqrt(inputs) - filter_size + 1), batch_size, LayerTypes::Convolution, name_),
	input_channels(channels), output_channels(filters), kernel_size(filter_size),
	output_map_size((sqrt(inputs) - filter_size + 1) * (sqrt(inputs) - filter_size + 1)) {


	W = mic::types::MatrixXf(filters, filter_size * filter_size * input_channels);
	b = (Eigen::VectorXf)Eigen::VectorXf::Zero(filters);

	//W << 0.1, 0, 0, 0, 0, 0, 0, 0, 0;
	size_t fan_in = channels * filter_size * filter_size;
	size_t fan_out = filters * filter_size * filter_size;
	double range = sqrt(6.0 / double(fan_in + fan_out));

	W.rand(-range, range);

	mW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
	mb = mic::types::VectorXf::Zero(b.rows());

};

void Convolution::forward(bool apply_dropout) {

	forwardGemm((*s['y']), (*s['x']));

}

void Convolution::forwardGemm(mic::types::MatrixXf& out, mic::types::MatrixXf& in) {

	//W is size [kernel_length x filters]
	//I is size [batch_size x kernel_length]
	//O is [batch_size x filters] = [batch_size x kernel_length] * [kernel_length x filters]

	//total number of operations proportional to
	//out_image_size * out_image_size * batch_size * kernel_length * filters

	size_t kernel_size = sqrt(W.cols() / input_channels);
	size_t channel_length = in.rows() / input_channels;
	size_t kernel_length = kernel_size * kernel_size * input_channels;
	size_t kernel_length_channel = kernel_size * kernel_size;
	size_t image_size = sqrt(in.rows() / input_channels);
	size_t batch_size = in.cols();
	size_t out_image_size = image_size - kernel_size + 1;
	size_t out_image_channel_length = out_image_size * out_image_size;
	size_t filters = W.rows();

	#pragma omp parallel for collapse(2)
	for (size_t x = 0; x < out_image_size; x++) {
		for (size_t y = 0; y < out_image_size; y++) {

			mic::types::MatrixXf O = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, filters);
			mic::types::MatrixXf I = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, kernel_length);

			//inputs(:, :) = images(:, x, y, :);
			for (size_t k0 = 0; k0 < kernel_size; k0++) {
				for (size_t k1 = 0; k1 < kernel_size; k1++) {

					for (size_t channel = 0; channel < input_channels; channel++) {

						size_t i = x + k0;
						size_t j = y + k1;
						size_t k = channel * kernel_length_channel + k0 * kernel_size + k1;
						I.col(k) = in.row(ADDRESS_3D_TO_1D(i, j, channel, image_size, channel_length));

					}

				}
			}


			O = I * W.transpose();
			O = O + b.transpose().replicate(batch_size, 1);

			for (size_t k = 0; k < filters; k++) {

				out.row(ADDRESS_3D_TO_1D(x, y, k, out_image_size, out_image_channel_length)) = O.col(k);

			}

		} 	// y loop
	}	// x loop

}

void Convolution::backward() {

	(*g['x']).setZero();
	//dW
	backwardGemm((*g['y']), (*s['x']));
	//dx
	backwardFullGemm((*g['y']), (*g['x']));
}


void Convolution::backwardGemm(mic::types::MatrixXf& out, mic::types::MatrixXf& in) {

	//W is size [filters x kernel_length]
	//I is size [batch_size x kernel_length]
	//O is [batch_size x filters] = [batch_size x kernel_length] * [kernel_length x filters]

	//total number of operations proportional to
	//out_image_size * out_image_size * batch_size * kernel_length * filters
	size_t channel_length = in.rows() / input_channels;
	size_t kernel_size = sqrt(W.cols() / input_channels);
	size_t kernel_length = kernel_size * kernel_size * input_channels;
	size_t kernel_length_channel = kernel_size * kernel_size;
	size_t image_size = sqrt(in.rows() / input_channels);
	size_t batch_size = in.cols();
	size_t out_image_size = image_size - kernel_size + 1;
	size_t out_image_channel_length = out_image_size * out_image_size;
	size_t filters = W.rows();

	W.setZero();
	b.setZero();
	#pragma omp parallel for collapse(2)
	for (size_t x = 0; x < out_image_size; x++) {

		for (size_t y = 0; y < out_image_size; y++) {

			mic::types::MatrixXf dW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
			mic::types::MatrixXf db = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(b.rows(), b.cols());
			mic::types::MatrixXf O = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, filters);
			mic::types::MatrixXf I = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, kernel_length);

			//inputs(:, : ) = images(:, x, y, : );
			for (size_t k0 = 0; k0 < kernel_size; k0++) {

				for (size_t k1 = 0; k1 < kernel_size; k1++) {

					for (size_t channel = 0; channel < input_channels; channel++) {

						size_t i = x + k0;
						size_t j = y + k1;
						size_t k = channel * kernel_length_channel + k0 * kernel_size + k1;

						I.col(k) = in.row(ADDRESS_3D_TO_1D(i, j, channel, image_size, channel_length));

					}

				}
			}

			for (size_t k = 0; k < filters; k++) {

				O.col(k) = out.row(ADDRESS_3D_TO_1D(x, y, k, out_image_size, out_image_channel_length));
				db(k) = O.col(k).sum() / batch_size;
			}

			dW = (O.transpose() * I);

			//reduction
			#pragma omp critical
			{
				W = W + dW / batch_size;
				b = b + db;
			}

		} 	// y loop
	}	// x loop

}


void Convolution::backwardFullGemm(mic::types::MatrixXf& out, mic::types::MatrixXf& in) {

	size_t channel_length = in.rows() / input_channels;
	size_t kernel_size = sqrt(W.cols() / input_channels);
	size_t kernel_length_channel = kernel_size * kernel_size;
	size_t image_size = sqrt(in.rows() / input_channels);
	size_t batch_size = in.cols();
	size_t out_image_size = image_size - kernel_size + 1;
	size_t filters = W.rows();

	//pad matrices
	size_t padded_size = image_size + kernel_size - 1;
	mic::types::MatrixXf out_padded = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size * padded_size * filters, batch_size);

	#pragma omp parallel for shared(out_padded)
	for (size_t b = 0; b < batch_size; b++) {

		mic::types::MatrixXf out_resized = (Eigen::MatrixXf)out.col(b);
		mic::types::MatrixXf padded_temp = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size * filters);
		out_resized.resize(out_image_size, out_image_size * filters);

		for (size_t f = 0; f < filters; f++) {

			mic::types::MatrixXf padded_temp2 = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size);
			mic::types::MatrixXf out_temp2 = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(out_image_size, out_image_size);
			out_temp2 = out_resized.block(0, f * out_image_size, out_image_size, out_image_size);
			padded_temp2.block(kernel_size - 1, kernel_size - 1, out_image_size, out_image_size) = out_temp2;
			padded_temp.block(0, f * padded_size, padded_size, padded_size) = padded_temp2;

		}

		padded_temp.resize(padded_size * padded_size * filters, 1);
		out_padded.col(b) = padded_temp;

	}

	mic::types::MatrixXf W_permuted = mic::types::MatrixXf(kernel_size * kernel_size * filters, input_channels);
	mic::types::MatrixXf temp_W2 = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(1, kernel_size * kernel_size);

	for (size_t c = 0; c < input_channels; c++) {

		for (size_t f = 0; f < filters; f++) {

			mic::types::MatrixXf temp_W2 = (Eigen::MatrixXf)W.block(f, c * kernel_size * kernel_size, 1, kernel_size * kernel_size);
			temp_W2.reverseInPlace();
			W_permuted.block(f * kernel_size * kernel_size, c, kernel_size * kernel_size, 1) = temp_W2.transpose().eval();
		}

	}

	#pragma omp parallel for collapse(2)

	for (size_t x = 0; x < image_size; x++) {
		for (size_t y = 0; y < image_size; y++) {

			mic::types::MatrixXf O = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, kernel_size * kernel_size * filters);
			mic::types::MatrixXf I = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(batch_size, input_channels);

			//inputs(:, :) = images(:, x, y, :);
			for (size_t k0 = 0; k0 < kernel_size; k0++) {

				for (size_t k1 = 0; k1 < kernel_size; k1++) {

					for (size_t channel = 0; channel < filters; channel++) {

						size_t i = x + k0;
						size_t j = y + k1;
						size_t k = channel * kernel_length_channel + k0 * kernel_size + k1;
						O.col(k) = out_padded.row(ADDRESS_3D_TO_1D(i, j, channel, padded_size, padded_size * padded_size));

					}

				}
			}

			I = O * W_permuted;

			for (size_t k = 0; k < input_channels; k++) {

				in.row(ADDRESS_3D_TO_1D(x, y, k, image_size, channel_length)) = I.col(k);

			}

		}
	}
}


void Convolution::resetGrads() {

	dW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
	db = mic::types::VectorXf::Zero(b.rows());;

}

void Convolution::applyGrads(double alpha, double decay) {

	//adagrad
	mW += dW.cwiseProduct(dW);
	mb += db.cwiseProduct(db);

	W = (1 - decay) * W + alpha * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
	b += alpha * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));

}


void Convolution::save_to_files(std::string prefix) {

	Layer::save_to_files(prefix);
	save_matrix_to_file(W, prefix + "_W.txt");
	// save_matrix_to_file(b, prefix + "_b.txt");

}

} /* convolution */
} /* namespace mlnn */
} /* namespace mic */
