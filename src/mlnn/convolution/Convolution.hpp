/*!
 * \file Convolution.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_CONVOLUTION_HPP_
#define SRC_MLNN_CONVOLUTION_HPP_

#include <mlnn/layer/Layer.hpp>

#include<types/MatrixTypes.hpp>
#include<types/MatrixArray.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

#define ADDRESS_3D_TO_1D(i, j, k, cols, channel_size) ((i) + (j) * (cols) + (k) * (channel_size))

/*!
 * \brief Class representing a convolution layer.
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Convolution : public mic::mlnn::Layer<eT> {
public:

	Convolution<eT>(size_t inputs, size_t channels, size_t filter_size, size_t filters, std::string name_ = "Convolution") :
		Layer<eT>(inputs * channels, filters * (sqrt(inputs) - filter_size + 1) * (sqrt(inputs) - filter_size + 1), 1, LayerTypes::Convolution, name_),
		input_channels(channels), output_channels(filters), kernel_size(filter_size),
		output_map_size((sqrt(inputs) - filter_size + 1) * (sqrt(inputs) - filter_size + 1)) {


		W = mic::types::Matrix<eT>(filters, filter_size * filter_size * input_channels);
		b = (Eigen::Vector<eT>)Eigen::Vector<eT>::Zero(filters);

		//W << 0.1, 0, 0, 0, 0, 0, 0, 0, 0;
		size_t fan_in = channels * filter_size * filter_size;
		size_t fan_out = filters * filter_size * filter_size;
		eT range = sqrt(6.0 / eT(fan_in + fan_out));

		W.rand(-range, range);

		mW = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(W.rows(), W.cols());
		mb = mic::types::Vector<eT>::Zero(b.rows());

	};

	virtual ~Convolution() {};

	void forward(bool test = false) {

		forwardGemm((*s['y']), (*s['x']));

	}

	/*!
	 * Outer loop over image locations, all images processed in parallel
	 * @param out
	 * @param in
	 */
	void forwardGemm(mic::types::Matrix<eT>& out, mic::types::Matrix<eT>& in) {

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

				mic::types::Matrix<eT> O = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, filters);
				mic::types::Matrix<eT> I = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, kernel_length);

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

	void backward() {

		(*g['x']).setZero();
		//dW
		backwardGemm((*g['y']), (*s['x']));
		//dx
		backwardFullGemm((*g['y']), (*g['x']));
	}

	void backwardGemm(mic::types::Matrix<eT>& out, mic::types::Matrix<eT>& in)  {

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

				mic::types::Matrix<eT> dW = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(W.rows(), W.cols());
				mic::types::Matrix<eT> db = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(b.rows(), b.cols());
				mic::types::Matrix<eT> O = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, filters);
				mic::types::Matrix<eT> I = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, kernel_length);

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

	void backwardFullGemm(mic::types::Matrix<eT>& out, mic::types::Matrix<eT>& in)  {

		size_t channel_length = in.rows() / input_channels;
		size_t kernel_size = sqrt(W.cols() / input_channels);
		size_t kernel_length_channel = kernel_size * kernel_size;
		size_t image_size = sqrt(in.rows() / input_channels);
		size_t batch_size = in.cols();
		size_t out_image_size = image_size - kernel_size + 1;
		size_t filters = W.rows();

		//pad matrices
		size_t padded_size = image_size + kernel_size - 1;
		mic::types::Matrix<eT> out_padded = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(padded_size * padded_size * filters, batch_size);

		#pragma omp parallel for shared(out_padded)
		for (size_t b = 0; b < batch_size; b++) {

			mic::types::Matrix<eT> out_resized = (Eigen::Matrix<eT>)out.col(b);
			mic::types::Matrix<eT> padded_temp = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(padded_size, padded_size * filters);
			out_resized.resize(out_image_size, out_image_size * filters);

			for (size_t f = 0; f < filters; f++) {

				mic::types::Matrix<eT> padded_temp2 = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(padded_size, padded_size);
				mic::types::Matrix<eT> out_temp2 = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(out_image_size, out_image_size);
				out_temp2 = out_resized.block(0, f * out_image_size, out_image_size, out_image_size);
				padded_temp2.block(kernel_size - 1, kernel_size - 1, out_image_size, out_image_size) = out_temp2;
				padded_temp.block(0, f * padded_size, padded_size, padded_size) = padded_temp2;

			}

			padded_temp.resize(padded_size * padded_size * filters, 1);
			out_padded.col(b) = padded_temp;

		}

		mic::types::Matrix<eT> W_permuted = mic::types::Matrix<eT>(kernel_size * kernel_size * filters, input_channels);
		mic::types::Matrix<eT> temp_W2 = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(1, kernel_size * kernel_size);

		for (size_t c = 0; c < input_channels; c++) {

			for (size_t f = 0; f < filters; f++) {

				mic::types::Matrix<eT> temp_W2 = (Eigen::Matrix<eT>)W.block(f, c * kernel_size * kernel_size, 1, kernel_size * kernel_size);
				temp_W2.reverseInPlace();
				W_permuted.block(f * kernel_size * kernel_size, c, kernel_size * kernel_size, 1) = temp_W2.transpose().eval();
			}

		}

		#pragma omp parallel for collapse(2)

		for (size_t x = 0; x < image_size; x++) {
			for (size_t y = 0; y < image_size; y++) {

				mic::types::Matrix<eT> O = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, kernel_size * kernel_size * filters);
				mic::types::Matrix<eT> I = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(batch_size, input_channels);

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

	void resetGrads()  {

		dW = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(W.rows(), W.cols());
		db = mic::types::Vector<eT>::Zero(b.rows());;

	}

	void applyGrads(double alpha)  {

		//adagrad
		mW += dW.cwiseProduct(dW);
		mb += db.cwiseProduct(db);

		W = (1 - decay) * W + alpha * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
		b += alpha * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));

	}

protected:
	mic::types::Matrix<eT> W;
	mic::types::Vector<eT> b;

	mic::types::Matrix<eT> dW;
	mic::types::Matrix<eT> db;

	mic::types::Matrix<eT> mW;
	mic::types::Matrix<eT> mb;

	size_t input_channels;

	size_t output_channels;

	size_t kernel_size;

	size_t output_map_size;

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using *this<eT>::b;

private:
	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Convolution<eT>() : Layer<eT> () { }

};

} /* convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVOLUTION_HPP_ */
