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

	/*!
	 * Creates the a convolutional layer.
	 * @param input_height_ Height of the input / rows (e.g. 28 for MNIST).
	 * @param input_width_ Width of the input / cols (e.g. 28 for MNIST).
	 * @param input_channels_ Number of channels of the input (e.g. 3 for RGB images).
	 * @param filter_size_ Size of filters (assuming square filters).
	 * @param stride_ Stride (assuming equal vertical and horizontal strides).
	 * @param number_of_filters_ Number of filters = Length of the output vector.
	 * @param name_ Name of the layer.
	 */
	Convolution<eT>(size_t input_height_, size_t input_width_, size_t input_channels_, size_t filter_size_, size_t stride_, size_t number_of_filters_, std::string name_ = "Convolution") :
		Layer<eT>(input_width_ * input_height_ * input_channels_, number_of_filters_ * (input_width_ - filter_size_ + 1) * (input_height_ - filter_size_ + 1), 1, LayerTypes::Convolution, name_),
	    input_width(input_width_),
	    input_channels(input_channels_),
		input_height(input_height_),
		filter_size(filter_size_),
		stride(stride_),
		number_of_filters(number_of_filters_)
	{
		// Calculate total input-output number of parameters - for initialization.
		size_t fan_in = filter_size * filter_size * input_channels;
		size_t fan_out = filter_size * filter_size * number_of_filters;

		// Create filters.
		for (size_t i=0; i< number_of_filters; i++) {
			// Create the weights matrix.
			p.add ("W"+std::to_string(i), filter_size, filter_size);
			// Create the bias for a given filter.
			p.add ("b"+std::to_string(i), 1, 1);

			// Initialize weights of the W matrix.
			eT range = sqrt(6.0 / eT(fan_in + fan_out));
			p["W"+std::to_string(i)]->rand(-range, range);
			p["b"+std::to_string(i)]->setZero();

		}//: for


		// Calculate number of receptive fields within a "single input channel".
		number_of_receptive_fields_horizontal = (input_width - filter_size)/stride + 1;
		number_of_receptive_fields_vertical = (input_height - filter_size)/stride + 1;
//		std::cout<<"number_of_receptive_fields_horizontal = " << number_of_receptive_fields_horizontal <<std::endl;
//		std::cout<<"number_of_receptive_fields_vertical = " << number_of_receptive_fields_vertical <<std::endl;
		// Filters must "exactly" fit!
		assert((number_of_receptive_fields_horizontal - 1) * stride + filter_size == input_width);
		assert((number_of_receptive_fields_vertical - 1) * stride + filter_size == input_height);

		// Allocate memory for "input receptive fields".
		// Create filters.
		for (size_t ry=0; ry< number_of_receptive_fields_vertical; ry++) {
			for (size_t rx=0; rx< number_of_receptive_fields_horizontal; rx++) {
				// Create receptive field matrix.
				m.add ("r"+std::to_string(ry)+std::to_string(rx), filter_size, filter_size);
			}//: for
		}//: for

	};

	virtual ~Convolution() {};

	void forward(bool test = false) {
		// Get input matrix.
		mic::types::Matrix<eT> x = (*s['x']);

		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> y = s['y'];

		// Forward pass.
		//(*y) = W * x + b.replicate(1, x.cols());
		// Cut x into "slices" - receptive fields.

		// Iterate through input channels.
		for (size_t c=0; c< input_channels; c++) {
			// Iterate through receptive fields.
			/*for (size_t r=0; r< number_of_receptive_fields; r++) {
				// Get receptive field
				mic::types::MatrixPtr<eT> field = m.["r"+std::to_string(r)];
			}//: for*/

		}//: for

		// Iterate through filters.
		for (size_t f=0; f< number_of_filters; f++) {
			// Get filter weight and bias.
			mic::types::Matrix<eT> W = (*p["W"+std::to_string(f)]);
			mic::types::Matrix<eT> b = (*p["b"+std::to_string(f)]);

		}//: for

	}

	void backward() {

/*		(*g['x']).setZero();
		//dW
		backwardGemm((*g['y']), (*s['x']));
		//dx
		backwardFullGemm((*g['y']), (*g['x']));*/
	}


	/*!
	 * Outer loop over image locations, all images processed in parallel
	 * @param out
	 * @param in
	 */
/*	void forwardGemm(mic::types::Matrix<eT>& out, mic::types::Matrix<eT>& in) {

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
	}*/

	void resetGrads()  {

/*		dW = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(W.rows(), W.cols());
		db = mic::types::Vector<eT>::Zero(b.rows());;*/

	}

	void applyGrads(double alpha)  {

		//adagrad
/*		mW += dW.cwiseProduct(dW);
		mb += db.cwiseProduct(db);

		W = (1 - decay) * W + alpha * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
		b += alpha * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));*/

	}

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
//    using Layer<eT>::input_size;
//    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

    /// Width of the input (e.g. 28 for MNIST).
    size_t input_width;

    /// Number of channels of the input (e.g. 3 for RGB images).
    size_t input_channels;

    /// Height of the input (e.g. 28 for MNIST).
    size_t input_height;

    /// Size of filters (assuming square filters). Filter_size^2 = length of the output vector.
	size_t filter_size;

	/// Number of filters = number of output channels.
	size_t number_of_filters;

	/// Number of receptive fields in a single channel - horizontal direction.
	size_t number_of_receptive_fields_horizontal;

	/// Number of receptive fields in a single channel - vertical direction.
	size_t number_of_receptive_fields_vertical;

	 /// Stride (assuming equal vertical and horizontal strides).
	 size_t stride;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;


	/*!
	 * Private constructor, used only during the serialization.
	 */
	Convolution<eT>() : Layer<eT> () { }

};

} /* convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVOLUTION_HPP_ */
