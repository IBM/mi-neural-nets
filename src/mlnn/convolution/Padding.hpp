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
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Padding : public mic::mlnn::Layer<eT> {
public:

	Padding<eT>(size_t inputs, size_t _channels, size_t _padding, std::string name_ = "Padding") :
		Layer<eT>(inputs_, channels_ * (sqrt(inputs_ / channels_) + padding_ * 2) * (sqrt(inputs_ / channels_) + padding_ * 2), 1, LayerTypes::Padding, name_),
		channels(channels_),
		padding(padding_)
	{

	};

	virtual ~Padding() {};


	void forward(bool test = false) {

		//pad((*s['x']), (*s['y']), channels, padding);

		size_t batch_size = (*s['x']).cols();
		size_t image_size = sqrt((*s['x']).rows() / channels);
		size_t padded_size = sqrt((*s['y']).rows() / channels);

		(*s['y']).setZero();

		#pragma omp parallel for
		for (size_t b = 0; b < batch_size; b++) {

			mic::types::MatrixXf im_channel = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(image_size, image_size);
			mic::types::MatrixXf im_padded = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size * channels);
			mic::types::MatrixXf image = (Eigen::MatrixXf)Eigen::MatrixXf::Zero((*s['x']).rows(), 1);

			image = (*s['x']).col(b);
			image.resize(image_size, image_size * channels);

			mic::types::MatrixXf padded_channel = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size);

			for (size_t f = 0; f < channels; f++) {

				im_channel = image.block(0, f * image_size, image_size, image_size);
				padded_channel.setZero();
				padded_channel.block(padding, padding, image_size, image_size) = im_channel;
				im_padded.block(0, f * padded_size, padded_size, padded_size) = padded_channel;

			}

			im_padded.resize(padded_size * padded_size * channels, 1);
			(*s['y']).col(b) = im_padded;

		}

	}

	void backward() {

		//unpad((*g['x']), (*g['y']), channels, padding);

		size_t batch_size = (*g['x']).cols();
		size_t image_size = sqrt((*g['x']).rows() / channels);
		size_t padded_size = sqrt((*g['y']).rows() / channels);

		(*g['x']).setZero();

		#pragma omp parallel for
		for (size_t b = 0; b < batch_size; b++) {

			mic::types::MatrixXf im_channel_padded = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size);
			mic::types::MatrixXf im = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(image_size, image_size * channels);
			mic::types::MatrixXf im_channel = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(image_size, image_size);
			mic::types::MatrixXf image_padded = (Eigen::MatrixXf)Eigen::MatrixXf::Zero((*g['y']).rows(), 1);

			image_padded = (*g['y']).col(b);
			image_padded.resize(padded_size, padded_size * channels);

			mic::types::MatrixXf padded_channel = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(padded_size, padded_size);

			for (size_t f = 0; f < channels; f++) {

				im_channel_padded = image_padded.block(0, f * padded_size, padded_size, padded_size);
				im_channel = im_channel_padded.block(padding, padding, image_size, image_size);
				im.block(0, f * image_size, image_size, image_size) = im_channel;

			}

			im.resize(image_size * image_size * channels, 1);
			(*g['x']).col(b) = im;

		}
	}

protected:

	size_t channels;

	size_t padding;

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Padding<eT>() : Layer<eT> () { }



};


} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_PADDING_HPP_ */
