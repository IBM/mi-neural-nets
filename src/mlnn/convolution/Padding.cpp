/*!
 * \file Padding.cpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#include <mlnn/convolution/Padding.hpp>

namespace mic {
namespace mlnn {
namespace convolution {


Padding::Padding(size_t inputs_, size_t channels_, size_t batch_size_, size_t padding_, std::string name_) :
	Layer(inputs_, channels_ * (sqrt(inputs_ / channels_) + padding_ * 2) * (sqrt(inputs_ / channels_) + padding_ * 2), batch_size_, LayerTypes::Padding, name_),
	channels(channels_),
	padding(padding_)
{

};


void Padding::forward(bool test) {

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



void Padding::backward() {

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

} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */
