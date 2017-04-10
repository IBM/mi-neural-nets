/*!
 * \file Pooling.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_POOLING_HPP_
#define SRC_MLNN_POOLING_HPP_

#include <mlnn/Layer.hpp>

namespace mic {
namespace mlnn {
namespace convolution {


/*!
 * \author krocki
 */
class Pooling : public mic::mlnn::Layer {
public:

	Pooling(size_t inputs_, size_t window_size_, size_t channels_, size_t batch_size_, std::string name_ = "Pooling");

	virtual ~Pooling() {};

	void forward(bool test_ = false);

	/*!
	 * Processes forwards data for a single channel.
	 * @param x Channel
	 * @param cache
	 * @param window_size
	 * @return
	 */
	mic::types::MatrixXf forwardChannel(mic::types::MatrixXf& x, mic::types::MatrixXf& cache);

	/*!
	 *
	 * @param out
	 * @param cache
	 * @param image
	 */
	void poolDisjoint2D(mic::types::MatrixXf& out, mic::types::MatrixXf& cache, mic::types::MatrixXf& image);

	void backward();

	/*!
	 * Processes backwards data for a single channel.
	 * @param dy
	 * @param cache_
	 * @return
	 */
	mic::types::MatrixXf backwardChannel(mic::types::MatrixXf& dy, mic::types::MatrixXf& cache_);

	/*!
	 *
	 * @param dx
	 * @param cache
	 * @param dy
	 */
	void unpoolDisjoint2D(mic::types::MatrixXf& dx, mic::types::MatrixXf& cache, mic::types::MatrixXf& dy);

	//this is mainly for debugging - TODO: proper serialization of layers
	void save_to_files(std::string prefix);

protected:

	mic::types::MatrixXf cache;

	/*!
	 * Number of image (matrix) channels.
	 */
	size_t channels;

	/*!
	 * Size of the window.
	 */
	size_t window_size;

private:

	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Pooling() : Layer () { }

};

} /* namespace convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_POOLING_HPP_ */
