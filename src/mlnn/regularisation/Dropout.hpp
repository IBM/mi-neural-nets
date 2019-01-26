/*!
 * \file Dropout.hpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#ifndef SRC_MLNN_DROPOUT_HPP_
#define SRC_MLNN_DROPOUT_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace regularisation {


/*!
 * \brief Droput layer - a layer used for the regularization of neural network by randomly dropping neurons during training.
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Dropout : public Layer<eT> {
public:

	/*!
	 * @param ratio_ Keep ratio denoting the probability of activations to be passed.
	 */
	Dropout<eT>(size_t inputs_, float ratio_, std::string name_ = "Dropout") :
		Layer<eT>(inputs_, 1, 1,
				inputs_, 1, 1,
				LayerTypes::Dropout, name_),
				keep_ratio(ratio_)
	{
		// Create matrices with temporary variables: random and dropout of size [inputxinput], so we can simply calculate: y=mask*x.
		m.add ("random", inputs_, inputs_);
		m.add ("dropout_mask", inputs_, inputs_);
	}

	virtual ~Dropout() {};

	/*!
	 * Changes the size of the batch - calls base Layer class resize and additionally resizes the cache size.
	 * @param New size of the batch.
	 */
	virtual void resizeBatch(size_t batch_size_) {
		// Call base Layer resize.
		Layer<eT>::resizeBatch(batch_size_);

		// Reshape pooling mask and map.
		m["dropout_mask"]->resize(Layer<eT>::inputSize(), batch_size_);
	}


	void forward(bool test = false) {
		if (test) {
			// In test run copy data as it is.
			(*s['y']) = (*s['x']);

		} else {
			// Get pointers to input and output batches.
			mic::types::MatrixPtr<eT> batch_x = s['x'];
			mic::types::MatrixPtr<eT> batch_y = s['y'];

			// Generate random matrix.
			mic::types::MatrixPtr<eT> rand = m["random"];
			rand->rand(0.0f, 1.0f);

			// Generate the dropout mask.
			mic::types::MatrixPtr<eT> mask = m["dropout_mask"];

			#pragma omp parallel for
			for(size_t i=0; i< (size_t)mask->size(); i++)
				(*mask)[i] = ((*rand)[i] < keep_ratio);

			// Apply the dropout_mask - discard the elements where mask is 0.
			(*batch_y) =  (*mask) * (*batch_x);

			// Normalize, so that we don't have to do anything at test time.
			(*batch_y) /= keep_ratio;

		}
	}

	void backward() {
		// Get pointers to input and output batches.
		mic::types::MatrixPtr<eT> batch_dx = g['x'];
		mic::types::MatrixPtr<eT> batch_dy = g['y'];
		mic::types::MatrixPtr<eT> mask = m["dropout_mask"];

		// Always use dropout mask as backward pass is used only during learning.
		(*batch_dy) =  (*mask).transpose() * (*batch_dx);
	}

	/*!
	 * Performs the update according to the calculated gradients and injected optimization method. Empty as this is a "const" layer.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	virtual void update(eT alpha_, eT decay_  = 0.0f) { };

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
    using Layer<eT>::batch_size;


	/*!
	 * Ratio denoting the probability of activations to be passed.
	 */
	eT keep_ratio;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Dropout<eT>() : Layer<eT> () { }


};


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_DROPOUT_HPP_ */
