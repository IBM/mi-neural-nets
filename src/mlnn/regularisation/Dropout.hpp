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
		Layer<eT>(inputs_, inputs_, 1, LayerTypes::Dropout, name_),  keep_ratio(ratio_)
	{
		// Create matrices with temporary variables: random and dropout.
		m.add ("random", inputs_, inputs_);
		m.add ("dropout_mask", inputs_, inputs_);
	}

	virtual ~Dropout() {};

	void forward(bool test = false) {
		if (test) {
			// In test run copy data as it is.
			(*s['y']) = (*s['x']);

		} else {
			// Generate random matrix.
			m["random"]->rand(0.0f, 1.0f);

			// Generate the dropout mask.
			//m["dropout_mask"]->array() = (m["random"]->array() < keep_ratio).cast<eT> ();
			for(size_t i=0; i< (size_t)m["dropout_mask"]->size(); i++)
				(*m["dropout_mask"])[i] = (*m["random"])[i] < keep_ratio;

			// Apply the dropout_mask - discard the elements where mask is 0.
			(*s['y']) = (*s['x']) * (*m["dropout_mask"]);

			// Normalize, so that we don't have to do anything at test time.
			(*s['y']) /= keep_ratio;

		}
	}

	void backward() {
		// Always use dropout mask as backward pass is used only during learning.
		(*g['x']) = (*g['y']) * (*m["dropout_mask"]);
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
    using Layer<eT>::input_size;
    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;


	/*!
	 * Ratio denoting the probability of activations to be passed.
	 */
	eT keep_ratio;

	/*!
	 * Dropout mask - computed in forward() method in every pass, do not have to be serialized, thus not included in the parameters vector (p).
	 */
	mic::types::Matrix<eT> dropout_mask;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp1, typename tmp2> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Dropout<eT>() : Layer<eT> () { }


};


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_DROPOUT_HPP_ */
