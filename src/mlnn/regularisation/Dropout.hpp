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
	Dropout<eT>(size_t inputs, float ratio_, std::string name = "Dropout") :
		Layer<eT>(inputs, inputs, 1, LayerTypes::Dropout, name_),  keep_ratio(ratio_)
	{

	}

	virtual ~Dropout() {};

	void forward(bool test = false) {
		if (test) {
			// Copy data as it is.
			(*s['y']) = (*s['x']);

		} else {
			// Generate random matrix.
			mic::types::Matrix<eT> rands = (Eigen::Matrix<eT>)Eigen::Matrix<eT>::Zero(s['y']->rows(), s['y']->cols());
			rands.rand(0.0f, 1.0f);

			// Generate the dropout mask.
			dropout_mask = (rands.array() < keep_ratio).cast <eT> ();

			// Apply the dropout_mask - discard the elements where mask is 0.
			s['y']->array() = s['x']->array() * dropout_mask.array();

			// Normalize, so that we don't have to do anything at test time.
			(*s['y']) /= keep_ratio;

		}
	}

	void backward() {
		// TODO: fix bug!!
		g['x']->array() = g['y']->array() * dropout_mask.array();
	}

protected:
	// Unhiding the template inherited fields via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;


	/*!
	 * Ratio denoting the probability of activations to be passed.
	 */
	float keep_ratio;

	/*!
	 * Dropout mask - computed in forward() method in every pass, do not have to be serialized, thus not included in the parameters vector (p).
	 */
	mic::types::Matrix<eT> dropout_mask;

private:
	// Adds the nn class the access to protected fields of class layer.
	friend class mic::mlnn::MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Dropout<eT>() : Layer<eT> () { }


};


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_DROPOUT_HPP_ */
