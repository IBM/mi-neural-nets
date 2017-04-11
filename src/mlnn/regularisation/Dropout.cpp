/*!
 * \file Dropout.cpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#include "regularisation/Dropout.hpp"

namespace mic {
namespace mlnn {
namespace regularisation {

Dropout::Dropout(size_t inputs, float ratio_, std::string name_) :
	Layer(inputs, inputs, 1, LayerTypes::Dropout, name_),  keep_ratio(ratio_) {

};


void Dropout::forward(bool test) {

	if (test) {
		// Copy data as it is.
		(*s['y']) = (*s['x']);

	} else {
		// Generate random matrix.
		mic::types::MatrixXf rands = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(s['y']->rows(), s['y']->cols());
		rands.rand(0.0f, 1.0f);

		// Generate the dropout mask.
		dropout_mask = (rands.array() < keep_ratio).cast <float> ();

		// Apply the dropout_mask - discard the elements where mask is 0.
		s['y']->array() = s['x']->array() * dropout_mask.array();

		// Normalize, so that we don't have to do anything at test time.
		(*s['y']) /= keep_ratio;

	}
}

void Dropout::backward() {

	g['x']->array() = g['y']->array() * dropout_mask.array();
}


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */
