/*!
 * \file Dropout.cpp
 * \brief 
 * \author tkornut
 * \date Apr 12, 2016
 */

#include <mlnn/regularisation/Dropout.hpp>

namespace mic {
namespace mlnn {
namespace regularisation {

Dropout::Dropout(size_t inputs, size_t outputs, size_t batch_size, float _ratio, std::string name_) :
	Layer(inputs, outputs, batch_size, LayerTypes::Dropout, name_),  keep_ratio(_ratio) {

};


void Dropout::forward(bool test) {

	if (test) { // skip at test time

		(*s['y']) = (*s['x']);

	} else {

		mic::types::MatrixXf rands = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(s['y']->rows(), s['y']->cols());
		//rand(rands, 0.0f, 1.0f);
		rands.rand(0.0f, 1.0f);

		//dropout mask - 1s - preserved elements
		dropout_mask = (rands.array() < keep_ratio).cast <float> ();

		// y = y .* dropout_mask, discard elements where mask is 0
		s['y']->array() = s['x']->array() * dropout_mask.array();

		// normalize, so that we don't have to do anything at test time
		(*s['y']) /= keep_ratio;

	}
}

void Dropout::backward() {

	g['x']->array() = g['y']->array() * dropout_mask.array();
}


} /* regularisation */
} /* namespace mlnn */
} /* namespace mic */
