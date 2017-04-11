/*!
 * \file Layer.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include "Layer.hpp"

#include <iostream>
#include <fstream>

namespace mic {
namespace mlnn {

Layer::Layer(size_t inputs_size_, size_t outputs_size_, size_t batch_size_, LayerTypes layer_type_, std::string name_) :
		inputs_size(inputs_size_),
		outputs_size(outputs_size_),
		batch_size(batch_size_),
		layer_type(layer_type_),
		layer_name(name_),
		s("state"),
		g("gradients"),
		p("parameters"),
		m("memory")

{
	// State.
	s.add ( "x", inputs_size, batch_size ); 	// inputs
	s.add ( "y", outputs_size, batch_size); 	// outputs

	// Gradients.
	g.add ( "x", inputs_size, batch_size ); 	// inputs
	g.add ( "y", outputs_size, batch_size); 	// outputs

};


size_t Layer::inputsSize() {
	return inputs_size;
}

size_t Layer::outputsSize() {
	return outputs_size;
}

size_t Layer::batchSize() {
	return batch_size;
}

void Layer::resizeBatch(size_t batch_size_) {
	// Change the "value". (depricated)
	batch_size = batch_size_;
	// Reshape the inputs...
	s['x']->resize(s['x']->rows(), batch_size_);
	g['x']->resize(g['x']->rows(), batch_size_);
	// ... and outputs.
	s['y']->resize(s['y']->rows(), batch_size_);
	g['y']->resize(g['y']->rows(), batch_size_);
}


} /* namespace mlnn */
} /* namespace mic */
