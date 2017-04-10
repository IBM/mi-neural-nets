/*!
 * \file Layer.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/Layer.hpp>

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
	s.add (
				{
					std::make_tuple ( "x", inputs_size, batch_size ), 	// inputs
					std::make_tuple ( "y", outputs_size, batch_size ) 	// outputs
				} );

	g.add (
				{
					std::make_tuple ( "x", inputs_size, batch_size ), 	// inputs
					std::make_tuple ( "y", outputs_size, batch_size ) 	// outputs
				} );

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



//this is mainly for debugging - TODO: proper serialization of layers and object NN
void Layer::save_to_files(std::string prefix) {

	save_matrix_to_file((*s['x']), prefix + "_x");
	save_matrix_to_file((*s['y']), prefix + "_y");
	save_matrix_to_file((*g['x']), prefix + "_dx");
	save_matrix_to_file((*g['y']), prefix + "_dy");

};


void Layer::save_matrix_to_file(Eigen::MatrixXf& m, std::string filename) {


	std::cout << "Saving a matrix to " << filename << "... " << std::endl;
	std::ofstream file(filename.c_str());

	if (file.is_open()) {

		file << m;
		file.close();

	} else {

		std::cout << "file save error: (" << filename << ")" << std::endl;

	}

}


} /* namespace mlnn */
} /* namespace mic */
