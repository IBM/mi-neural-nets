/*!
 * \file Identity.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/fully_connected/Identity.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {


Identity::Identity(size_t inputs, size_t outputs, size_t batch_size, std::string name_) :
	Layer(inputs, outputs, batch_size, LayerTypes::Identity, name_) {

}

void Identity::forward(bool test) {

	// y = x;
	(*s['y']) = (*s['x']);

}

void Identity::backward() {

	// dx = dy;
	(*g['x']) = (*g['y']);

}

} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */
