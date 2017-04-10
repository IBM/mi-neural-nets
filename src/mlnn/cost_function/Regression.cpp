/*!
 * \file Regression.cpp
 * \brief 
 * \author tkornut
 * \date Apr 22, 2016
 */

#include <mlnn/cost_function/Regression.hpp>

namespace mic {
namespace mlnn {
namespace cost_function {


Regression::Regression(size_t inputs_, size_t outputs_, size_t batch_size_, std::string name_) :
	Layer(inputs_, outputs_, batch_size_, LayerTypes::Regression, name_) {

}

void Regression::forward(bool test_) {

	// Pass inputs to outputs.
	(*s['y']) = (*s['x']);

}

void Regression::backward() {

	// dx = 2*(dy - y);
	(*g['x']) = 2 *( (*g['y']) - (*s['y']));

}


} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */
