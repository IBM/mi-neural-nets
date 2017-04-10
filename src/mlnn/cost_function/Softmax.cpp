/*!
 * \file Softmax.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <mlnn/cost_function/Softmax.hpp>

namespace mic {
namespace mlnn {
namespace cost_function {


Softmax::Softmax(size_t inputs_, size_t outputs_, size_t batch_size_, std::string name_) :
	Layer(inputs_, outputs_, batch_size_, LayerTypes::Softmax, name_) {

}

void Softmax::forward(bool test_) {

//	y = softmax(x);

	mic::types::MatrixXf e = (Eigen::MatrixXf)((*s['x']).unaryExpr(std::ptr_fun(::expf)));

	mic::types::VectorXf sum = e.colwise().sum();

	for (int i = 0; i < e.rows(); i++) {
		for (int j = 0; j < e.cols(); j++) {

			(*s['y'])(i, j) = e(i, j) / sum(j);
		}
	}

}

void Softmax::backward() {

	// dx = dy - y;
	(*g['x']) = (*g['y']) - (*s['y']);

}


} /* namespace cost_function */
} /* namespace mlnn */
} /* namespace mic */
