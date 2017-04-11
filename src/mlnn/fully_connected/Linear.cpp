/*!
 * \file Linear.cpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#include <fully_connected/Linear.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {


Linear::Linear(size_t inputs_, size_t outputs_, std::string name_) :
	Layer(inputs_, outputs_, 1, LayerTypes::Linear, name_) {

	// Create the weights matrix.
	p.add ("W", outputs_, inputs_);

	// Create the bias vector.
	p.add ("b", outputs_, 1);

	// Initialize weights of the W matrix.
	double range = sqrt(6.0 / double(inputs_ + outputs_));

	p['W']->rand(-range, range);
	p['b']->rand(-range, range);// setZero();

	//mW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
	m.add ("W", outputs_, inputs_);


	//mb = mic::types::VectorXf::Zero(b.rows());
	m.add ("b", outputs_, 1);

	// Add W and b gradients.
	g.add ("W", outputs_, inputs_);
	g.add ("b", outputs_, 1 );

};

void Linear::forward(bool test_) {
	// y = W * x + b.replicate(1, x.cols());
	(*s['y']) = (*p['W']) * (*s['x']) + (*p['b']).replicate(1, (*s['x']).cols());

}

void Linear::backward() {

	/*dW = dy * x.transpose();
	db = dy.rowwise().sum();
	dx = W.transpose() * dy;*/

	(*g['W']) = (*g['y']) * ((*s['x']).transpose());
	(*g['b']) = (*g['y']).rowwise().sum();
	(*g['x']) = (*p['W']).transpose() * (*g['y']);

}

void Linear::resetGrads() {

	(*g['W']).setZero(); // = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
	(*g['b']).setZero(); // = mic::types::VectorXf::Zero(b.rows());
}

void Linear::applyGrads(double alpha_, double decay_) {

	//adagrad
	//mW += dW.cwiseProduct(dW);
	(*m['W']) += (*g['W']).cwiseProduct((*g['W']));

	//mb += db.cwiseProduct(db);
	(*m['b']) += (*g['b']).cwiseProduct((*g['b']));

	//W = (1 - decay_) * W + alpha_ * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
	(*p['W']) = (1.0f - decay_) * (*p['W']) + alpha_ * (*g['W']).cwiseQuotient((*m['W']).unaryExpr(std::ptr_fun(sqrt_eps)));

	//b += alpha_ * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));
	(*p['b']) += alpha_ * (*g['b']).cwiseQuotient((*m['b']).unaryExpr(std::ptr_fun(sqrt_eps)));

	// 'plain' fixed learning rate update
	// b += alpha * db;
	// W += alpha * dW;

}

} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */
