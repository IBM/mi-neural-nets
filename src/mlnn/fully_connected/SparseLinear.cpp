/*!
 * @file: SparseLinear.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   05-04-2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#include <mlnn/fully_connected/SparseLinear.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {


SparseLinear::SparseLinear(size_t inputs_, size_t outputs_, size_t batch_size_, std::string name_) :
	Layer(inputs_, outputs_, batch_size_, LayerTypes::SparseLinear, name_) {

	// Allocate parameters: weight and bias.
	p.add (std::make_tuple ( "W", outputs_, inputs_ ));
	p.add (std::make_tuple ( "b", outputs_, 1 ));

	// Initialize weights of the W matrix.
	double range = sqrt(6.0 / double(inputs_ + outputs_));
	p['W']->rand(-range, range);
	p['b']->rand(-range, range);// setZero();

	//mW = (Eigen::MatrixXf)Eigen::MatrixXf::Zero(W.rows(), W.cols());
	m.add (std::make_tuple ( "W", outputs_, inputs_ ));
	m.add (std::make_tuple ( "b", outputs_, 1 ));

	// Allocate gradients: W and b.
	g.add (std::make_tuple ( "W", outputs_, inputs_ ));
	g.add (std::make_tuple ( "b", outputs_, 1 ));

};

void SparseLinear::forward(bool test_) {
	// y = W * x + b
	(*s['y']) = (*p['W']) * (*s['x']) + (*p['b']).replicate(1, (*s['x']).cols());

}

void SparseLinear::backward() {

	/*dW = dy * x.transpose();
	db = dy.rowwise().sum();
	dx = W.transpose() * dy;*/

	(*g['W']) = (*g['y']) * ((*s['x']).transpose());
	(*g['b']) = (*g['y']).rowwise().sum();
	(*g['x']) = (*p['W']).transpose() * (*g['y']);

}

void SparseLinear::resetGrads() {

	(*g['W']).setZero();
	(*g['b']).setZero();
}

void SparseLinear::applyGrads(double alpha_, double decay_) {

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
