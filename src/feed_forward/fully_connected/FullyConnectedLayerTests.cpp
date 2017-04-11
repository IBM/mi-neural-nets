/*!
 * @file: FullyConnectedLayerTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 4, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "FullyConnectedLayerTests.hpp"

/*!
 * Makes sure that the layer is properly initialized - initial W weights must be non zero and b must be zero.
 */
TEST_F(FullyConnectedLayer5x2Float, WbInitialization) {
	for (size_t i=0; i<10; i++)
		ASSERT_NE( (*layer.W)[i], 0.0 ) << "Weight W is zero at position i=" << i;

	for (size_t i=0; i<2; i++)
		ASSERT_EQ( (*layer.b)[i], 0.0 ) << "Bias b is non-zero at position i=" << i;
}

/*!
 * Makes sure that the layer is properly initialized - all W are numbers!
 */
TEST_F(FullyConnectedLayer5x2Float, WIsNaN) {
	for (size_t i=0; i<10; i++)
		ASSERT_EQ( std::isnan((*layer.W)[i]), false )  << "Weight W is NaN at position i=" << i;
}

/*!
 * Makes sure that the layer is properly initialized - all W are finite.
 */
TEST_F(FullyConnectedLayer5x2Float, WIsNotInf) {
	for (size_t i=0; i<10; i++)
		ASSERT_EQ( std::isinf((*layer.W)[i]), false ) << "Weight W is Inf at position i=" << i;
}


/*!
 * Makes sure that the layer is properly initialized and all W are different.
 */
TEST_F(FullyConnectedLayer5x2Float, WAreDifferent) {
	for (size_t i=0; i<10; i++) {
		for (size_t j=i+1; j<10; j++)
			ASSERT_NE( (*layer.W)[i], (*layer.W)[j] ) << "Weights at positions i=" << i << " and j=" << j << "are equal";
	}//: for i
}


/*!
 * Makes sure that the layer calculates y = w*x + b, size of layer: is 1x1.
 */
TEST_F(FullyConnectedLayer1x1Float, Forward_y) {
	mic::types::MatrixPtr<float> input = MAKE_MATRIX_PTR(float, 1, 1);

	(*input)[0] = 0.0;
	ASSERT_EQ( (*layer.forward( input ))[0], 1.0 );

	(*input)[0] = 1.0;
	ASSERT_EQ( (*layer.forward( input ))[0], 2.0 );
}

/*!
 * Makes sure that the layer calculates y = w*x + b, size of layer: is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Float, Forward_y) {
	mic::types::MatrixPtr<float> y = layer.forward(const_x);
	ASSERT_EQ((*y)[0], -2 );
	ASSERT_EQ((*y)[1], 0 );
	ASSERT_EQ((*y)[2], 2 );
}

/*!
 * Makes sure that the two stacked layers will return right result.
 */
TEST(FullyConnectedLayerStacked1x2x3Float, Forward_y) {
	// Initialize network consisting  of two layers.
	mic::neural_nets::feed_forward::FullyConnectedLayer<float> l1(1,2);
	(*l1.W) << 1.0, 2.0;
	(*l1.b) << 0.0, 1.0;

	mic::neural_nets::feed_forward::FullyConnectedLayer<float> l2(2,3);
	(*l2.W) << -1, -2, -3, -5, 6, 9;
	(*l2.b) << -3, -2, -1;

	// Input.
	mic::types::MatrixPtr<float> x = MAKE_MATRIX_PTR(float, 1, 1);
	(*x) << -1;

	// Check the result.
	mic::types::MatrixPtr<float> y = l2.forward(l1.forward(x));
	ASSERT_EQ((*y)[0], 0 );
	ASSERT_EQ((*y)[1], 6 );
	ASSERT_EQ((*y)[2], -16 );
}


/*!
 * Tests backward pass in the  y = w*x + b, size of layer: is 2x1.
 */
TEST(FullyConnectedLayer2x1Float, Backward_dx) {
	mic::neural_nets::feed_forward::FullyConnectedLayer<> layer(2,1);
	(*layer.W) << 1.0, 2.0;
	(*layer.b) << 1.0;

	// Output.
	mic::types::MatrixPtr<float> dy = MAKE_MATRIX_PTR(float, 1, 1);
	(*dy) << 2.0;

	mic::types::MatrixPtr<float> dx = layer.backward(dy);
	ASSERT_EQ((*dx)[0], 2 );
	ASSERT_EQ((*dx)[1], 4 );
}


/*!
 * Tests backward pass in the  y = w*x + b (dx gradient), size of layer: is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Float, Backward_dx) {
	mic::types::MatrixPtr<float> dx = layer.backward(const_dy);

	// Check dx.
	ASSERT_EQ((*dx)[0], -1);
	ASSERT_EQ((*dx)[1], -3);
}

/*!
 * Tests gradients dW and db, size of layer is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Float, Backward_dWdb) {
	// Forward pass.
	mic::types::MatrixPtr<float> y = layer.forward(const_x);
	// Backward pass.
	mic::types::MatrixPtr<float> dx = layer.backward(const_dy);

	// Check dW.
	ASSERT_EQ((*layer.dW)[0], 1);
	ASSERT_EQ((*layer.dW)[1], -1);
	ASSERT_EQ((*layer.dW)[2], 2);
	ASSERT_EQ((*layer.dW)[3], -2);
	ASSERT_EQ((*layer.dW)[4], -1);
	ASSERT_EQ((*layer.dW)[5], 1);

	// Check db.
	for (size_t i=0; i<3; i++)
		ASSERT_EQ((*layer.db)[i], (*const_dy)[i]);
}


double calculateLoss (mic::types::MatrixPtr<double> predicted_y_, mic::types::MatrixPtr<double> target_y_) {
	// Sizes must match.
	assert(predicted_y_->size() == target_y_->size());

	// Calculate loss.
	double loss =0;
	for (size_t i=0; i <(size_t)predicted_y_->size(); i++) {
		loss += ((*predicted_y_)[i] - (*target_y_)[i])*((*predicted_y_)[i] - (*target_y_)[i]);
	}
	return loss;
}

/*!
 * Numerical gradient test dW, size of layer is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Double, NumericalGradientCheck_dW) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dW = MAKE_MATRIX_PTR(double, *layer.dW);

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nW = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, layer.W, loss, delta);

	// Compare gradients.
	double eps = 1e-8;
	for (size_t i=0; i<(size_t)dW->size(); i++)
		EXPECT_LE(((*dW)[i] - (*nW)[i]), eps) << "Too big difference between dW and numerical dW at position i=" << i;
}


/*!
 * Numerical gradient test db, size of layer is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Double, NumericalGradientCheck_db) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> db = MAKE_MATRIX_PTR(double, *layer.db);

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nb = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, layer.b, loss, delta);

	// Compare gradients.
	double eps = 1e-8;
	for (size_t i=0; i<(size_t)db->size(); i++)
		EXPECT_LE(((*db)[i] - (*nb)[i]), eps) << "Too big difference between db and numerical db at position i=" << i;
}


/*!
 * Numerical gradient test dx, size of layer is 2x3.
 */
TEST_F(FullyConnectedLayer2x3Double, NumericalGradientCheck_dx) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dx = MAKE_MATRIX_PTR(double, *layer.backward(dy));

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nx = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, const_x, loss, delta);

	// Compare gradients.
	double eps = 1e-8;
	for (size_t i=0; i<(size_t)dx->size(); i++)
		EXPECT_LE(((*dx)[i] - (*nx)[i]), eps) << "Too big difference between dx and numerical dx at position i=" << i;
}



/*!
 * Numerical gradient test dW, size of layer is 50x100.
 */
TEST_F(FullyConnectedLayer50x100Double, NumericalGradientCheck_dW) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dW = MAKE_MATRIX_PTR(double, *layer.dW);

//	(*layer.W)[0] = 50000.0;

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nW = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, layer.W, loss, delta);


	// Compare gradients.
	double eps = 1e-6;
	for (size_t i=0; i<(size_t)dW->size(); i++){
		//std::cout << "i=" << i << " (*dW)[i]= " << (*dW)[i] << " (*nW)[i]= " << (*nW)[i] << std::endl;
		ASSERT_LE(((*dW)[i] - (*nW)[i]), eps) << "Too big difference between dW and numerical dW at position i=" << i;
	}
}


/*!
 * Numerical gradient test db, size of layer is 50x100.
 */
TEST_F(FullyConnectedLayer50x100Double, NumericalGradientCheck_db) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	layer.backward(dy);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> db = MAKE_MATRIX_PTR(double, *layer.db);

//	(*layer.W)[0] = 50000.0;

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nb = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, layer.b, loss, delta);

	// Compare gradients.
	double eps = 1e-6;
	for (size_t i=0; i<(size_t)db->size(); i++)
		ASSERT_LE(((*db)[i] - (*nb)[i]), eps) << "Too big difference between db and numerical db at position i=" << i;
}


/*!
 * Numerical gradient test dx, size of layer is 50x100.
 */
TEST_F(FullyConnectedLayer50x100Double, NumericalGradientCheck_dx) {

	// Calculate gradients.
	mic::types::MatrixPtr<double> predicted_y = layer.forward(const_x);
	mic::types::MatrixPtr<double> dy = loss.calculateGradient(predicted_y, target_y);
	// Store resulting gradients - make a copy!
	mic::types::MatrixPtr<double> dx = MAKE_MATRIX_PTR(double, *layer.backward(dy));

//	(*layer.W)[0] = 50000.0;

	// Calculate numerical gradients.
	double delta = 1e-5;
	mic::types::MatrixPtr<double> nx = layer.calculateNumericalGradient<mic::neural_nets::loss::RegressionLoss<double> >(const_x, target_y, const_x, loss, delta);

	// Compare gradients.
	double eps = 1e-6;
	for (size_t i=0; i<(size_t)dx->size(); i++)
		ASSERT_LE(((*dx)[i] - (*nx)[i]), eps) << "Too big difference between dx and numerical dx at position i=" << i;
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
