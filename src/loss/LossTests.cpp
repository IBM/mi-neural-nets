/*!
 * @file: LossTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 9, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "LossTests.hpp"
#include "LossTypes.hpp"


/*!
 * Tests squared error loss function on vector with four floats.
 */
TEST_F(Vectors4x1Float, SquaredErrorLoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;

	ASSERT_EQ(loss.calculateLoss(target_y, predicted_y), (float)2.0);
}

/*!
 * Tests mean squared error (MSE) loss function on vector with four floats.
 */
TEST_F(Vectors4x1Float, SquaredErrorMSELoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;

	ASSERT_EQ(loss.calculateMeanLoss(target_y, predicted_y), (float)2.0);
}

/*!
 * Tests squared error loss function on vectors with four elements.
 */
TEST_F(Vectors4x1Float, SquaredErrorGradient) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	mic::types::MatrixPtr<float> dy = loss.calculateGradient(target_y, predicted_y);

	for (size_t i=0; i<(size_t)dy->size(); i++){
		ASSERT_EQ((*dy)[i], -1.0) << "Gradient error at position i=" << i << " (*dy)[i] is " << (*dy)[i] << " whereas -1.0 is expected";
	}
}

/*!
 * Tests squared error loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4x1Float2, SquaredErrorLoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(target_y, predicted_y1);
	EXPECT_LE(fabs(l1-0.0225), eps);

	float l2 = loss.calculateLoss(target_y, predicted_y2);
	EXPECT_LE(fabs(l2-0.0225), eps);
}

/*!
 * Tests mean squared error loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4x1Float2, SquaredErrorMSELoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateMeanLoss(target_y, predicted_y1);
	EXPECT_LE(fabs(l1-0.0225), eps);

	float l2 = loss.calculateMeanLoss(target_y, predicted_y2);
	EXPECT_LE(fabs(l2-0.0225), eps);
}

/*!
 * Tests squared error gradient on 2 different predicted vectors with four elements.
 */
TEST_F(Vectors4x1Float2, SquaredErrorGradient) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	mic::types::MatrixPtr<float> dy1 = loss.calculateGradient(target_y, predicted_y1);
	EXPECT_LE(fabs((*dy1)[0] + 0.15), eps) << "Gradient error at position i=0";
	EXPECT_LE(fabs((*dy1)[1] - 0.15), eps) << "Gradient error at position i=1";
	EXPECT_LE(fabs((*dy1)[2] - 0.0), eps) << "Gradient error at position i=2";
	EXPECT_LE(fabs((*dy1)[3] - 0.0), eps) << "Gradient error at position i=3";

	mic::types::MatrixPtr<float> dy2 = loss.calculateGradient(target_y, predicted_y2);
	EXPECT_LE(fabs((*dy2)[0] - 0.0), eps) << "Gradient error at position i=0";
	EXPECT_LE(fabs((*dy2)[1] - 0.0), eps) << "Gradient error at position i=1";
	EXPECT_LE(fabs((*dy2)[2] + 0.15), eps) << "Gradient error at position i=2";
	EXPECT_LE(fabs((*dy2)[3] - 0.15), eps) << "Gradient error at position i=3";
}



/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4x1Float2, CrossEntropyLoss) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(target_y, predicted_y1);
	EXPECT_LE(fabs(l1 - 2.0), eps);

	float l2 = loss.calculateLoss(target_y, predicted_y2);
	EXPECT_LE(fabs(l2 - 2.02193), eps);
}

/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4x1Float2, CrossEntropyACELoss) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateMeanLoss(target_y, predicted_y1);
	EXPECT_LE(fabs(l1 - 2.0), eps);

	float l2 = loss.calculateMeanLoss(target_y, predicted_y2);
	EXPECT_LE(fabs(l2 - 2.02193), eps);
}


/*!
 * Tests cross-entropy gradient on 2 different predicted vectors with four elements.
 */
TEST_F(Vectors4x1Float2, CrossEntropyGradient) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	double eps = 1e-5;

	mic::types::MatrixPtr<float> dy1 = loss.calculateGradient(target_y, predicted_y1);
	EXPECT_LE(fabs((*dy1)[0] + 0.15), eps) << "Gradient error at position i=0";
	EXPECT_LE(fabs((*dy1)[1] - 0.15), eps) << "Gradient error at position i=1";
	EXPECT_LE(fabs((*dy1)[2] + 0.0), eps) << "Gradient error at position i=2";
	EXPECT_LE(fabs((*dy1)[3] + 0.0), eps) << "Gradient error at position i=3";

	mic::types::MatrixPtr<float> dy2 = loss.calculateGradient(target_y, predicted_y2);
	EXPECT_LE(fabs((*dy2)[0] + 0.0), eps) << "Gradient error at position i=0";
	EXPECT_LE(fabs((*dy2)[1] + 0.0), eps) << "Gradient error at position i=1";
	EXPECT_LE(fabs((*dy2)[2] + 0.15), eps) << "Gradient error at position i=2";
	EXPECT_LE(fabs((*dy2)[3] - 0.15), eps) << "Gradient error at position i=3";
}


/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors3x2Float, SquaredErrorLoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(target_y, predicted_y);
	EXPECT_LE(fabs(l1 - 0.145), eps);
}

/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors3x2Float, SquaredErrorACELoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateMeanLoss(target_y, predicted_y);
	EXPECT_LE(fabs(l1 - 0.0725), eps);
}


/*!
 * Tests squared error loss function on vectors with four elements.
 */
TEST_F(Vectors3x2Float, SquaredErrorGradient) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	double eps = 1e-5;

	mic::types::MatrixPtr<float> dy = loss.calculateGradient(target_y, predicted_y);

	EXPECT_LE(fabs((*dy)(0,0) - 0.1), eps) << "Gradient error at position (0,0)";
	EXPECT_LE(fabs((*dy)(0,1) + 0.0), eps) << "Gradient error at position (0,1)";
	EXPECT_LE(fabs((*dy)(1,0) - 0.1), eps) << "Gradient error at position (1,0)";
	EXPECT_LE(fabs((*dy)(1,1) + 0.1), eps) << "Gradient error at position (1,1)";
	EXPECT_LE(fabs((*dy)(2,0) + 0.1), eps) << "Gradient error at position (2,0)";
	EXPECT_LE(fabs((*dy)(2,1) - 0.5), eps) << "Gradient error at position (2,1)";
}



/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors3x2Float, CrossEntropyLoss) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(target_y, predicted_y);
	EXPECT_LE(fabs(l1 - 2.42782), eps);

}

/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors3x2Float, CrossEntropyACELoss) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateMeanLoss(target_y, predicted_y);
	EXPECT_LE(fabs(l1 - 1.21391), eps);
}

/*!
 * Tests squared error loss function on vectors with four elements.
 */
TEST_F(Vectors3x2Float, CrossEntropyGradient) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	double eps = 1e-5;

	mic::types::MatrixPtr<float> dy = loss.calculateGradient(target_y, predicted_y);

	EXPECT_LE(fabs((*dy)(0,0) - 0.1), eps) << "Gradient error at position (0,0)";
	EXPECT_LE(fabs((*dy)(0,1) + 0.0), eps) << "Gradient error at position (0,1)";
	EXPECT_LE(fabs((*dy)(1,0) - 0.1), eps) << "Gradient error at position (1,0)";
	EXPECT_LE(fabs((*dy)(1,1) + 0.1), eps) << "Gradient error at position (1,1)";
	EXPECT_LE(fabs((*dy)(2,0) + 0.1), eps) << "Gradient error at position (2,0)";
	EXPECT_LE(fabs((*dy)(2,1) - 0.5), eps) << "Gradient error at position (2,1)";
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
