/*!
 * @file: LossTests.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 9, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#include "LossTests.hpp"
#include "SquaredErrorLoss.hpp"
#include "CrossEntropyLoss.hpp"


/*!
 * Tests squared error loss function on vectors with four floats.
 */
TEST_F(Vectors4Float, SquareErrorLoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;

	ASSERT_EQ(loss.calculateLoss(predicted_y, target_y), (float)4.0);
}

/*!
 * Tests squared error loss function on vectors with four elements.
 */
TEST_F(Vectors4Float, SquareErrorGradient) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	mic::types::MatrixPtr<float> dy = loss.calculateGradient(predicted_y, target_y);

	for (size_t i=0; i<dy->size(); i++){
		ASSERT_EQ((*dy)[i], -2.0) << "Gradient error at position i=" << i;
	}
}

/*!
 * Tests squared error loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4Float3, SquareErrorLoss) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(predicted_y1, target_y);
	ASSERT_LE(std::abs(l1-0.045), eps);

	float l2 = loss.calculateLoss(predicted_y2, target_y);
	ASSERT_LE(std::abs(l2-0.045), eps);
}

/*!
 * Tests squared error gradient on 2 different predicted vectors with four elements.
 */
TEST_F(Vectors4Float3, SquareErrorGradient) {
	// Loss function.
	mic::neural_nets::loss::SquaredErrorLoss<float> loss;
	float eps = 1e-5;

	mic::types::MatrixPtr<float> dy1 = loss.calculateGradient(predicted_y1, target_y);
	ASSERT_LE(std::abs((*dy1)[0] + 0.3), eps) << "Gradient error at position i=0";
	ASSERT_LE(std::abs((*dy1)[1] - 0.3), eps) << "Gradient error at position i=1";
	ASSERT_LE(std::abs((*dy1)[2] + 0.0), eps) << "Gradient error at position i=2";
	ASSERT_LE(std::abs((*dy1)[3] + 0.0), eps) << "Gradient error at position i=3";

	mic::types::MatrixPtr<float> dy2 = loss.calculateGradient(predicted_y2, target_y);
	ASSERT_LE(std::abs((*dy2)[0] + 0.0), eps) << "Gradient error at position i=0";
	ASSERT_LE(std::abs((*dy2)[1] + 0.0), eps) << "Gradient error at position i=1";
	ASSERT_LE(std::abs((*dy2)[2] + 0.3), eps) << "Gradient error at position i=2";
	ASSERT_LE(std::abs((*dy2)[3] - 0.3), eps) << "Gradient error at position i=3";
}



/*!
 * Tests cross-entropy loss function on 2 different predicted vectors with four floats.
 */
TEST_F(Vectors4Float3, CrossEntropyLoss) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	float eps = 1e-5;

	float l1 = loss.calculateLoss(predicted_y1, target_y);
	ASSERT_LE(std::abs(l1 - 2.0), eps);

	float l2 = loss.calculateLoss(predicted_y2, target_y);
	ASSERT_LE(std::abs(l2-2.02193), eps);
}


/*!
 * Tests cross-entropy gradient on 2 different predicted vectors with four elements.
 */
TEST_F(Vectors4Float3, CrossEntropyGradient) {
	// Loss function.
	mic::neural_nets::loss::CrossEntropyLoss<float> loss;
	double eps = 1e-5;

	mic::types::MatrixPtr<float> dy1 = loss.calculateGradient(predicted_y1, target_y);
	ASSERT_LE(std::abs((*dy1)[0] - 0.15), eps) << "Gradient error at position i=0";
	ASSERT_LE(std::abs((*dy1)[1] + 0.15), eps) << "Gradient error at position i=1";
	ASSERT_LE(std::abs((*dy1)[2] + 0.0), eps) << "Gradient error at position i=2";
	ASSERT_LE(std::abs((*dy1)[3] + 0.0), eps) << "Gradient error at position i=3";

	mic::types::MatrixPtr<float> dy2 = loss.calculateGradient(predicted_y2, target_y);
	ASSERT_LE(std::abs((*dy2)[0] + 0.0), eps) << "Gradient error at position i=0";
	ASSERT_LE(std::abs((*dy2)[1] + 0.0), eps) << "Gradient error at position i=1";
	ASSERT_LE(std::abs((*dy2)[2] - 0.15), eps) << "Gradient error at position i=2";
	ASSERT_LE(std::abs((*dy2)[3] + 0.15), eps) << "Gradient error at position i=3";
}




int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
