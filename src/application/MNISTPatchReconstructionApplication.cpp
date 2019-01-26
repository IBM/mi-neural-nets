/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * @file: MNISTPatchReconstructionApplication.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   27-03-2017
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#include "MNISTPatchReconstructionApplication.hpp"



namespace mic {
namespace application {

/*!
 * \brief Registers application.
 * \author tkornuta
 */
void RegisterApplication (void) {
	REGISTER_APPLICATION(mic::applications::MNISTPatchReconstructionApplication);
}

} /* namespace application */

namespace applications {

MNISTPatchReconstructionApplication::MNISTPatchReconstructionApplication(std::string node_name_) : OpenGLContinuousLearningApplication(node_name_),
		mlnn_filename("mlnn_filename", "mlnn.txt"),
		mlnn_save("mlnn_save", false),
		mlnn_load("mlnn_load", false)
	{
	// Register properties - so their values can be overridden (read from the configuration file).
	registerProperty(mlnn_filename);
	registerProperty(mlnn_save);
	registerProperty(mlnn_load);

	LOG(LINFO) << "Properties registered";

	// Create importers.
	training_dataset_importer = new mic::data_io::MNISTPatchImporter("mnist_training_dataset_importer");
	test_dataset_importer = new mic::data_io::MNISTPatchImporter("mnist_test_dataset_importer");

}

MNISTPatchReconstructionApplication::~MNISTPatchReconstructionApplication() {
	delete(training_dataset_importer);
	delete(test_dataset_importer);
}

void MNISTPatchReconstructionApplication::initialize(int argc, char* argv[]) {

	// Initialize GLUT! :]
	VGL_MANAGER->initializeGLUT(argc, argv);

	// Create two visualization windows
	w2d_input = new WindowMatrix2D("Input matrix", 0, 0, 256, 256);
	w2d_reconstruction = new WindowMatrix2D("Reconstructed matrix", 320, 0, 256, 256);

	collector_ptr = std::make_shared < mic::data_io::DataCollector<std::string, float> >( );
	// Add containers to collector.
	collector_ptr->createContainer("training_loss",  mic::types::color_rgba(0, 0, 255, 180));
	collector_ptr->createContainer("test_loss",  mic::types::color_rgba(0, 255, 0, 180));

	// Create the visualization windows - must be created in the same, main thread :]
	w_chart = new WindowCollectorChart<float>("MNISTPatchReconstruction", 0, 310, 512, 256);
	w_chart->setDataCollectorPtr(collector_ptr);

}

void MNISTPatchReconstructionApplication::initializePropertyDependentVariables() {
	LOG(LTRACE) << "MNISTClassificationSoftmaxApplication::initializePropertyDependentVariables";

	// Get patch size.
	patch_size = training_dataset_importer->getPatchSize();

	// Allocate memory for images.
	input_image = std::make_shared<mic::types::MatrixXf >(patch_size, patch_size);
	reconstructed_image = std::make_shared<mic::types::MatrixXf >(patch_size, patch_size);

	// Set displayed matrix pointers.
	w2d_input->setMatrixPointerSynchronized(input_image);
	w2d_reconstruction->setMatrixPointerSynchronized(reconstructed_image);

	// Load datasets.
	if (!training_dataset_importer->importData())
		return;

	if (!test_dataset_importer->importData())
		return;

	// Try to load neural network from file.
	if ((mlnn_load) && (neural_net.load(mlnn_filename))) {
		// Do nothing ;)
	} else {
		// Create a simple autoencoder.
		neural_net.pushLayer(new Linear<float>(patch_size*patch_size, 20));
		neural_net.pushLayer(new ReLU<float>(20));

/*		neural_net.pushLayer(new Linear<float>(20, 5));
		neural_net.pushLayer(new ReLU<float>(5));

		neural_net.pushLayer(new Linear<float>(5, 20));
		neural_net.pushLayer(new ReLU<float>(20));*/

		neural_net.pushLayer(new Linear<float>(20, patch_size*patch_size));
		neural_net.pushLayer(new ReLU<float>(patch_size*patch_size));
		neural_net.setLoss<  mic::neural_nets::loss::SquaredErrorLoss<float> >();

		LOG(LINFO) << "Generated new neural network";
	}//: else

}


bool MNISTPatchReconstructionApplication::performLearningStep() {

	// Random select sample from training dataset.
	mic::types::MNISTSample<float> sample = training_dataset_importer->getRandomSample();

	// Copy sample data to input matrix - for visualization.
	(*input_image) = (*sample.data());
	//std::cout << " input: \n" << *(input_image) << std::endl;

	// Encode sample data....
	mic::types::MatrixXfPtr encoded_patch (new mic::types::MatrixXf(*sample.data()));
	// ... i.e. reshape it.
	encoded_patch->resize(patch_size*patch_size, 1);

	// Train the autoencoder.
	float loss = neural_net.train (encoded_patch, encoded_patch, 0.005);
	//std::cout << loss << std::endl;
	collector_ptr->addDataToContainer("training_loss", loss);

	// Get reconstruction.
	mic::types::MatrixXfPtr encoded_reconstruction = neural_net.getPredictions();

	(*reconstructed_image) = (*encoded_reconstruction);
	reconstructed_image->resize(patch_size, patch_size);
	return true;
}


void MNISTPatchReconstructionApplication::collectTestStatistics() {
	// Random select sample from test dataset.
	mic::types::MNISTSample<float> sample = test_dataset_importer->getRandomSample();

	// Copy sample data to input matrix - for visualization.
	(*input_image) = (*sample.data());

	// Encode sample data....
	mic::types::MatrixXfPtr encoded_patch (new mic::types::MatrixXf(*sample.data()));
	// ... i.e. reshape it.
	encoded_patch->resize(patch_size*patch_size, 1);

	// Train the autoencoder.
	float loss = neural_net.test (encoded_patch, encoded_patch);

	// Get reconstruction.
	mic::types::MatrixXfPtr decoded_reconstruction = neural_net.getPredictions();

	(*reconstructed_image) = (*decoded_reconstruction);
	reconstructed_image->resize(patch_size, patch_size);

	// Collect statistics.
	collector_ptr->addDataToContainer("test_loss", loss);

}


void MNISTPatchReconstructionApplication::populateTestStatistics() {
	// Average the sums.
	/*classification_cost_sum /= (float)number_of_averaged_test_measures;
	correct_classification_factor_sum /= (float)number_of_averaged_test_measures;

	LOG(LINFO)<< "Iteration = " << iteration << " classification_cost_sum = " << classification_cost_sum;

	// Add data to chart window.

	// Reset partial sums.
	classification_cost_sum = 0;
	correct_classification_factor_sum = 0;
	*/

	//collector_ptr->addDataToContainer("reconstruction", classification_cost_sum);
	//classification_cost_sum += .001;

	LOG(LINFO)<< "Iteration = " << iteration;

	// Save nn to file.
	if (mlnn_save)
		neural_net.save(mlnn_filename);
}



} /* namespace applications */
} /* namespace mic */
