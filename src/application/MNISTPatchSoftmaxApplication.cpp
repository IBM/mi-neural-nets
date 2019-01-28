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
 * @file: MNISTPatchSoftmaxApplication.cpp
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

#include "MNISTPatchSoftmaxApplication.hpp"



namespace mic {
namespace application {

/*!
 * \brief Registers application.
 * \author tkornuta
 */
void RegisterApplication (void) {
	REGISTER_APPLICATION(mic::applications::MNISTPatchSoftmaxApplication);
}

} /* namespace application */

namespace applications {

MNISTPatchSoftmaxApplication::MNISTPatchSoftmaxApplication(std::string node_name_) : OpenGLContinuousLearningApplication(node_name_),
		autoencoder_filename("autoencoder_filename", "autoencoder.txt"),
		autoencoder_layers_to_be_removed("autoencoder_layers_to_be_removed", 0),
		softmax_filename("softmax_filename", "softmax.txt"),
		softmax_save("softmax_save", false),
		softmax_load("softmax_load", false)
	{
	// Register properties - so their values can be overridden (read from the configuration file).
	registerProperty(autoencoder_filename);
	registerProperty(autoencoder_layers_to_be_removed);
	registerProperty(softmax_filename);
	registerProperty(softmax_save);
	registerProperty(softmax_load);

	// Create importers.
	training_dataset_importer = new mic::data_io::MNISTPatchImporter("mnist_training_dataset_importer");
	test_dataset_importer = new mic::data_io::MNISTPatchImporter("mnist_test_dataset_importer");

	LOG(LINFO) << "Properties registered";
}

MNISTPatchSoftmaxApplication::~MNISTPatchSoftmaxApplication() {
	delete(training_dataset_importer);
	delete(test_dataset_importer);
	// Delete visualization windows.
	delete (w2d_input);
	delete (w_prob);
	delete (w_chart);
	// Delete encoders.
	delete (mnist_encoder);
	delete (label_encoder);	
}

void MNISTPatchSoftmaxApplication::initialize(int argc, char* argv[]) {

	// Initialize GLUT! :]
	VGL_MANAGER->initializeGLUT(argc, argv);

	// Create visualization window.
	w2d_input = new WindowMatrix2D("Input matrix", 256, 256, 0, 0);

	w_prob = new WindowProbability("Probabilty", 128, 256, 320, 0);

	collector_ptr = std::make_shared < mic::data_io::DataCollector<std::string, float> >( );
	// Add containers to collector.
	collector_ptr->createContainer("training_loss",  mic::types::color_rgba(0, 0, 255, 180));
	collector_ptr->createContainer("test_loss",  mic::types::color_rgba(0, 255, 0, 180));

	// Create the visualization windows - must be created in the same, main thread :]
	w_chart = new WindowCollectorChart<float>("MNISTPatchReconstruction", 0, 310, 512, 256);
	w_chart->setDataCollectorPtr(collector_ptr);

}

void MNISTPatchSoftmaxApplication::initializePropertyDependentVariables() {
	LOG(LTRACE) << "MNISTClassificationSoftmaxApplication::initializePropertyDependentVariables";

	// Get patch size.
	patch_size = training_dataset_importer->getPatchSize();

	// Allocate memory for images.
	input_image = std::make_shared<mic::types::MatrixXf >(patch_size, patch_size);
	input_target = std::make_shared<mic::types::MatrixXf >(10,1);
	decoded_prediction = std::make_shared<mic::types::MatrixXf >(10,1);

	// Set displayed matrix pointers.
	w2d_input->setMatrixPointerSynchronized(input_image);
	w_prob->setMatrixPointer1(input_target);
	w_prob->setMatrixPointer2(decoded_prediction);

	// Load datasets.
	if (!training_dataset_importer->importData())
		return;

	if (!test_dataset_importer->importData())
		return;

	// Initialize the encoders.
	mnist_encoder = new mic::encoders::MatrixXfMatrixXfEncoder(patch_size, patch_size);
	label_encoder = new mic::encoders::UIntMatrixXfEncoder(10);

	// Try to load autoencoder from file.
	if ((!softmax_load) && (neural_net.load(autoencoder_filename))) {
		LOG(LINFO) << "Loaded the autoencoder network";
		LOG(LINFO) << neural_net;

		// Remove n last layers and add softmax on top.
		neural_net.popLayer(autoencoder_layers_to_be_removed);
		neural_net.pushLayer(new Linear<float>(20, 10));
		neural_net.pushLayer(new Softmax<float>(10));
		neural_net.setLoss<  mic::neural_nets::loss::CrossEntropyLoss<float> >();

		LOG(LINFO) << "The resulting softmax network";
		LOG(LINFO) << neural_net;
	} else if ((softmax_load) && (neural_net.load(softmax_filename))) {
		// Softmax network loaded - there is nothing more to do here...
		LOG(LINFO) << "Loaded the softmax network";
		LOG(LINFO) << neural_net;
	} else {
		// We do not have the network!
		exit(1);
	}//: else

}


bool MNISTPatchSoftmaxApplication::performLearningStep() {

	// Random select sample from training dataset.
	mic::types::MNISTSample<float> sample = training_dataset_importer->getRandomSample();

	// Copy sample data to input matrix - for visualization.
	(*input_image) = (*sample.data());
	//std::cout << " input: \n" << *(input_image) << std::endl;

	// Encode data.
	mic::types::MatrixXfPtr encoded_patch = mnist_encoder->encodeSample(sample.data());
	mic::types::MatrixXfPtr encoded_label = label_encoder->encodeSample(sample.label());

	// Train the autoencoder.
	float loss = neural_net.train (encoded_patch, encoded_label, 0.005);

	// Copy encoded and label prediction - for visualization.
	(*input_target) = (*encoded_label);
	(*decoded_prediction) = (*neural_net.getPredictions());

	// Collect statistics.
	collector_ptr->addDataToContainer("training_loss", loss);

	return true;
}


void MNISTPatchSoftmaxApplication::collectTestStatistics() {
	// Random select sample from test dataset.
	mic::types::MNISTSample<float> sample = test_dataset_importer->getRandomSample();

	// Copy sample data to input matrix - for visualization.
	(*input_image) = (*sample.data());

	// Encode data.
	mic::types::MatrixXfPtr encoded_patch = mnist_encoder->encodeSample(sample.data());
	mic::types::MatrixXfPtr encoded_label = label_encoder->encodeSample(sample.label());

	// Train the autoencoder.
	float loss = neural_net.test (encoded_patch, encoded_label);

	// Collect statistics.
	collector_ptr->addDataToContainer("test_loss", loss);
}


void MNISTPatchSoftmaxApplication::populateTestStatistics() {
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
	if (softmax_save)
		neural_net.save(softmax_filename);
}



} /* namespace applications */
} /* namespace mic */
