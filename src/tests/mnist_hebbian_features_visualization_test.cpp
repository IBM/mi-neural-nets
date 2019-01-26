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
 * @file mnist_hebbian_features_visualization_test.cpp
 * @brief Program for visualization of features of hebbian network trained on MNIST digits.
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 16, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>

#include <data_io/MNISTMatrixImporter.hpp>

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <application/ApplicationState.hpp>

#include <configuration/ParameterServer.hpp>

#include <opengl/visualization/WindowManager.hpp>
#include <opengl/visualization/WindowGrayscaleBatch.hpp>
using namespace mic::opengl::visualization;

// Hebbian neural net.
#include <mlnn/HebbianNeuralNetwork.hpp>
using namespace mic::mlnn;

// Encoders.
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

/// Window for displaying the MNIST batch.
WindowGrayscaleBatch<float>* w_input;
WindowGrayscaleBatch<float>* w_reconstruction;
/// Window for displaying the weights.
WindowGrayscaleBatch<float>* w_weights1;
WindowGrayscaleBatch<float>* w_weights2;

/// MNIST importer.
mic::data_io::MNISTMatrixImporter<float>* importer;
/// Multi-layer neural network.
HebbianNeuralNetwork<float> neural_net;

/// MNIST matrix encoder.
mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
//mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 4;
const size_t output_units = 12;

/*!
 * \brief Function for batch sampling.
 * \author tkornuta
 */
void batch_function (void) {

/*	if (neural_net.load(fileName)) {
		LOG(LINFO) << "Loaded neural network from a file";
	} else {*/
		{
		// Create a simple hebbian network.
		neural_net.pushLayer(new BinaryCorrelator<float>(patch_size*patch_size, output_units, 0.6, 28*28*0.01));
		neural_net.setOptimization<  mic::neural_nets::learning::NormalizedHebbianRule<float> >();

		LOG(LINFO) << "Generated new neural network";
	}//: else

	size_t iteration = 0;

	// Main application loop.
	while (!APP_STATE->Quit()) {

		// If not paused.
		if (!APP_STATE->isPaused()) {

			// If single step mode - pause after the step.
			if (APP_STATE->isSingleStepModeOn())
				APP_STATE->pressPause();

			{ // Enter critical section - with the use of scoped lock from AppState!
				APP_DATA_SYNCHRONIZATION_SCOPED_LOCK();

				// Retrieve the next minibatch.
				mic::types::MNISTBatch<float> bt = importer->getRandomBatch();

				// Set batch to be displayed.
				w_input->setBatchUnsynchronized(bt.data());

				// Encode data.
				mic::types::MatrixXfPtr encoded_batch = mnist_encoder->encodeBatch(bt.data());
				mic::types::MatrixXfPtr encoded_labels = mnist_encoder->encodeBatch(bt.data());

				// Train the autoencoder.
				float loss = neural_net.train (encoded_batch, 0.05);

				// Get reconstruction.
				mic::types::MatrixXfPtr encoded_reconstruction = neural_net.getPredictions();

				std::vector<mic::types::MatrixXfPtr> decoded_reconstruction = mnist_encoder->decodeBatch(encoded_reconstruction);
				w_reconstruction->setBatchUnsynchronized(decoded_reconstruction);

				if (iteration%10 == 0) {
					// Visualize the weights.
					std::shared_ptr<mic::mlnn::BinaryCorrelator<float> > layer1 = neural_net.getLayer<mic::mlnn::BinaryCorrelator<float> >(0);
					w_weights1->setBatchUnsynchronized(layer1->getActivations(patch_size, patch_size));

				}//: if

				iteration++;
				LOG(LINFO) << "Iteration: " << iteration << "loss= " << loss;
			}//: end of critical section

		}//: if

		// Sleep.
		APP_SLEEP();
	}//: while

}//: image_encoder_and_visualization_test



/*!
 * \brief Main program function. Runs two threads: main (for GLUT) and another one (for data processing).
 * \author tkornuta
 * @param[in] argc Number of parameters (passed to glManaged).
 * @param[in] argv List of parameters (passed to glManaged).
 * @return (not used)
 */
int main(int argc, char* argv[]) {
	// Set console output to logger.
	LOGGER->addOutput(new ConsoleOutput());
	LOG(LINFO) << "Logger initialized. Starting application";

	// Parse parameters.
	PARAM_SERVER->parseApplicationParameters(argc, argv);

	// Initilize application state ("touch it") ;)
	APP_STATE;

	// Load dataset.
	importer = new mic::data_io::MNISTMatrixImporter<float>();
	importer->setBatchSize(batch_size);

	// Initialize the encoders.
	mnist_encoder = new mic::encoders::MatrixXfMatrixXfEncoder(patch_size, patch_size);
	//label_encoder = new mic::encoders::UIntMatrixXfEncoder(batch_size);

	// Set parameters of all property-tree derived objects - USER independent part.
	PARAM_SERVER->loadPropertiesFromConfiguration();

	// Initialize property-dependent variables of all registered property-tree objects - USER dependent part.
	PARAM_SERVER->initializePropertyDependentVariables();

	// Import data from datasets.
	if (!importer->importData())
		return -1;

	// Initialize GLUT! :]
	VGL_MANAGER->initializeGLUT(argc, argv);

	// Create batch visualization window.
	w_input = new WindowGrayscaleBatch<float>("Input batch", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70, 0, 250, 250);
	w_reconstruction = new WindowGrayscaleBatch<float>("Reconstructed batch", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 320, 0, 250, 250);
	w_weights1 = new WindowGrayscaleBatch<float>("Permanences", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 570, 0, 250, 250);
//	w_weights2 = new WindowGrayscaleBatch<float>("Connectivity", 1092, 0, 512, 512);

	boost::thread batch_thread(boost::bind(&batch_function));

	// Start visualization thread.
	VGL_MANAGER->startVisualizationLoop();

	LOG(LINFO) << "Waiting for threads to join...";
	// End test thread.
	batch_thread.join();
	LOG(LINFO) << "Threads joined - ending application";
}//: main
