/*!
 * @file mnist_mlnn_features_visualization_test.cpp
 * @brief Program for visualization of features of mlnn layer trained on MNIST digits.
 * @author tkornuta
 * @date:   03-04-2017
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

// Neural net.
#include <mlnn/MultiLayerNeuralNetwork.hpp>
using namespace mic::mlnn;

// Encoders.
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

/// Window for displaying the MNIST batch.
WindowGrayscaleBatch* w_input;
WindowGrayscaleBatch* w_reconstruction;
/// Window for displaying the weights.
WindowGrayscaleBatch* w_weights;

/// MNIST importer.
mic::data_io::MNISTMatrixImporter* importer;
/// Multi-layer neural network.
MultiLayerNeuralNetwork<float, mic::neural_nets::loss::CrossEntropyLoss<float> > neural_net;

/// MNIST matrix encoder.
mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
//mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 16;
const size_t hidden_layer_units = 100;
const char* fileName = "nn_autoencoder_weights_visualization.txt";


/*!
 * \brief Function for batch sampling.
 * \author tkornuta
 */
void batch_function (void) {

/*	if (neural_net.load(fileName)) {
		LOG(LINFO) << "Loaded neural network from a file";
	} else {*/
		{
		// Create a simple autoencoder.
		neural_net.pushLayer(new Linear<float>(patch_size*patch_size, hidden_layer_units));
		neural_net.pushLayer(new ReLU<float>(hidden_layer_units));

/*		neural_net.pushLayer(new Linear(hidden_layer_units, 20));
		neural_net.pushLayer(new ReLU(20));

		neural_net.pushLayer(new Linear(20, hidden_layer_units));
		neural_net.pushLayer(new ReLU(hidden_layer_units));*/

		neural_net.pushLayer(new Linear<float>(hidden_layer_units, patch_size*patch_size));
		neural_net.pushLayer(new ReLU<float>(patch_size*patch_size));
		LOG(LINFO) << "Generated new neural network";
	}//: else

	size_t iteration = 0;

	// Prepare temporary variables.
	Eigen::MatrixXf tmp;
	std::vector< std::shared_ptr <mic::types::MatrixXf> > weights;
	for (size_t i=0; i < (size_t)hidden_layer_units; i++) {
		// Create .
		mic::types::MatrixXfPtr row = MAKE_MATRIX_PTR(float, patch_size, patch_size);
		weights.push_back(row);
	}
	w_weights->setBatchDataSynchronized(weights);


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
				mic::types::MNISTBatch bt = importer->getRandomBatch();

				// Set batch to be displayed.
				w_input->setBatchDataUnsynchronized(bt.data());

				// Encode data.
				mic::types::MatrixXfPtr encoded_batch = mnist_encoder->encodeBatch(bt.data());
				mic::types::MatrixXfPtr encoded_labels = mnist_encoder->encodeBatch(bt.data());

				// Train the autoencoder.
				float loss = neural_net.train (encoded_batch, encoded_labels, 0.005);

				// Get reconstruction.
				mic::types::MatrixXfPtr encoded_reconstruction = neural_net.getPredictions();
				std::vector<mic::types::MatrixXfPtr> decoded_reconstruction = mnist_encoder->decodeBatch(encoded_reconstruction);
				w_reconstruction->setBatchDataUnsynchronized(decoded_reconstruction);

				if (iteration%10 == 0) {
					// Visualize the weights.
					std::shared_ptr<mic::mlnn::Linear<float> > layer = neural_net.getLayer<mic::mlnn::Linear<float> >(0);
					mic::types::MatrixXfPtr W = layer->getParam("W");
					// Iterate through "units" and generate "activation image" for each one.
					for (size_t i=0; i < (size_t)hidden_layer_units; i++) {
						// Get row.
						mic::types::MatrixXfPtr row = weights[i];
						// Copy data.
						(*row) = W->row(i);
						row->resize(patch_size, patch_size);
						// Calculate l2 norm.
						float l2 = row->norm();
						// Normalize the inputs to <-0.5,0.5> and add 0.5f -> range <0.0, 1.0>.
						(*row) = row->unaryExpr ( [&] ( float x ) { return ( x / l2 + 0.5f); } );
					}
				}//: if

				if (iteration%100 == 0) {
					std::shared_ptr<mic::mlnn::Linear<float> > layer = neural_net.getLayer<mic::mlnn::Linear<float> >(0);
					mic::types::MatrixXfPtr W = layer->getParam("W");
					//std::cout<< *W << std::endl;
					// Save nn.
					neural_net.save(fileName);
				}//: if

				iteration++;
				LOG(LINFO) << "Iteration: " << iteration << " loss =" << loss ;
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
	importer = new mic::data_io::MNISTMatrixImporter();
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
	w_input = new WindowGrayscaleBatch("Input batch", 512, 512, 0, 0);
	w_reconstruction = new WindowGrayscaleBatch("Reconstructed batch", 512, 512, 0, 580);
	w_weights = new WindowGrayscaleBatch("L0 weights", 512, 512, 580, 0);

	boost::thread batch_thread(boost::bind(&batch_function));

	// Start visualization thread.
	VGL_MANAGER->startVisualizationLoop();

	LOG(LINFO) << "Waiting for threads to join...";
	// End test thread.
	batch_thread.join();
	LOG(LINFO) << "Threads joined - ending application";
}//: main
