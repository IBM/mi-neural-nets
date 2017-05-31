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
#include <mlnn/BackpropagationNeuralNetwork.hpp>
using namespace mic::mlnn;

// Encoders.
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

/// Window for displaying the MNIST batch.
WindowGrayscaleBatch* w_input;
WindowGrayscaleBatch* w_reconstruction;
/// Window for displaying the weights.
WindowGrayscaleBatch* w_weights1;
WindowGrayscaleBatch* w_weights2;
WindowGrayscaleBatch* w_weights3;
WindowGrayscaleBatch* w_weights4;
WindowGrayscaleBatch* w_weights5;

/// MNIST importer.
mic::data_io::MNISTMatrixImporter* importer;
/// Multi-layer neural network.
BackpropagationNeuralNetwork<float> neural_net;

/// MNIST matrix encoder.
mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 4;
const size_t output_size = 4608;
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
		//neural_net.pushLayer(new Linear<float>(patch_size*patch_size, patch_size*patch_size));
		//neural_net.pushLayer(new ReLU<float>(patch_size*patch_size));
		//neural_net.setLoss<  mic::neural_nets::loss::SquaredErrorLoss<float> >();
		//neural_net.setOptimization<  mic::neural_nets::optimization::Adam<float> >();

			neural_net.pushLayer(new mic::mlnn::convolution::Convolution<float>(28, 28, 1, 8, 5, 1));
			neural_net.pushLayer(new ReLU<float>(output_size));
			neural_net.pushLayer(new Linear<float>(output_size, 100));
			neural_net.pushLayer(new ReLU<float>(100));
			neural_net.pushLayer(new Linear<float>(100, 10));
			neural_net.pushLayer(new Softmax<float>(10));
			neural_net.verify();
			//neural_net.setLoss<  mic::neural_nets::loss::SquaredErrorLoss<float> >();
			neural_net.setOptimization<  mic::neural_nets::optimization::Momentum<float> >();

		LOG(LINFO) << "Generated new neural network";
	}//: else

	size_t iteration = 0;

	// Retrieve the next minibatch.
	//mic::types::MNISTBatch bt = importer->getNextBatch();

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
				mic::types::MNISTBatch bt = importer->getNextBatch();

				// Set batch to be displayed.
				w_input->setBatchDataUnsynchronized(bt.data());

				// Encode data.
				mic::types::MatrixXfPtr encoded_batch = mnist_encoder->encodeBatch(bt.data());
				mic::types::MatrixXfPtr encoded_labels = label_encoder->encodeBatch(bt.labels());

/*				mic::types::MatrixPtr<float> encoded_batch = MAKE_MATRIX_PTR(float, patch_size*patch_size, 1);
				for (size_t i=0; i<patch_size*patch_size; i++)
					(*encoded_batch)[i]= 1.0 -(float)i/(patch_size*patch_size);*/
				/*mic::types::MatrixPtr<float> encoded_labels = MAKE_MATRIX_PTR(float, output_size, 1);
				encoded_labels->setZero();
				(*encoded_labels)[0]= 1.0;*/
				/*(*encoded_labels)[6]= 1.0;
				(*encoded_labels)[9]= 1.0;
				(*encoded_labels)[15]= 1.0;*/

				// Train the autoencoder.
				float loss = neural_net.train (encoded_batch, encoded_labels, 0.01, 0.001);

				// Get reconstruction.
				/*mic::types::MatrixXfPtr encoded_reconstruction = neural_net.getPredictions();
				std::vector<mic::types::MatrixXfPtr> decoded_reconstruction = mnist_encoder->decodeBatch(encoded_reconstruction);
				w_reconstruction->setBatchDataUnsynchronized(decoded_reconstruction);*/

				if (iteration%10 == 0) {
					// Visualize the weights.
					std::shared_ptr<mic::mlnn::convolution::Convolution<float> > layer1 = neural_net.getLayer<mic::mlnn::convolution::Convolution<float> >(0);
					w_weights1->setBatchDataUnsynchronized(layer1->getWeightActivations());

					w_weights2->setBatchDataUnsynchronized(layer1->getWeightGradientActivations());
					w_weights3->setBatchDataUnsynchronized(layer1->getReceptiveFields(false));
					w_weights4->setBatchDataUnsynchronized(layer1->getOutputActivations(false));
					w_weights5->setBatchDataUnsynchronized(layer1->getOutputGradientActivations());
//					w_weights5->setBatchDataUnsynchronized(layer1->getInverseReceptiveFields(false));

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
	label_encoder = new mic::encoders::UIntMatrixXfEncoder(10);

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
	w_input = new WindowGrayscaleBatch("Input batch", 512, 512, 100, 100);
//	w_reconstruction = new WindowGrayscaleBatch("Reconstructed batch", 512, 512, 0, 580);
	w_weights1 = new WindowGrayscaleBatch("L0 weights", 512, 512, 612, 100);
	w_weights2 = new WindowGrayscaleBatch("L0 dx", 512, 512, 1124, 100);
	w_weights3 = new WindowGrayscaleBatch("L0 receptive fields", 512, 512, 100, 612);
	w_weights4 = new WindowGrayscaleBatch("L0 output activations", 512, 512, 612, 612);
	w_weights5 = new WindowGrayscaleBatch("L0 dy", 512, 512, 1124, 612);
//	w_weights5 = new WindowGrayscaleBatch("L0 inverse receptive fields", 512, 512, 1124, 612);

	boost::thread batch_thread(boost::bind(&batch_function));

	// Start visualization thread.
	VGL_MANAGER->startVisualizationLoop();

	LOG(LINFO) << "Waiting for threads to join...";
	// End test thread.
	batch_thread.join();
	LOG(LINFO) << "Threads joined - ending application";
}//: main
