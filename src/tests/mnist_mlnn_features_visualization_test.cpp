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

/// Windows for displaying activations.
WindowGrayscaleBatch *w_conv10, *w_conv11, *w_conv12, *w_conv13;
WindowGrayscaleBatch *w_conv20, *w_conv21, *w_conv22, *w_conv23;
WindowGrayscaleBatch *w_conv30, *w_conv31;


/// MNIST importer.
mic::data_io::MNISTMatrixImporter* importer;
/// Multi-layer neural network.
BackpropagationNeuralNetwork<float> neural_net;

/// MNIST matrix encoder.
mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 9;
const size_t output_size = 3872;
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

/*			neural_net.pushLayer(new mic::mlnn::convolution::Convolution<float>(28, 28, 1, 20, 7, 1));
			neural_net.pushLayer(new ReLU<float>(22*22*20));
/*			neural_net.pushLayer(new mic::mlnn::convolution::Convolution<float>(22, 22, 8, 20, 5, 1));
			neural_net.pushLayer(new ReLU<float>(6480));
			neural_net.pushLayer(new Linear<float>(6480, 100));
			neural_net.pushLayer(new ReLU<float>(100));*/
			neural_net.pushLayer(new Linear<float>(28*28, 28*28));
//			neural_net.pushLayer(new ReLU<float>(28*28));
//			neural_net.pushLayer(new Softmax<float>(10));
			if (!neural_net.verify())
				exit(-1);


			neural_net.setLoss<  mic::neural_nets::loss::SquaredErrorLoss<float> >();
			neural_net.setOptimization<  mic::neural_nets::optimization::GradientDescent<float> >();

		LOG(LINFO) << "Generated new neural network";
	}//: else

	// Import data from datasets.
	if (!importer->importData())
		exit(-1);


	size_t iteration = 0;

	// Retrieve the next minibatch.
	//mic::types::MNISTBatch bt = importer->getNextBatch();
	//importer->setNextSampleIndex(7);

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
				//w_input->setBatchDataUnsynchronized(bt.data());

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
				float loss = neural_net.train (encoded_batch, encoded_batch, 0.01, 0.001);

				// Get reconstruction.
				/*mic::types::MatrixXfPtr encoded_reconstruction = neural_net.getPredictions();
				std::vector<mic::types::MatrixXfPtr> decoded_reconstruction = mnist_encoder->decodeBatch(encoded_reconstruction);
				w_reconstruction->setBatchDataUnsynchronized(decoded_reconstruction);*/

				{//if (iteration%10 == 0) {
					// Visualize the weights.
					std::shared_ptr<mic::mlnn::convolution::Convolution<float> > conv1 = neural_net.getLayer<mic::mlnn::convolution::Convolution<float> >(0);
					//std::shared_ptr<mic::mlnn::convolution::Convolution<float> > conv2 = neural_net.getLayer<mic::mlnn::convolution::Convolution<float> >(2);
					w_conv10->setBatchDataUnsynchronized(conv1->getInputActivations(false));
					w_conv11->setBatchDataUnsynchronized(conv1->getInputGradientActivations());
					w_conv12->setBatchDataUnsynchronized(conv1->getWeightActivations());
					w_conv13->setBatchDataUnsynchronized(conv1->getWeightGradientActivations());

					w_conv20->setBatchDataUnsynchronized(conv1->getOutputActivations());
					w_conv21->setBatchDataUnsynchronized(conv1->getOutputGradientActivations());
					//w_weights3->setBatchDataUnsynchronized(layer1->getReceptiveFields(false));
					//w_weights5->setBatchDataUnsynchronized(layer1->getInverseReceptiveFields(false));

					/*w_conv20->setBatchDataUnsynchronized(conv2->getInputActivations());
					w_conv21->setBatchDataUnsynchronized(conv2->getInputGradientActivations());
					w_conv22->setBatchDataUnsynchronized(conv2->getWeightActivations());
					w_conv23->setBatchDataUnsynchronized(conv2->getWeightGradientActivations());

					w_conv30->setBatchDataUnsynchronized(conv2->getOutputActivations());
					w_conv31->setBatchDataUnsynchronized(conv2->getOutputGradientActivations());*/

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

	// Initialize GLUT! :]
	VGL_MANAGER->initializeGLUT(argc, argv);

	// Create batch visualization window.
	w_conv10 = new WindowGrayscaleBatch("L0 x", 256, 256, 50, 50);
	w_conv11 = new WindowGrayscaleBatch("L0 dx", 256, 256, 316, 50);
	w_conv12 = new WindowGrayscaleBatch("L0 W", 256, 256, 562, 50);
	w_conv13 = new WindowGrayscaleBatch("L0 dW", 256, 256, 818, 50);

	w_conv20 = new WindowGrayscaleBatch("L1 x", 256, 256, 50, 336);
	w_conv21 = new WindowGrayscaleBatch("L1 dx", 256, 256, 316, 336);
	w_conv22 = new WindowGrayscaleBatch("L1 W", 256, 256, 562, 336);
	w_conv23 = new WindowGrayscaleBatch("L1 dW", 256, 256, 818, 336);

	w_conv30 = new WindowGrayscaleBatch("L2 x", 256, 256, 50, 622);
	w_conv31 = new WindowGrayscaleBatch("L2 dx", 256, 256, 312, 622);

//	w_weights5 = new WindowGrayscaleBatch("L0 inverse receptive fields", 512, 512, 1124, 612);

	boost::thread batch_thread(boost::bind(&batch_function));

	// Start visualization thread.
	VGL_MANAGER->startVisualizationLoop();

	LOG(LINFO) << "Waiting for threads to join...";
	// End test thread.
	batch_thread.join();
	LOG(LINFO) << "Threads joined - ending application";
}//: main
