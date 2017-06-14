/*!
 * @file mnist_conv_hebbian.cpp
 * @brief Program for visualization of hebbian formed filters trained on MNIST digits.
 * @Author: Alexis Asseman <alexis.asseman@ibm.com>, Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   June 8, 2017
 *
 * Copyright (c) 2017, Alexis Asseman, Tomasz Kornuta, IBM Corporation. All rights reserved.
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

#include <mlnn/experimental/ConvHebbian.hpp>
using namespace mic::mlnn::experimental;

/// Window for displaying the MNIST batch.
WindowGrayscaleBatch<float>* w_input;
/// Window for displaying the weights.
WindowGrayscaleBatch<float>* w_weights1;

/// MNIST importer.
mic::data_io::MNISTMatrixImporter* importer;
/// Multi-layer neural network.
HebbianNeuralNetwork<float> neural_net;

/// MNIST matrix encoder.
mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
//mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 1;
const size_t input_channels = 1;
const size_t filter_size[] = {7};
const size_t filters[] = {20};


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
        neural_net.pushLayer(new ConvHebbian<float>(patch_size, patch_size, input_channels, filters[0], filter_size[0], 1));

        LOG(LINFO) << "Generated new neural network";
    }//: else

    size_t iteration = 0;
    // Set training parameters.
    const float learning_rate = 0.1;
    const float weight_decay = 0.0;
    const size_t iterations = importer->size() / batch_size;
    const size_t samples = 2000;


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

                MNISTBatch next_batch = importer->getNextBatch();
                encoded_batch  = mnist_encoder->encodeBatch(next_batch.data());

                neural_net.train(encoded_batch, learning_rate);

                //if (iteration % 10 == 0) {
                    //Visualize the weights.
                    std::shared_ptr<mic::mlnn::experimental::ConvHebbian<float> > layer1 =
                            neural_net.getLayer<mic::mlnn::experimental::ConvHebbian<float> >(0);
                    w_weights1->setBatchDataUnsynchronized(layer1->getWeightActivations());
                //}//: if

                iteration++;
                LOG(LINFO) << "Iteration: " << iteration;
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
    importer->setDataFilename("../data/mnist/train-images.idx3-ubyte");
    importer->setLabelsFilename("../data/mnist/train-labels.idx1-ubyte");
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
    w_weights1 = new WindowGrayscaleBatch<float>("Permanences", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 320, 0, 250, 250);

    boost::thread batch_thread(boost::bind(&batch_function));

    // Start visualization thread.
    VGL_MANAGER->startVisualizationLoop();

    LOG(LINFO) << "Waiting for threads to join...";
    // End test thread.
    batch_thread.join();
    LOG(LINFO) << "Threads joined - ending application";
}//: main
