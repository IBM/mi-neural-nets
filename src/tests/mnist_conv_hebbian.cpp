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

#include <importers/MNISTMatrixImporter.hpp>

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <application/ApplicationState.hpp>

#include <configuration/ParameterServer.hpp>

#include <opengl/visualization/WindowManager.hpp>
#include <opengl/visualization/WindowGrayscaleBatch.hpp>
#include <opengl/visualization/WindowCollectorChart.hpp>

using namespace mic::opengl::visualization;

// Hebbian neural net.
#include <mlnn/HebbianNeuralNetwork.hpp>
using namespace mic::mlnn;

// Encoders.
#include <encoders/ColMatrixEncoder.hpp>
#include <encoders/UIntMatrixEncoder.hpp>

#include <mlnn/experimental/ConvHebbian.hpp>
using namespace mic::mlnn::experimental;

/// Window for displaying the MNIST batch.
WindowGrayscaleBatch<double>* w_input;
/// Window for displaying the weights.
WindowGrayscaleBatch<double>* w_weights1;

WindowGrayscaleBatch<double>* w_output;
WindowGrayscaleBatch<double>* w_reconstruction;
WindowGrayscaleBatch<double>* w_similarity;

/// Data collector.
WindowCollectorChart<double>* w_chart;
mic::utils::DataCollectorPtr<std::string, double> collector_ptr;

/// MNIST importer.
mic::importers::MNISTMatrixImporter<double>* importer;
/// Multi-layer neural network.
HebbianNeuralNetwork<double> neural_net;

/// MNIST matrix encoder.
mic::encoders::ColMatrixEncoder<double>* mnist_encoder;
/// Label 2 matrix encoder (1 hot).
//mic::encoders::UIntMatrixXfEncoder* label_encoder;

const size_t patch_size = 28;
const size_t batch_size = 1;
const size_t input_channels = 1;
const size_t filter_size[] = {5};
const size_t filters[] = {16};
const size_t stride[] = {1};


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
        neural_net.pushLayer(new ConvHebbian<double>(patch_size, patch_size, input_channels, filters[0], filter_size[0], stride[0]));

        LOG(LINFO) << "Generated new neural network";
    }//: else

    std::shared_ptr<mic::mlnn::experimental::ConvHebbian<double> > layer1 =
            neural_net.getLayer<mic::mlnn::experimental::ConvHebbian<double> >(0);

    size_t iteration = 0;
    // Set training parameters.
    const double learning_rate = 5e-3;

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
                mic::types::MNISTBatch<double> bt = importer->getRandomBatch();

                // Encode data.
                mic::types::MatrixPtr<double> encoded_batch = mnist_encoder->encodeBatch(bt.data());

                MNISTBatch<double> next_batch = importer->getNextBatch();
                encoded_batch  = mnist_encoder->encodeBatch(next_batch.data());

                neural_net.train(encoded_batch, learning_rate);

                if (iteration % 10 == 0) {
                    //Visualize the weights.
                    // Set batch to be displayed.
                    w_input->setBatchUnsynchronized(layer1->getInputActivations());
                    w_weights1->setBatchUnsynchronized(layer1->getWeightActivations());
                    w_similarity->setBatchUnsynchronized(layer1->getWeightSimilarity(true));
                    w_output->setBatchUnsynchronized(layer1->getOutputActivations());
                    w_reconstruction->setBatchUnsynchronized(layer1->getOutputReconstruction());
                    collector_ptr->addDataToContainer("Reconstruction error", layer1->getOutputReconstructionError());
                    LOG(LINFO) << "Iteration: " << iteration;
                }//: if

                iteration++;
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
    importer = new mic::importers::MNISTMatrixImporter<double>();
    importer->setDataFilename("../data/mnist/train-images.idx3-ubyte");
    importer->setLabelsFilename("../data/mnist/train-labels.idx1-ubyte");
    importer->setBatchSize(batch_size);

    // Initialize the encoders.
    mnist_encoder = new mic::encoders::ColMatrixEncoder<double>(patch_size, patch_size);
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
    w_input = new WindowGrayscaleBatch<double>("Input batch", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70, 0, 250, 250);
    w_weights1 = new WindowGrayscaleBatch<double>("Permanences", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70+250, 0, 250, 250);
    w_similarity = new WindowGrayscaleBatch<double>("Cosine similarity matrix", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70+(2*250), 0, 250, 250);
    w_output = new WindowGrayscaleBatch<double>("Output", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70+(3*250), 0, 250, 250);
    w_reconstruction = new WindowGrayscaleBatch<double>("Reconstruction", Grayscale::Norm_HotCold, Grayscale::Grid_Both, 70+(4*250), 0, 250, 250);

    // Chart.
    w_chart = new WindowCollectorChart<double>("Statistics", 60, 878, 512, 256);
    collector_ptr= std::make_shared < mic::utils::DataCollector<std::string, double> >( );
    w_chart->setDataCollectorPtr(collector_ptr);

    // Create data containers.
    collector_ptr->createContainer("Reconstruction error", mic::types::color_rgba(255, 255, 255, 180));

    boost::thread batch_thread(boost::bind(&batch_function));

    // Start visualization thread.
    VGL_MANAGER->startVisualizationLoop();

    LOG(LINFO) << "Waiting for threads to join...";
    // End test thread.
    batch_thread.join();
    LOG(LINFO) << "Threads joined - ending application";
}//: main
