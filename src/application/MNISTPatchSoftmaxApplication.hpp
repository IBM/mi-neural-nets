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
 * @file: MNISTPatchSoftmaxApplication.h
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   27-03-2017
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef MNISTPATCHSOFTMAXAPPLICATION_H_
#define MNISTPATCHSOFTMAXAPPLICATION_H_


#include <opengl/application/OpenGLContinuousLearningApplication.hpp>

#include <opengl/visualization/WindowMatrix2D.hpp>
#include <opengl/visualization/WindowProbability.hpp>
#include <opengl/visualization/WindowCollectorChart.hpp>
using namespace mic::opengl::visualization;

#include <data_io/MNISTPatchImporter.hpp>

#include <mlnn/BackpropagationNeuralNetwork.hpp>
using namespace mic::mlnn;

#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>


namespace mic {
namespace applications {

/*!
 * \brief Class implementing a simple MNIST patch softmax classification with multi-layer neural net - imported from previously loaded auto-encoder net and adds softmax layer "at the top".
 * \author tkornuta
 */
class MNISTPatchSoftmaxApplication: public mic::opengl::application::OpenGLContinuousLearningApplication {
public:
	/*!
	 * Default Constructor. Sets the application/node name, default values of variables, initializes classifier etc.
	 * @param node_name_ Name of the application/node (in configuration file).
	 */
	MNISTPatchSoftmaxApplication(std::string node_name_ = "mnist_patch_autoencoder_reconstruction");

	/*!
	 * Destructor.
	 */
	virtual ~MNISTPatchSoftmaxApplication();

protected:
	/*!
	 * Initializes all variables that are property-dependent, imports MNIST datasets, for both learning and testing purposes.
	 */
	virtual void initializePropertyDependentVariables();

	/*!
	 * Method initializes GLUT and OpenGL windows.
	 * @param argc Number of application parameters.
	 * @param argv Array of application parameters.
	 */
	virtual void initialize(int argc, char* argv[]);

	/*!
	 * Performs learning step.
	 */
	virtual bool performLearningStep();

	/*!
	 * Collects test statistics, executed in every testing step.
	 */
	virtual void collectTestStatistics();

	/*!
	 * Populates test statistics, executed only every number_of_averaged_test_measures testing steps.
	 */
	virtual void populateTestStatistics();


private:
	/// Importer responsible for loading training dataset.
	mic::data_io::MNISTPatchImporter* training_dataset_importer;

	/// Importer responsible for loading testing dataset.
	mic::data_io::MNISTPatchImporter* test_dataset_importer;

	/// MNIST matrix encoder.
	mic::encoders::MatrixXfMatrixXfEncoder* mnist_encoder;

	/// Label 2 matrix encoder (1 hot).
	mic::encoders::UIntMatrixXfEncoder* label_encoder;

	/// Window for displaying the input image.
	WindowMatrix2D* w2d_input;

	/// Window for displaying chart with statistics.
	WindowCollectorChart<float>* w_chart;

	/// Window displaying the probabilities.
	WindowProbability* w_prob;

	/// Input image/matrix.
	mic::types::MatrixXfPtr input_image;

	/// Probability.
	mic::types::MatrixXfPtr decoded_prediction;

	/// Probability.
	mic::types::MatrixXfPtr input_target;

	/// Size of the patch - copied from importers.
	size_t patch_size;

	/// Data collector.
	mic::data_io::DataCollectorPtr<std::string, float> collector_ptr;

	/// Property: name of the file to which the original auto-encoder will be loaded from.
	mic::configuration::Property<std::string> autoencoder_filename;

	/// Property: number of layers to be removed (starting from the last one).
	mic::configuration::Property<size_t> autoencoder_layers_to_be_removed;

	/// Property: name of the file with softmax nn.
	mic::configuration::Property<std::string> softmax_filename;

	/// Property: flag denoting whether the nn should be saved to a file (after every episode end).
	mic::configuration::Property<bool> softmax_save;

	/// Property: flag denoting whether the nn should be loaded from a file (at the initialization of the task) - if not, the program will try to import and reshape the autoencoder.
	mic::configuration::Property<bool> softmax_load;

	/// Multi-layer neural network.
	BackpropagationNeuralNetwork<float> neural_net;

};

} /* namespace applications */
} /* namespace mic */

#endif /* MNISTPATCHSOFTMAXAPPLICATION_H_ */
