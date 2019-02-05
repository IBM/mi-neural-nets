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
 * @file: MNISTPatchReconstructionApplication.h
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   27-03-2017
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 */

#ifndef MNISTPATCHRECONSTRUCTIONAPPLICATION_H_
#define MNISTPATCHRECONSTRUCTIONAPPLICATION_H_


#include <opengl/application/OpenGLContinuousLearningApplication.hpp>

#include <opengl/visualization/WindowMatrix2D.hpp>
#include <opengl/visualization/WindowCollectorChart.hpp>
using namespace mic::opengl::visualization;

#include <importers/MNISTPatchImporter.hpp>

#include <mlnn/BackpropagationNeuralNetwork.hpp>
using namespace mic::mlnn;

namespace mic {
namespace applications {

/*!
 * \brief Class implementing a simple MNIST patch reconstruction with multi-layer neural net.
 * \author tkornuta
 */
class MNISTPatchReconstructionApplication: public mic::opengl::application::OpenGLContinuousLearningApplication {
public:
	/*!
	 * Default Constructor. Sets the application/node name, default values of variables, initializes classifier etc.
	 * @param node_name_ Name of the application/node (in configuration file).
	 */
	MNISTPatchReconstructionApplication(std::string node_name_ = "mnist_patch_autoencoder_reconstruction");

	/*!
	 * Destructor.
	 */
	virtual ~MNISTPatchReconstructionApplication();

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
	/// Input image/matrix.
	mic::types::MatrixXfPtr input_image;

	/// Reconstructed image/matrix.
	mic::types::MatrixXfPtr reconstructed_image;

	/// Importer responsible for loading training dataset.
	mic::importers::MNISTPatchImporter* training_dataset_importer;

	/// Importer responsible for loading testing dataset.
	mic::importers::MNISTPatchImporter* test_dataset_importer;

	/// Window for displaying the input image.
	WindowMatrix2D* w2d_input;

	/// Window for displaying the reconstructed image.
	WindowMatrix2D* w2d_reconstruction;

	/// Window for displaying chart with statistics.
	WindowCollectorChart<float>* w_chart;

	/// Size of the patch - copied from importers.
	size_t patch_size;

	/// Data collector.
	mic::utils::DataCollectorPtr<std::string, float> collector_ptr;

	/// Property: name of the file to which the neural network will be serialized (or deserialized from).
	mic::configuration::Property<std::string> mlnn_filename;

	/// Property: flag denoting whether the nn should be saved to a file (after every episode end).
	mic::configuration::Property<bool> mlnn_save;

	/// Property: flag denoting whether the nn should be loaded from a file (at the initialization of the task).
	mic::configuration::Property<bool> mlnn_load;

	/// Multi-layer neural network.
	BackpropagationNeuralNetwork<float> neural_net;

};

} /* namespace applications */
} /* namespace mic */

#endif /* MNISTPATCHRECONSTRUCTIONAPPLICATION_H_ */
