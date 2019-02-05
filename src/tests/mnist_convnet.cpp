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
/*
* @Author: kmrocki/tkornuta
* @Date:   2016-03-10 09:43:05
*/

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iomanip>

#include <importers/MNISTMatrixImporter.hpp>
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

#include <mlnn/BackpropagationNeuralNetwork.hpp>

using namespace mic::types;
// Using multi layer neural networks
using namespace mic::mlnn;
using namespace mic::mlnn::convolution;

int main() {
	// Task parameters.
	size_t 	epochs = 100;
	size_t 	batch_size = 1;

	// Set console output.
	ConsoleOutput* co = new ConsoleOutput();
	LOGGER->addOutput(co);

	//[60000, 784]
	// Load the MNIST training...
    mic::importers::MNISTMatrixImporter<float> training;
	// Manually set paths. DEPRICATED! Used here only for simplification of the test.
	training.setDataFilename("../data/mnist/train-images.idx3-ubyte");
	training.setLabelsFilename("../data/mnist/train-labels.idx1-ubyte");
	training.setBatchSize(batch_size);

	if (!training.importData())
		return -1;

	// ... and test datasets.
    mic::importers::MNISTMatrixImporter<float> test;
	// Manually set paths. DEPRICATED! Used here only for simplification of the test.
	test.setDataFilename("../data/mnist/t10k-images.idx3-ubyte");
	test.setLabelsFilename("../data/mnist/t10k-labels.idx1-ubyte");
	test.setBatchSize(batch_size);

	if (!test.importData())
		return -1;

	// Initialize the encoders.
	mic::encoders::MatrixXfMatrixXfEncoder mnist_encoder(28, 28);
	mic::encoders::UIntMatrixXfEncoder label_encoder(10);

	// Create a convolutional neural network.
	// 20 Epochs
	// Train : 99.99 %
	// Test  : 99.61 %

	// Neural net.
	BackpropagationNeuralNetwork<float> nn("ConvNet");

	// Convolution 1
	nn.pushLayer(new mic::mlnn::convolution::Cropping<float>(28, 28, 1, 1));
	nn.pushLayer(new mic::mlnn::convolution::Convolution<float>(26, 26, 1, 16, 3, 1));
	nn.pushLayer(new ELU<float>(24, 24, 16));
	nn.pushLayer(new mic::mlnn::convolution::MaxPooling<float>(24, 24, 16, 2));

	// Convolution 2
	nn.pushLayer(new mic::mlnn::convolution::Convolution<float>(12, 12, 16, 32, 3, 1));
	nn.pushLayer(new ELU<float>(10, 10, 32));
	nn.pushLayer(new mic::mlnn::convolution::MaxPooling<float>(10, 10, 32, 2));

	// Linear + dropout
	nn.pushLayer(new Linear<float>(5, 5, 32, 100, 1, 1));
	nn.pushLayer(new ELU<float>(100, 1, 1));
	nn.pushLayer(new Dropout<float>(100, 0.5f));

	// Softmax
	nn.pushLayer(new Linear<float>(100, 10));
	nn.pushLayer(new Softmax<float>(10));
	if (!nn.verify())
		exit(-1);

	// Set batch size.
	nn.resizeBatch(batch_size);

	// Change optimization function from default GradientDescent to Adam.
	nn.setOptimization<mic::neural_nets::optimization::Adam<float> >();

	// Set training parameters.
	double 	learning_rate = 1e-4;
	double 	weight_decay = 1e-5;
	size_t iterations = training.size() / batch_size;

	MatrixXfPtr encoded_batch, encoded_targets;
	// For all epochs.
	for (size_t e = 0; e < epochs; e++) {
		LOG(LSTATUS) << "Epoch " << e + 1 << ": starting the training of neural network...";
		// Perform the training.
		for (size_t ii = 0; ii < iterations; ii++) {
			std::cout<< "[" << std::setw(4) << ii << "/" << std::setw(4) << iterations << "] ";

			// Get random batch [784 x batch_size].
            MNISTBatch<float> rand_batch = training.getRandomBatch();
			encoded_batch  = mnist_encoder.encodeBatch(rand_batch.data());
			encoded_targets  = label_encoder.encodeBatch(rand_batch.labels());

			// Train network with batch.
			float loss = nn.train (encoded_batch, encoded_targets, learning_rate, weight_decay);
			std::cout << " loss = " << loss << std::endl;
		}//: for iteration

		// Save results to file.
		nn.save("mnist_conv");

		LOG(LSTATUS) << "Training finished";

		// Check performance on the test dataset.
		LOG(LSTATUS) << "Calculating performance for test dataset...";
		size_t correct = 0;
		test.setNextSampleIndex(0);
		while(!test.isLastBatch()) {

			// Get next batch [784 x batch_size].
            MNISTBatch<float> next_batch = test.getNextBatch();
			encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
			encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

			// Test network response.
			correct += nn.test(encoded_batch, encoded_targets);

		}
		double test_acc = (double)correct / (double)(test.size());
		LOG(LINFO) << "Test accuracy  : " << std::setprecision(3) << 100.0 * test_acc << " %";

		// Check performance on the training dataset.
		LOG(LSTATUS) << "Calculating performance for the training dataset...";
		correct = 0;
		training.setNextSampleIndex(0);
		while(!training.isLastBatch()) {

			// Get next batch [784 x batch_size].
            MNISTBatch<float> next_batch = training.getNextBatch();
			encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
			encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

			// Test network response.
			correct += nn.test(encoded_batch, encoded_targets);

		}
		double train_acc = (double)correct / (double)(training.size());
		LOG(LINFO) << "Trainin accuracy : " << std::setprecision(3) << 100.0 * train_acc << " %";

	}//: for epoch

}
