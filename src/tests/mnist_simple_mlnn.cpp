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
* @Date:   2016-02-24 09:43:05
*/

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iomanip>

#include <importers/MNISTMatrixImporter.hpp>
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

#include <mlnn/BackpropagationNeuralNetwork.hpp>

// Using multi layer neural networks
using namespace mic::mlnn;
using namespace mic::types;

int main() {
	// Task parameters.
	size_t batch_size = 20;
	size_t iterations = 60000/batch_size;

	// Set console output.
	LOGGER->addOutput(new ConsoleOutput());
//	LOGGER->setSeverityLevel(LTRACE);

	// Create a simple NN for classification (should give around 95.3% accuracy)
	//MNIST - 28x28 -> 256 -> 100 -> 10
	BackpropagationNeuralNetwork<float> nn("3layerReLUSofmax");
	/*nn.pushLayer(new Linear<float>(28 * 28, 256));
	nn.pushLayer(new ReLU<float>(256));
	nn.pushLayer(new Linear<float>(256, 100));
	nn.pushLayer(new ReLU<float>(100));
	nn.pushLayer(new Linear<float>(100, 10));
	nn.pushLayer(new Softmax<float>(10));
	*/
	nn.pushLayer(new mic::mlnn::convolution::Convolution<float>(28, 28, 1, 20, 14, 7));
	nn.pushLayer(new ReLU<float>(180));
	nn.pushLayer(new Linear<float>(180, 10));
	nn.pushLayer(new Softmax<float>(10));
	nn.verify();

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


	LOG(LSTATUS) << "Starting the training of neural network...";
	float learning_rate = 0.001;
	MatrixXfPtr encoded_batch, encoded_targets;

	// Perform the training.
	for (size_t ii = 0; ii < iterations; ii++) {
		LOG(LINFO) << "Batch " << std::setw(4) << ii << "/" << std::setw(4) << iterations;

		// Get random batch [784 x batch_size].
		MNISTBatch<float> rand_batch = training.getRandomBatch();
		encoded_batch  = mnist_encoder.encodeBatch(rand_batch.data());
		encoded_targets  = label_encoder.encodeBatch(rand_batch.labels());

		// Train network with batch.
		float loss = nn.train (encoded_batch, encoded_targets, learning_rate);
		LOG(LINFO) << "Training: loss = " << std::setprecision(8) << loss;
	}//: for
	LOG(LSTATUS) << "Training finished";

	// Check performance on the test dataset.
	LOG(LSTATUS) << "Calculating performance for test dataset...";
	size_t correct = 0;
	float loss = 0.0;
	test.setNextSampleIndex(0);
	while(!test.isLastBatch()) {

		// Get next batch [784 x batch_size].
		MNISTBatch<float> next_batch = test.getNextBatch();
		encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
		encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

		// Test network response.
		// Skip dropout layers at test time
		nn.forward(encoded_batch, true);
		// Get predictions.
		mic::types::MatrixXfPtr encoded_predictions = nn.getPredictions();
		// Calculate the loss and correct predictions.
		loss += nn.calculateMeanLoss(encoded_targets, encoded_predictions);
		correct += nn.countCorrectPredictions(encoded_targets, encoded_predictions);

	}//: while
	double test_acc = (double)correct / (double)(test.size());
	LOG(LINFO) << "Test  : loss = " << std::setprecision(3) << loss << " correct = " << std::setprecision(3) << 100.0 * test_acc << " %";

	// Check performance on the training dataset.
	LOG(LSTATUS) << "Calculating performance for the training dataset...";
	correct = 0;
	loss = 0;
	training.setNextSampleIndex(0);
	while(!training.isLastBatch()) {

		// Get next batch [784 x batch_size].
		MNISTBatch<float> next_batch = training.getNextBatch();
		encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
		encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

		// Test network response.
		// Skip dropout layers at test time
		nn.forward(encoded_batch, true);
		// Get predictions.
		mic::types::MatrixXfPtr encoded_predictions = nn.getPredictions();
		// Calculate the loss and correct predictions.
		loss += nn.calculateMeanLoss(encoded_targets, encoded_predictions);
		correct += nn.countCorrectPredictions(encoded_targets, encoded_predictions);
	}
	double train_acc = (double)correct / (double)(training.size());
	LOG(LINFO) << "Train  : loss = " << std::setprecision(3) << loss << " correct = " << std::setprecision(3) << 100.0 * train_acc << " %";

}
