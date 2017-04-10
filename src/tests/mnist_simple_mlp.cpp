/*
* @Author: kmrocki
* @Date:   2016-02-24 09:43:05
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-02-24 15:30:27
*/

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iomanip>

#include <data_io/MNISTMatrixImporter.hpp>
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

#include <mlnn/MultiLayerNeuralNetwork.hpp>

// Using multi layer neural networks
using namespace mic::mlnn;
using namespace mic::types;

int main() {
	// Task parameters.
	size_t batch_size = 100;
	size_t iterations = 60000/batch_size;

	// Set console output.
	LOGGER->addOutput(new ConsoleOutput());


	//[60000, 784]
	// Load the MNIST training...
	mic::data_io::MNISTMatrixImporter training;
	// Manually set paths. DEPRICATED! Used here only for simplification of the test.
	training.setDataFilename("../data/mnist/train-images.idx3-ubyte");
	training.setLabelsFilename("../data/mnist/train-labels.idx1-ubyte");
	training.setBatchSize(batch_size);

	if (!training.importData())
		return -1;

	// ... and test datasets.
	mic::data_io::MNISTMatrixImporter test;
	// Manually set paths. DEPRICATED! Used here only for simplification of the test.
	test.setDataFilename("../data/mnist/t10k-images.idx3-ubyte");
	test.setLabelsFilename("../data/mnist/t10k-labels.idx1-ubyte");
	test.setBatchSize(batch_size);

	if (!test.importData())
		return -1;

	// Initialize the encoders.
	mic::encoders::MatrixXfMatrixXfEncoder mnist_encoder(28, 28);
	mic::encoders::UIntMatrixXfEncoder label_encoder(10);

	// Create a simple NN for classification (should give around 95.3% accuracy)
	//MNIST - 28x28 -> 256 -> 100 -> 10
	MultiLayerNeuralNetwork nn;
	nn.pushLayer(new Linear(28 * 28, 256, batch_size));
	nn.pushLayer(new ReLU(256, 256, batch_size));
	nn.pushLayer(new Linear(256, 100, batch_size));
	nn.pushLayer(new ReLU(100, 100, batch_size));
	nn.pushLayer(new Linear(100, 10, batch_size));
	nn.pushLayer(new Softmax(10, 10, batch_size));


	// LOG(LSTATUS) << "Before training..." << std::endl;
	// nn.test(test_data);

	LOG(LSTATUS) << "Starting the training of neural network...";
	double 	learning_rate = 0.005;
	double 	weight_decay = 0;
	MatrixXfPtr encoded_batch, encoded_targets;

	// Perform the training.
	for (size_t ii = 0; ii < iterations; ii++) {
		LOG(LINFO) << "Batch " << std::setw(4) << ii << "/" << std::setw(4) << iterations;

		// Get random batch [784 x batch_size].
		MNISTBatch rand_batch = training.getRandomBatch();
		encoded_batch  = mnist_encoder.encodeBatch(rand_batch.data());
		encoded_targets  = label_encoder.encodeBatch(rand_batch.labels());

		// Train network with batch.
		nn.train (encoded_batch, encoded_targets, learning_rate, weight_decay);

	}//: for
	LOG(LSTATUS) << "Training finished";

	// Check performance on the test dataset.
	LOG(LSTATUS) << "Calculating performance for test dataset...";
	size_t correct = 0;
	float loss = 0.0;
	test.setNextSampleIndex(0);
	while(!test.isLastBatch()) {

		// Get next batch [784 x batch_size].
		MNISTBatch next_batch = test.getNextBatch();
		encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
		encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

		// Test network response.
		// Skip dropout layers at test time
		nn.forward(*encoded_batch, true);
		// Get predictions.
		mic::types::MatrixXfPtr encoded_predictions = nn.getPredictions();
		// Calculate the loss and correct predictions.
		loss += nn.calculateLossFunction(encoded_targets, encoded_predictions);
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
		MNISTBatch next_batch = training.getNextBatch();
		encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
		encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

		// Test network response.
		// Skip dropout layers at test time
		nn.forward(*encoded_batch, true);
		// Get predictions.
		mic::types::MatrixXfPtr encoded_predictions = nn.getPredictions();
		// Calculate the loss and correct predictions.
		loss += nn.calculateLossFunction(encoded_targets, encoded_predictions);
		correct += nn.countCorrectPredictions(encoded_targets, encoded_predictions);
	}
	double train_acc = (double)correct / (double)(training.size());
	LOG(LINFO) << "Train  : loss = " << std::setprecision(3) << loss << " correct = " << std::setprecision(3) << 100.0 * train_acc << " %";

}
