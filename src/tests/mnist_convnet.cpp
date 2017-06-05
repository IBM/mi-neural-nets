/*
* @Author: kmrocki/tkornuta
* @Date:   2016-03-10 09:43:05
*/

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iomanip>

#include <data_io/MNISTMatrixImporter.hpp>
#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

#include <mlnn/BackpropagationNeuralNetwork.hpp>

using namespace mic::types;
// Using multi layer neural networks
using namespace mic::mlnn;
using namespace mic::mlnn::convolution;

int main() {
	// Task parameters.
//	size_t iterations = 500;
	size_t 	epochs = 500;
	size_t 	batch_size = 16;

	// Set console output.
	ConsoleOutput* co = new ConsoleOutput();
	LOGGER->addOutput(co);

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

	// Create a convolutional neural network.
	// 20 Epochs
	// Train : 99.99 %
	// Test  : 99.61 %

	size_t 	input_channels = 1;
	size_t 	filter_size[] = {3, 3, 3, 3, 3};
	size_t 	filters[] = {32, 32, 128, 128, 256};

	size_t 	pooling_window = 2;
	float 	dropout = 0.5f;

	size_t  fully_connected_size = 256;

	// Neural net.
	BackpropagationNeuralNetwork<float> nn("ConvNet");

	//CONV 3x3 -> CONV 3x3 -> POOL 2x
/*	nn.pushLayer(new Convolution<float>(28*28, input_channels, filter_size[0], filters[0]));
	nn.pushLayer(new ReLU<float>(nn.lastLayerOutputsSize()));
	nn.pushLayer(new Convolution<float>(nn.lastLayerOutputsSize() / filters[0], filters[0], filter_size[1], filters[1]));
	nn.pushLayer(new ReLU<float>(nn.lastLayerOutputsSize()));
	nn.pushLayer(new Pooling<float>(nn.lastLayerOutputsSize(), pooling_window, filters[2]));

	//CONV 3x3 -> CONV 3x3 -> POOL 2x
	nn.pushLayer(new Convolution<float>(nn.lastLayerOutputsSize() / filters[1], filters[1], filter_size[2], filters[2]));
	nn.pushLayer(new ReLU<float>(nn.lastLayerOutputsSize()));
	nn.pushLayer(new Convolution<float>(nn.lastLayerOutputsSize() / filters[2], filters[2], filter_size[3], filters[3]));
	nn.pushLayer(new ReLU<float>(nn.lastLayerOutputsSize()));
	nn.pushLayer(new Pooling<float>(nn.lastLayerOutputsSize(), pooling_window, filters[3]));

	//CONV 3x3 -> POOL 2x
	nn.pushLayer(new Convolution<float>(nn.lastLayerOutputsSize() / filters[3], filters[3], filter_size[4], filters[4]));
	nn.pushLayer(new Pooling<float>(nn.lastLayerOutputsSize(), pooling_window, filters[4]));

	//FULLY CONNECTED
	nn.pushLayer(new Linear<float>(nn.lastLayerOutputsSize(), fully_connected_size));
	nn.pushLayer(new ReLU<float>(nn.lastLayerOutputsSize()));
	nn.pushLayer(new Dropout<float>(nn.lastLayerOutputsSize(), dropout));

	//SOFTMAX
	nn.pushLayer(new Linear<float>(nn.lastLayerOutputsSize(), 10));
	nn.pushLayer(new Softmax<float>(10));*/

	// Set batch size.
	nn.resizeBatch(batch_size);

	// Change optimization function from default GradientDescent to Adam.
	nn.setOptimization<mic::neural_nets::optimization::Adam<float> >();

	// Set training parameters.
	double 	learning_rate = 1e-2;
	double 	weight_decay = 0;
	size_t iterations = training.size() / batch_size;

	MatrixXfPtr encoded_batch, encoded_targets;
	// For all epochs.
	for (size_t e = 0; e < epochs; e++) {
		LOG(LSTATUS) << "Epoch " << e + 1 << ": starting the training of neural network...";
		// Perform the training.
		for (size_t ii = 0; ii < iterations; ii++) {
			std::cout<< "[" << std::setw(4) << ii << "/" << std::setw(4) << iterations << "] ";

			// Get random batch [784 x batch_size].
			MNISTBatch rand_batch = training.getRandomBatch();
			encoded_batch  = mnist_encoder.encodeBatch(rand_batch.data());
			encoded_targets  = label_encoder.encodeBatch(rand_batch.labels());

			// Train network with batch.
			nn.train (encoded_batch, encoded_targets, learning_rate, weight_decay);

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
			MNISTBatch next_batch = test.getNextBatch();
			encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
			encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

			// Test network response.
			correct += nn.test(encoded_batch, encoded_targets);

		}
		double test_acc = (double)correct / (double)(test.size());
		LOG(LINFO) << "Test  : " << std::setprecision(3) << 100.0 * test_acc << " %";

		// Check performance on the training dataset.
		LOG(LSTATUS) << "Calculating performance for the training dataset...";
		correct = 0;
		training.setNextSampleIndex(0);
		while(!training.isLastBatch()) {

			// Get next batch [784 x batch_size].
			MNISTBatch next_batch = training.getNextBatch();
			encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
			encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

			// Test network response.
			correct += nn.test(encoded_batch, encoded_targets);

		}
		double train_acc = (double)correct / (double)(training.size());
		LOG(LINFO) << "Train : " << std::setprecision(3) << 100.0 * train_acc << " %";

	}//: for epoch

}
