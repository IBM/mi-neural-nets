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

#include <mlnn/HebbianNeuralNetwork.hpp>
#include <mlnn/experimental/ConvHebbian.hpp>

using namespace mic::types;
// Using multi layer neural networks
using namespace mic::mlnn;
using namespace mic::mlnn::convolution;
using namespace mic::mlnn::experimental;

int main() {
	// Task parameters.
//	size_t iterations = 500;
    size_t 	samples = 2000;
    size_t 	batch_size = 1;

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

	size_t 	input_channels = 1;
    size_t 	filter_size[] = {7};
    size_t 	filters[] = {20};

	// Neural net.
    HebbianNeuralNetwork<float> nn("HebbianFilters");
    nn.pushLayer(new ConvHebbian<float>(28, 28, input_channels, filters[0], filter_size[0], 1));

	// Set batch size.
	nn.resizeBatch(batch_size);

	// Set training parameters.
	double 	learning_rate = 1e-2;
	double 	weight_decay = 0;
	size_t iterations = training.size() / batch_size;

	MatrixXfPtr encoded_batch, encoded_targets;
	// For all epochs.
    for(size_t i = 0 ; i < samples ; i++){
        LOG(LSTATUS) << "Sample " << i + 1;

        MNISTBatch next_batch = test.getNextBatch();
        encoded_batch  = mnist_encoder.encodeBatch(next_batch.data());
        encoded_targets  = label_encoder.encodeBatch(next_batch.labels());

        nn.train(encoded_batch, learning_rate);
    }

}
