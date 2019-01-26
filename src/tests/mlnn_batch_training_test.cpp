/*!
 * \file mlnn_batch_training_test.cpp
 * \brief Contains program for testing batch training of a multi-layer neural network.
 * \author tkornut
 * \date Feb 17, 2016
 */

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iostream>

#include <mlnn/BackpropagationNeuralNetwork.hpp>

#include <encoders/MatrixXfMatrixXfEncoder.hpp>
#include <encoders/UIntMatrixXfEncoder.hpp>

#include <types/Batch.hpp>


// Using multi-layer neural networks
using namespace mic::mlnn;
using namespace mic::types;

int main() {
	// Set console output.
	LOGGER->addOutput(new ConsoleOutput());

	// Generate a dataset.
	size_t dataset_size = 15;
	size_t classes = 5;
	Batch<MatrixXf, MatrixXf> dataset;
	for(size_t i=0; i< dataset_size; i++) {
		// Generate "pose".
		MatrixXfPtr pose (new MatrixXf(dataset_size, 1));
		pose->setZero();
		(*pose)(i,0)=1;
		dataset.data().push_back(pose);

		// Generate desired target.
		MatrixXfPtr target (new MatrixXf(classes, 1));
		target->setZero();
		(*target)(i%classes,0)= 1;//(i%4);
		dataset.labels().push_back(target);

		// Add index.
		dataset.indices().push_back(i);
	}//: for

	// Neural net.
	BackpropagationNeuralNetwork<float> nn("simple_linear_network");
	nn.pushLayer(new Linear<float>(dataset_size, 15, "Linear1"));
	nn.pushLayer(new ReLU<float>(15, "ReLU1"));
	nn.pushLayer(new Linear<float>(15, classes, "Linear2"));
	nn.pushLayer(new Softmax<float>(classes, "Softmax"));
	// Change optimization function from default GradientDescent to Adam.
	nn.setOptimization<mic::neural_nets::optimization::Adam<float> >();

	// Initialize the encoders.
	mic::encoders::MatrixXfMatrixXfEncoder data_encoder(dataset_size, 1);
	mic::encoders::MatrixXfMatrixXfEncoder label_encoder(classes, 1);

	// Training.
	size_t iteration = 0;
	while (iteration < 10000) {
		Batch <MatrixXf, MatrixXf> batch = dataset.getRandomBatch();
		//std::cout << "[" << iteration << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		MatrixXfPtr encoded_batch, encoded_targets;
		encoded_batch  = data_encoder.encodeBatch(batch.data());
		encoded_targets  = label_encoder.encodeBatch(batch.labels());

		// Train network with batch.
		float loss = nn.train (encoded_batch, encoded_targets, 0.1);

		if (iteration % 1000 == 0){
			std::cout<<"[" << iteration << "]: Loss        : " << loss << std::endl;
		}

		// Compare results
		// MatrixXf predictions = (*nn.getPredictions());
		//std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		//std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;
		iteration++;
	}//: while

	// Test network
	iteration = 0;
	dataset.setNextSampleIndex(0);
	while (iteration < dataset_size) {
		Sample <MatrixXf, MatrixXf> sample = dataset.getNextSample();
		std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = nn.test(sample.data(), sample.label());
		// Compare results
		MatrixXf predictions = (*nn.getPredictions());
		std::cout<<"Loss        : " << loss << std::endl;
		std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;

	}//: while

}
