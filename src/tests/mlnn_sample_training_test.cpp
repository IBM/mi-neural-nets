/*!
 * \file mlnn_sample_training_test.cpp
 * \brief Contains program for testing of training of a multi-layer neural network.
 * \author tkornut
 * \date Feb 17, 2016
 */

#include <logger/Log.hpp>
#include <logger/ConsoleOutput.hpp>
using namespace mic::logger;

#include <iostream>

#include <mlnn/MultiLayerNeuralNetwork.hpp>

#include <types/Batch.hpp>



// Using multi-layer neural networks
using namespace mic::mlnn;
using namespace mic::types;

int main() {
	// Set console output.
	LOGGER->addOutput(new ConsoleOutput());

	MultiLayerNeuralNetwork<float> nn("simple_linear_network");
	nn.pushLayer(new Linear<float>(10, 20, "First Linear"));
	nn.pushLayer(new ReLU<float>(20, "ReLU"));
	nn.pushLayer(new Linear<float>(20, 4, "Second Linear"));
	nn.pushLayer(new ReLU<float>(4, "ReLU"));
	nn.pushLayer(new Regression<float>(4, "Final Regression"));

	// Generate a dataset.
	size_t dataset_size = 10;
	Batch<MatrixXf, MatrixXf> dataset;
	for(size_t i=0; i< dataset_size; i++) {
		// Generate "pose".
		MatrixXfPtr pose (new MatrixXf(dataset_size, 1));
		pose->setZero();
		(*pose)(i,0)=1;
		dataset.data().push_back(pose);

		// Generate desired target.
		MatrixXfPtr target (new MatrixXf(4, 1));
		target->setZero();
		(*target)(i%4,0)= (i%4);
		dataset.labels().push_back(target);

		// Add index.
		dataset.indices().push_back(i);
	}//: for

	// Training.
	size_t iteration = 0;
	while (iteration < 100000) {
		Sample <MatrixXf, MatrixXf> sample = dataset.getRandomSample();
		//std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = nn.train(sample.data(), sample.label(), 0.005, 0.0);

		// Compare results.
		MatrixXf predictions = (*nn.getPredictions());
		if (iteration % 1000 == 0){
			std::cout<<"[" << iteration << "]: Loss        : " << loss << std::endl;
		}
		//std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		//std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;
		iteration++;
	}//: while

	// Test network
	iteration = 0;
	while (iteration < 10) {
		Sample <MatrixXf, MatrixXf> sample = dataset.getRandomSample();
		std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = nn.test(sample.data(), sample.label());
		// Compare results
		MatrixXf predictions = (*nn.getPredictions());
		std::cout<<"Loss        : " << loss << std::endl;
		std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;

	}//: while

}
