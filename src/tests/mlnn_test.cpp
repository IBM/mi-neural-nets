/*!
 * \file mlnn_test.cpp
 * \brief Contains program for testing creation/serialization of multi-layer neural networks.
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

	// Create 1 layer (linear) network.
	MultiLayerNeuralNetwork nn("simple_linear_network");
	nn.pushLayer(new Linear(10, 20, "First Linear"));
	nn.pushLayer(new ReLU(20, "ReLU"));
	nn.pushLayer(new Linear(20, 4, "Second Linear"));
	nn.pushLayer(new ReLU(4, "ReLU"));
	nn.pushLayer(new Regression(4, "Final Regression"));

/*	// Generate sample.
	MatrixXfPtr sample (new MatrixXf(36, 1));
	(*sample) << 0,   0,   0,   0,   0,   0,   0,  10,   0,   0, -10,   0, -10,   0, -10,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0;
	std::cout<<"Sample: " << sample->transpose() << std::endl;

	// Forward the sample and get prediction - used later as target.
	nn.forward(*sample);
	MatrixXfPtr targets (new MatrixXf(*nn.getPredictions()));

	// Train several times for the same state and rewards.
	for(size_t i=0; i< 10; i++) {
		float loss = nn.train(sample, targets, 0.005, 0.0);
		// Compare results
		MatrixXf predictions = (*nn.getPredictions());
		std::cout<<"Loss        : " << loss << std::endl;
		std::cout<<"Targets     : " << targets->transpose() << std::endl;
		std::cout<<"Predictions : " << predictions.transpose() << std::endl;
	}//: for*/


	// Generate a batch.
	Batch<MatrixXf, MatrixXf> batch;
	for(size_t i=0; i< 10; i++) {
		// Generate "pose".
		MatrixXfPtr pose (new MatrixXf(10, 1));
		pose->setZero();
		(*pose)(i,0)=1;
		batch.data().push_back(pose);

		// Generate desired action .
		MatrixXfPtr action (new MatrixXf(4, 1));
		action->setZero();
//		(*action)(i%4,0)= 1;
		(*action)(i%4,0)= 10*((i%3) - 1.0);
		batch.labels().push_back(action);

		// Add index.
		batch.indices().push_back(i);
	}//: for

	// Training.
	size_t iteration = 0;
	while (iteration < 100000) {
		Sample <MatrixXf, MatrixXf> sample = batch.getRandomSample();
		//std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = nn.train(sample.data(), sample.label(), 0.005, 0.0);

		/*std::shared_ptr<Linear> layer1 = nn.getLayer<Linear>	(0);
		std::cout<<"Layer1: " << *layer1  << std::endl;
		std::shared_ptr<ReLU> layer2 = nn.getLayer<ReLU>	(1);
		std::cout<<"Layer2: " << *layer2  << std::endl;

		std::cout << (*layer1->getParam("W"))  <<std::endl;*/

		// Compare results
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
		Sample <MatrixXf, MatrixXf> sample = batch.getRandomSample();
		std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = nn.test(sample.data(), sample.label());
		// Compare results
		MatrixXf predictions = (*nn.getPredictions());
		std::cout<<"Loss        : " << loss << std::endl;
		std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;

	}//: while

	// Save network to file.
	const char* fileName = "saved.txt";
	nn.save(fileName);
	std::cout << "Saved network: \n" << nn;


	// Load network from file.
	MultiLayerNeuralNetwork restored_nn("simple_linear_network_loaded");
	restored_nn.load(fileName);
	std::cout << "Restored network: \n" << restored_nn;


	// Test network
	iteration = 0;
	while (iteration < 10) {
		Sample <MatrixXf, MatrixXf> sample = batch.getRandomSample();
		std::cout << "[" << iteration++ << "]: sample (" << sample.index() << "): "<< sample.data()->transpose() << "->" << sample.label()->transpose() << std::endl;

		float loss = restored_nn.test(sample.data(), sample.label());
		// Compare results
		MatrixXf predictions = (*restored_nn.getPredictions());
		std::cout<<"Loss        : " << loss << std::endl;
		std::cout<<"Targets     : " << sample.label()->transpose() << std::endl;
		std::cout<<"Predictions : " << predictions.transpose() << std::endl << std::endl;

	}//: while

}
