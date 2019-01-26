# Machine Intelligence Core: Neural Nets

![Language](https://img.shields.io/badge/language-C%2B%2B-blue.svg)
[![GitHub license](https://img.shields.io/github/license/IBM/mi-neural-nets.svg)](https://github.com/IBM/mi-neural-nets/blob/master/LICENSE)
![](https://img.shields.io/github/release/IBM/mi-neural-nets.svg)
[![Build Status](https://travis-ci.com/IBM/mi-neural-nets.svg?branch=master)](https://travis-ci.com/IBM/mi-neural-nets)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/IBM/mi-neural-nets.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/mi-neural-nets/context:cpp)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/mi-neural-nets.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/mi-neural-nets/alerts/)

## Description

A subproject of Machine Intelligence Core framework.

The repository contains solutions and applications related to multi-layer (deep) feed-forward (for now) neural nets.

## MIC dependencies

   * MIToolchain - the core of MIC framework.

## External dependencies

Additionally it depends on the following external libraries:
   * Boost - library of free (open source) peer-reviewed portable C++ source libraries.
   * Eigen - a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

### Installation of the dependencies/required tools

On Linux (Ubuntu 14.04): 

    sudo apt-get install git cmake cmake-curses-gui doxygen libboost1.54-all-dev libeigen3-dev

## Applications
   *  mnist_patch_autoencoder_reconstruction -- application realizing MNIST patch autoencoder-based reconstruction
   *  mnist_patch_autoencoder_softmax -- application realizing MNIST patch autoencoder-based softmax classifier, using the imported, previously trained auto-encoder
   *  mlnn_sample_training_test -- (test) application for testing of training of a multi-layer neural network
   *  mlnn_batch_training_test -- (test) application for ttesting batch training of a multi-layer neural network
   *  mnist_convnet -- (test) application using Convolutional Neural Network for recognition of MNIST digits
   *  mnist_simple_mlnn_app -- (test) application using a simple multi-Layer neural net for recognition of MNIST digits
   *  mnist_batch_visualization_test -- the MNIST batch visualization test application
   *  mnist_mlnn_features_visualization_test -- program for visualization of features of mlnn layer trained on MNIST digits

## Unit tests
   *  loss/lossTestsRunner -- loss functions unit tests
   *  optimization/artificialLandscapesTestsRunner -- artificial landscapes used for optimization testing unit tests
   *  optimization/optimizationFunctionsTestsRunner -- unit tests of different optimization functions/methods
   *  mlnn/mlnnTestsRunner -- unit tests for multi-layer neural network
   *  mlnn/cost_function/softmaxTestsRunner -- unit tests of the softmax layer
   *  mlnn/fully_connected/linearTestsRunner -- unit tests for linear (fully-connected) layer

 
## Installation

In order to download, configure, make and install new "clean" version of mi-neural-nets please execute the following:

    cd ~/workspace
    git clone git@github.ibm.com:tkornut/mi-neural-nets.git
    cd mi-algorithms
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=~/workspace/mic/
    make -j4 install

## Documentation

In order to generate a "living" documentation of the code please run Doxygen:

    cd ~/workspace/mi-neural-nets
    doxygen mi-neural-nets.doxyfile
    firefox html/index.html

The current documentation (generated straight from the code and automatically uploaded to github pages by Travis) is available at:

https://ibm.github.io/mi-neural-nets/

Maintainer
----------
tkornuta

