# Machine Intelligence Core: Neural Nets

![Language](https://img.shields.io/badge/language-C%2B%2B-blue.svg)
[![GitHub license](https://img.shields.io/github/license/IBM/mi-neural-nets.svg)](https://github.com/IBM/mi-neural-nets/blob/master/LICENSE)
![](https://img.shields.io/github/release/IBM/mi-neural-nets.svg)
[![Build Status](https://travis-ci.com/IBM/mi-neural-nets.svg?branch=master)](https://travis-ci.com/IBM/mi-neural-nets)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/IBM/mi-neural-nets.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/mi-neural-nets/context:cpp)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/mi-neural-nets.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/mi-neural-nets/alerts/)

## Description

A subproject of Machine Intelligence Core framework.

The repository contains solutions and applications related to (deep) reinforcement learning. In particular, it contains several classical problems (N-armed bandits, several variations of Gridworld), POMDP environments (Gridworld, Maze of Digits, MNIST digit) and algorithms (from simple Value Iteartion and Q-learning to DQN with Experience Replay).

### Applications
   *  mnist_patch_autoencoder_reconstruction -- application realizing MNIST patch autoencoder-based reconstruction
   *  mnist_patch_autoencoder_softmax -- application realizing MNIST patch autoencoder-based softmax classifier, using the imported, previously trained auto-encoder
   *  mlnn_sample_training_test -- (test) application for testing of training of a multi-layer neural network
   *  mlnn_batch_training_test -- (test) application for testing batch training of a multi-layer neural network
   *  mnist_convnet -- (test) application using Convolutional Neural Network for recognition of MNIST digits
   *  mnist_simple_mlnn_app -- (test) application using a simple multi-Layer neural net for recognition of MNIST digits
   *  mnist_batch_visualization_test -- the MNIST batch visualization test application
   *  mnist_mlnn_features_visualization_test -- program for visualization of features of mlnn layer trained on MNIST digits

### Unit tests
   *  loss/lossTestsRunner -- loss functions unit tests
   *  optimization/artificialLandscapesTestsRunner -- artificial landscapes used for optimization testing unit tests
   *  optimization/optimizationFunctionsTestsRunner -- unit tests of different optimization functions/methods
   *  mlnn/mlnnTestsRunner -- unit tests for multi-layer neural network
   *  mlnn/cost_function/softmaxTestsRunner -- unit tests of the softmax layer
   *  mlnn/fully_connected/linearTestsRunner -- unit tests for linear (fully-connected) layer

 
## External dependencies

Additionally it depends on the following external libraries:
   * Boost - library of free (open source) peer-reviewed portable C++ source libraries.
   * Eigen - a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
   * OpenGL/GLUT - a cross-language, cross-platform application programming interface for rendering 2D and 3D vector graphics.
   * OpenBlas (optional) - An optimized library implementing BLAS routines. If present - used for fastening operation on matrices.
   * Doxygen (optional) - Tool for generation of documentation.
   * GTest (optional) - Framework for unit testing.

### Installation of the dependencies/required tools

#### On Linux (Ubuntu 14.04):

    sudo apt-get install git cmake doxygen libboost1.54-all-dev libeigen3-dev freeglut3-dev libxmu-dev libxi-dev

To install GTest on Ubuntu:

    sudo apt-get install libgtest-dev

#### On Mac (OS X 10.14): (last tested on: Feb/01/2019)

    brew install git cmake doxygen boost eigen glfw3

To install GTest on Mac OS X:

    brew install --HEAD https://gist.githubusercontent.com/Kronuz/96ac10fbd8472eb1e7566d740c4034f8/raw/gtest.rb

## MIC dependencies

   * [MI-Toolchain](https://github.com/IBM/mi-toolchain) - the core of MIC framework.
   * [MI-Algorithms](https://github.com/IBM/mi-algorithms) - contains basic (core) types and algorithms.
   * [MI-Visualization](https://github.com/IBM/mi-visualization) - contains OpenGL-based visualization.

### Installation of all MIC dependencies (optional)

This step is required only when not downloaded/installed the listed MIC dependencies earlier.

In directory scripts one can find script that will download and install all required MIC modules.

    git clone git@github.com:IBM/mi-neural-nets.git
    cd mi-neural-nets
    ./scripts/install_mic_deps.sh ../mic

Then one can install the module by calling the following.

    ./scripts/build_mic_module.sh ../mic

Please note that it will create a directory 'deps' and download all sources into that directory.
After compilation all dependencies will be installed in the directory '../mic'.

## Installation of MI-Neural-Nets
The following assumes that all MIC dependencies are installed in the directory '../mic'.

    git clone git@github.com:IBM/mi-neural-nets.git
    cd mi-neural-nets
    ./scripts/build_mic_module.sh ../mic

### Make commands

   * make install - install applications to ../mic/bin, headers to ../mic/include, libraries to ../mic/lib, cmake files to ../mic/share
   * make configs - install config files to ../mic/bin
   * make datasets - install config files to ../mic/datasets

## Documentation

In order to generate a "living" documentation of the code please run Doxygen:

    cd mi-neural-nets
    doxygen mi-neural-nets.doxyfile
    firefox html/index.html

The current documentation (generated straight from the code and automatically uploaded to github pages by Travis) is available at:

https://ibm.github.io/mi-neural-nets/

[tkornuta](http://github.com/tkornut)

[![HitCount](http://hits.dwyl.io/tkornut/ibm/mi-neural-nets.svg)](http://hits.dwyl.io/tkornut/ibm/mi-neural-nets)

