/*!
 * \file LayerTypes.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_LAYERTYPES_HPP_
#define SRC_MLNN_LAYERTYPES_HPP_

// Activation functions - implemented as layers.

#include <mlnn/activation_function/ELU.hpp>

#include <mlnn/activation_function/ReLU.hpp>

#include <mlnn/activation_function/Sigmoid.hpp>

// Convolution-related layers.

#include <mlnn/convolution/Convolution.hpp>

#include <mlnn/convolution/Cropping.hpp>

#include <mlnn/convolution/MaxPooling.hpp>

#include <mlnn/convolution/Padding.hpp>


// Cost functions - implemented as layers.

#include <mlnn/cost_function/Softmax.hpp>

// Fully connected layers.

#include <mlnn/fully_connected/Linear.hpp>

#include <mlnn/fully_connected/SparseLinear.hpp>

// Regularisation layers.

#include <mlnn/regularisation/Dropout.hpp>

#include <mlnn/fully_connected/HebbianLinear.hpp>

#include <mlnn/fully_connected/BinaryCorrelator.hpp>

// Experimental layers

#include <mlnn/experimental/ConvHebbian.hpp>

#endif /* SRC_MLNN_LAYERTYPES_HPP_ */
