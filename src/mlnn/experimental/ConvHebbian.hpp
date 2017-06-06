/*!
 * @file ConvHebbian.hpp
 * @brief
 * @Author: Alexis Asseman <alexis.asseman@ibm.com>, Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 30, 2017
 *
 * Copyright (c) 2017, Alexis Asseman, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef SRC_MLNN_CONVHEBBIAN_HPP_
#define SRC_MLNN_CONVHEBBIAN_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace experimental {

/*!
 * \brief Class implementing a convolutional hebbian layer.
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class ConvHebbian : public mic::mlnn::Layer<eT> {
public:

    /*!
     * Creates the convolutional hebbian layer.
     * @param inputs_ Length of the input vector.
     * @param outputs_ Length of the output vector.
     * @param name_ Name of the layer.
     */
    ConvHebbian<eT>(size_t input_l, size_t input_h, size_t channels, size_t nfilters, size_t filter_size, size_t stride = 1, std::string name_ = "ConvHebbian") :
        output_l((input_l / stride) - filter_size),
        output_h((input_h / stride) - filter_size),
        input_l(input_l),
        input_h(input_h),
        channels(channels),
        nfilters(nfilters),
        filter_size(filter_size),
        stride(stride),
        Layer<eT>(input_l * input_h * channels, this->output_l * this->output_h, 1, LayerTypes::ConvHebbian, name_) {

        // Set normalized, zero sum, hebbian learning as default optimization function.
        Layer<eT>::template setOptimization<mic::neural_nets::learning::NormalizedZerosumHebbianRule<eT> > ();

        // Create the weights matrix, each column is a filter kernel
        p.add('W', output_l * output_h, filter_size * filter_size);

        // Initialize weights of all the columns of W.
        for(size_t i = 0 ; i < p['W']->rows() ; i++) {
            p['W']->row(i).rand();
            // Make the matrix Zero Sum
            p['W']->row(i) -= (p['W']->row(i).sum() / p['W']->row(i).cols());
            // Normalize
            p['W']->row(i) /= p['W']->row(i).squaredNorm();
        }
    }


    /*!
     * Virtual destructor - empty.
     */
    virtual ~ConvHebbian() {}

    /*!
     * Forward pass.
     * @param test_ It is set to true in test mode (network verification).
     */
    void forward(bool test_ = false) {
        // Get input matrices.
        mic::types::Matrix<eT> x = (*s['x']);
        mic::types::Matrix<eT> W = (*p['W']);
        // Get output pointer - so the results will be stored!
        mic::types::MatrixPtr<eT> y = s['y'];

        // IM2COL
        mic::types::Matrix<eT> x2col(filter_size * filter_size, output_l * output_h);
        // Iterate over the output matrix (number of image patches)
        for(size_t oy = 0 ; oy < output_h ; oy++){
            for(size_t ox = 0 ; ox < output_l ; ox++){
                // Iterate over the rows of the patch
                for(size_t patch_y = 0 ; patch_y < filter_size ; patch_y++){
                    // Copy each row of the image patch into appropriate position in x2col
                    x2col.block(patch_y * filter_size, ox + (output_l * oy), filter_size, 1) =
                            x.block(ox * stride, oy * stride * filter_size, filter_size, 1);
                }
            }
        }

        // Forward pass.
        (*y) = W * x;
        for (size_t i = 0; i < (size_t)s['x']->rows() * s['x']->cols(); i++) {
            // Sigmoid.
            //(*y)[i] = 1.0f / (1.0f +::exp(-(*y)[i]));
            // Threshold.
            //(*y)[i] = ((*y)[i] > 0.8) ? 1.0f : 0.0f;
            //ReLU
            (*y) = (*y).cwiseMax(0);
        }//: for
    }

    /*!
     * Backward pass.
     */
    void backward() {
        //LOG(LERROR) << "Backward propagation should not be used with layers using Hebbian learning!";
    }

    /*!
     * Applies the gradient update, using the selected hebbian rule.
     * @param alpha_ Learning rate - passed to the optimization functions of all layers.
     * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
     */
    void update(eT alpha_, eT decay_  = 0.0f) {
        opt["W"]->update(p['W'], s['x'], s['y'], alpha_);
    }

    /*!
     * Returns activations of neurons of a given layer (simple visualization).
     */
    std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getActivations(size_t height_, size_t width_) {
        // Check if memory for the activations was allocated.
        if (neuron_activations.size() == 0) {
            for (size_t i=0; i < output_size; i++) {
                // Allocate memory for activation of every neuron.
                mic::types::MatrixPtr<eT> row = MAKE_MATRIX_PTR(eT, input_size, 1);
                neuron_activations.push_back(row);
            }//: for
        }//: if

        // Epsilon added for numerical stability.
        eT eps = 1e-10;

        mic::types::MatrixPtr<eT> W =  p["W"];
        // Iterate through "neurons" and generate "activation image" for each one.
        for (size_t i=0; i < output_size; i++) {
            // Get row.
            mic::types::MatrixPtr<eT> row = neuron_activations[i];
            // Copy data.
            (*row) = W->row(i);
            // Resize row.
            row->resize( height_, width_);
            // Calculate l2 norm.
            eT l2 = row->norm() + eps;
            // Normalize the inputs to <-0.5,0.5> and add 0.5f -> range <0.0, 1.0>.
            (*row) = row->unaryExpr ( [&] ( eT x ) { return ( x / l2 + 0.5f); } );
        }//: for

        // Return activations.
        return neuron_activations;
    }


    // Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
    using Layer<eT>::forward;
    using Layer<eT>::backward;

protected:
    // Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::s;
    using Layer<eT>::m;
    using Layer<eT>::p;
    using Layer<eT>::input_size;
    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

    // Output size (2D)
    size_t output_l = 0;
    size_t output_h = 0;
    size_t input_l = 0;
    size_t input_h = 0;
    size_t channels = 0;
    size_t nfilters = 0;
    size_t filter_size = 0;
    size_t stride = 0;
    // Vector of channels, Each containing a vector of filters
    std::vector<std::vector<mic::types::Matrix<eT> > > W;

private:
    // Friend class - required for using boost serialization.
    template<typename tmp> friend class MultiLayerNeuralNetwork;

    /// Vector containing activations of neurons.
    std::vector< std::shared_ptr <mic::types::MatrixXf> > neuron_activations;

    /*!
     * Private constructor, used only during the serialization.
     */
    ConvHebbian<eT>() : Layer<eT> () { }
};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVHEBBIAN_HPP_ */

