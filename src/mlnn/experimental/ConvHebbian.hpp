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
    ConvHebbian<eT>(size_t input_width, size_t input_height, size_t input_depth, size_t nfilters, size_t filter_size, size_t stride = 1, std::string name_ = "ConvHebbian") :
        Layer<eT>(input_height, input_width, input_depth, (input_height / stride) - filter_size, (input_width / stride) - filter_size, 1, LayerTypes::ConvHebbian, name_),
        nfilters(nfilters),
        filter_size(filter_size),
        stride(stride),
        x2col(new mic::types::Matrix<eT>(filter_size * filter_size, output_width * output_height))
    {

        // Create the weights matrix, each row is a filter kernel
        p.add("W", nfilters, filter_size * filter_size);
        mic::types::MatrixPtr<eT> W = p["W"];

        // Set normalized, zero sum, hebbian learning as default optimization function.
        Layer<eT>::template setOptimization<mic::neural_nets::learning::NormalizedZerosumHebbianRule<eT> > ();

        // Initialize weights of all the columns of W.
        W->rand();
        for(auto i = 0 ; i < W->rows() ; i++) {
            // Make the matrix Zero Sum
            W->row(i).array() -= W->row(i).sum() / W->row(i).cols();
            if(W->row(i).norm() != 0){
                W->row(i) = W->row(i).normalized();
            }
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
        mic::types::Matrix<eT> x = (*s["x"]);
        mic::types::Matrix<eT> W = (*p["W"]);
        // Get output pointer - so the results will be stored!
        mic::types::MatrixPtr<eT> y = s["y"];

        // IM2COL
        // Iterate over the output matrix (number of image patches)
        for(size_t oy = 0 ; oy < output_height ; oy++){
            for(size_t ox = 0 ; ox < output_width ; ox++){
                // Iterate over the rows of the patch
                for(size_t patch_y = 0 ; patch_y < filter_size ; patch_y++){
                    // Copy each row of the image patch into appropriate position in x2col
                    x2col->block(patch_y * filter_size, ox + (output_width * oy), filter_size, 1) =
                            x.block((oy * stride + patch_y) * input_width + ox * stride, 0, filter_size, 1);
                }
            }
        }
        // Forward pass.
        (*y) = W * (*x2col);
        // ReLU
        //(*y) = (*y).cwiseMax(0);
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
        opt["W"]->update(p["W"], x2col, s["y"], alpha_);
    }



    /*!
     * Returns activations of weights.
     */
    std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getOutputActivations(bool normalize_ = true) {

        // Allocate memory.
        lazyAllocateMatrixVector(o_activations, nfilters, output_height * output_width, 1);

        mic::types::MatrixPtr<eT> W = s["y"];

        // Iterate through "neurons" and generate "activation image" for each one.
        for (size_t i = 0 ; i < nfilters ; i++) {
            // Get row.
            mic::types::MatrixPtr<eT> row = o_activations[i];
            // Copy data.
            (*row) = W->row(i);
            // Resize row.
            row->resize(output_width, output_height);

            // Normalize.
            if (normalize_)
                normalizeMatrixForVisualization(row);
        }//: for filters

        // Return activations.
        return o_activations;
    }

    /*!
     * Returns activations of weights.
     */
    std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getWeightActivations(bool normalize_ = true) {

        // Allocate memory.
        lazyAllocateMatrixVector(w_activations, nfilters, filter_size*filter_size, 1);

        mic::types::MatrixPtr<eT> W = p["W"];

        // Iterate through "neurons" and generate "activation image" for each one.
        for (size_t i = 0 ; i < nfilters ; i++) {
            // Get row.
            mic::types::MatrixPtr<eT> row = w_activations[i];
            // Copy data.
            (*row) = W->row(i);
            // Resize row.
            row->resize(filter_size, filter_size);

            // Normalize.
            if (normalize_)
                normalizeMatrixForVisualization(row);
        }//: for filters

        // Return activations.
        return w_activations;
    }


    // Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
    using Layer<eT>::forward;
    using Layer<eT>::backward;

protected:
    // Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::s;
    using Layer<eT>::m;
    using Layer<eT>::p;
    using Layer<eT>::opt;

    // Uncover "sizes" for visualization.
    using Layer<eT>::input_height;
    using Layer<eT>::input_width;
    using Layer<eT>::input_depth;
    using Layer<eT>::output_height;
    using Layer<eT>::output_width;
    using Layer<eT>::output_depth;
    using Layer<eT>::batch_size;

    // Uncover methods useful in visualization.
    using Layer<eT>::lazyAllocateMatrixVector;
    using Layer<eT>::normalizeMatrixForVisualization;

    size_t nfilters = 0;
    size_t filter_size = 0;
    size_t stride = 0;
    // Vector of channels, Each containing a vector of filters
    std::vector<std::vector<mic::types::Matrix<eT> > > W;
    mic::types::MatrixPtr<eT> x2col;

private:
    // Friend class - required for using boost serialization.
    template<typename tmp> friend class MultiLayerNeuralNetwork;

    /// Vector containing activations of neurons.
    std::vector< std::shared_ptr <mic::types::Matrix<eT> > > w_activations;
    std::vector< std::shared_ptr <mic::types::Matrix<eT> > > o_activations;

    /*!
     * Private constructor, used only during the serialization.
     */
    ConvHebbian<eT>() : Layer<eT> () { }
};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVHEBBIAN_HPP_ */
