/*!
 * Copyright (C) tkornuta, IBM Corporation 2015-2019
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * @file: NormalizedZerosumHebbianRule.hpp
 * @Author: Alexis Asseman <alexis.asseman@ibm.com>, Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   May 30, 2017
 *
 * Copyright (c) 2017, Tomasz Kornuta, IBM Corporation. All rights reserved.
 *
 */

#ifndef NORMALIZEDZEROSUMHEBBIANRULE_HPP_
#define NORMALIZEDZEROSUMHEBBIANRULE_HPP_

#include <optimization/OptimizationFunction.hpp>
#include <random>

namespace mic {
namespace neural_nets {
namespace learning {

/*!
 * \brief Updates according to a modified Hebbian rule (wij += ni * f(x, y)) with additional normalization and zero summing for optimal edge detection
 *
 * \author tkornuta/Alexis-Asseman
 */
template <typename eT=float>
class NormalizedZerosumHebbianRule : public mic::neural_nets::optimization::OptimizationFunction<eT> {
public:
    /*!
     * Constructor. Sets dimensions.
     * @param rows_ Number of rows of the update matrix.
     * @param cols_ Number of columns of the update matrix.
     */
    NormalizedZerosumHebbianRule(size_t rows_, size_t cols_) {
        delta = MAKE_MATRIX_PTR(eT, rows_, cols_);
        delta->zeros();
    }

    // Virtual destructor - empty.
    virtual ~NormalizedZerosumHebbianRule() { }


    /*!
     * Updates the weight matrix according to the hebbian rule with normalization (l2 norm).
     * @param p_ Pointer to the parameter (weight) matrix.
     * @param x_ Pointer to the input data matrix.
     * @param y_ Pointer to the output data matrix.
     * @param learning_rate_ Learning rate (default=0.001).
     */
    virtual void update(mic::types::MatrixPtr<eT> p_, mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_ = 0.001) {
        assert(p_->rows() == y_->rows());
        assert(p_->cols() == x_->rows());
        assert(x_->cols() == y_->cols());

        // Calculate the update using hebbian "fire together, wire together".
        mic::types::MatrixPtr<eT> delta = calculateUpdate(x_, y_, learning_rate_);

        // weight += delta;
        (*p_) += (*delta);
        // Eigen doesn't check for div by 0 (the doc lies...)
        for(auto i = 0 ; i < p_->rows() ; i++){
            if(p_->row(i).norm() != 0){
                p_->row(i) = p_->row(i).normalized();
            }
        }
    }

    /*!
     * Calculates the update according to the hebbian rule.
     * @param x_ Pointer to the input data matrix.
     * @param y_ Pointer to the output data matrix.
     * @param learning_rate_ Learning rate (default=0.001).
     */
    virtual mic::types::MatrixPtr<eT> calculateUpdate(mic::types::MatrixPtr<eT> x_, mic::types::MatrixPtr<eT> y_, eT learning_rate_) {
        // delta based on winner take all: Best corresponding kernel gets to learn for each slice
        // Winner take all happens for each column of the output matrix, between the rows of the kernels matrix

        // Iterate over the output columns
        delta->zeros();
        typename mic::types::Matrix<eT>::Index argmax, argmin;

        //Randomize access to the indices of image patches
        std::vector<typename mic::types::Matrix<eT>::Index> shuffled_indices;
        for(auto i = 0 ; i < y_->cols() ; i++) shuffled_indices.push_back(i);
        std::random_shuffle(std::begin(shuffled_indices), std::end(shuffled_indices));

        for(auto i: shuffled_indices){
            y_->col(i).maxCoeff(&argmax);
            y_->col(i).minCoeff(&argmin);
            //argmax = am(y_->col(i));
            if(argmin != argmax){ // If all filters respond equally, then do nothing about this input patch
                // Pick the image slice and apply it to best matching filter (ie: row of p['W'])
                delta->row(argmax) = x_->col(i);
                // Make the vector zero-sum
                delta->row(argmax).array() -= delta->row(argmax).sum() / delta->cols();
                // Eigen doesn't check for div by 0 (the doc lies...)
                if(delta->row(argmax).norm() != 0){
                    delta->row(argmax) = delta->row(argmax).normalized();
                }
            }

        }
        (*delta) *= learning_rate_;
        return delta;
    }


protected:
    /// Calculated update.
    mic::types::MatrixPtr<eT> delta;
};

} //: namespace learning
} /* namespace neural_nets */
} /* namespace mic */

#endif /* NORMALIZEDZEROSUMHEBBIANRULE_HPP_ */
