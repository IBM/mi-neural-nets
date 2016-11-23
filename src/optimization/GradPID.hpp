/*!
 * @file: GradPID.hpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Nov 18, 2016
 *
 * Copyright (c) 2016, IBM Corporation. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef GRADPID_HPP_
#define GRADPID_HPP_

#include <optimization/OptimizationFunction.hpp>


/*!
 * \brief GradPID - adaptive gradient descent with proportional, integral and derivative coefficients.
 * \author tkornuta
 */
template <typename eT=float>
class GradPID : public OptimizationFunction<eT> {
public:

	/// Constructor.
	GradPID(size_t dims_, eT learning_rate_=0.01, eT decay_ = 0.9, eT eps_ = 1e-8) : decay(decay_), eps(eps_) {

		Edx = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*Edx)[i] = 0.0;

		dx_prev = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*dx_prev)[i] = 0.0;

		deltaP = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaP)[i] = 0.0;

		deltaI = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaI)[i] = 0.0;

		deltaD = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaD)[i] = 0.0;

		delta = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*delta)[i] = 0.0;

		// Initialize ratios and variables.
		p_rate = learning_rate_ * learning_rate_ * learning_rate_ * learning_rate_ ;
		i_rate = learning_rate_;
		d_rate = learning_rate_ * learning_rate_ * learning_rate_ ;
	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == Edx->size());

		// Update decaying sum of gradients - up to time t.
		for (size_t i=0; i<Edx->size(); i++) {
			(*Edx)[i] = decay *(*Edx)[i] + (1.0 - decay) * (*dx_)[i];
			assert(std::isfinite((*Edx)[i]));
		}

		// DEBUG
/*		for(size_t i=0; i< delta->size(); i++){
			std::cout<< "(*x_)[" << i << "]=" << (*x_)[i] << std::endl;
		}
		for(size_t i=0; i< delta->size(); i++){
			std::cout<< "(*dx_)[" << i << "]=" << (*dx_)[i] << std::endl;
		}*/

		// Calculate gradients updates.
		// Proportional.
		for(size_t i=0; i< delta->size(); i++){
			(*deltaP)[i] = p_rate * (*dx_)[i];
//			std::cout<< "(*deltaP)[" << i << "]=" << (*deltaP)[i] << std::endl;
		}

		// Integral.
		for(size_t i=0; i< delta->size(); i++){
			(*deltaI)[i] = i_rate * (*Edx)[i];
//			std::cout<< "(*deltaI)[" << i << "]=" << (*deltaI)[i] << std::endl;
		}

		// Derivative.
		for(size_t i=0; i< delta->size(); i++){
			(*deltaD)[i] = d_rate * ((*dx_)[i] - (*dx_prev)[i]);
//			std::cout<< "(*deltaD)[" << i << "]=" << (*deltaD)[i] << std::endl;
		}

		// Calculate update.
		for(size_t i=0; i< delta->size(); i++) {
			(*delta)[i] = (*deltaP)[i] + (*deltaI)[i] + (*deltaD)[i];
//			std::cout<< "(*delta)[" << i << "]=" << (*delta)[i] << std::endl;
			assert(std::isfinite((*delta)[i]));
		}

		// Perform the update.
		for (size_t i=0; i<delta->size(); i++) {
			(*x_)[i] -= (*delta)[i];
		}

		// Store past gradients.
		// Perform the update.
		for (size_t i=0; i<dx_->size(); i++) {
			(*dx_prev)[i] = (*dx_)[i];
		}

//		std::cout << std::endl;
	}

protected:

	/// Decay ratio, similar to momentum.
	eT decay;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Adaptive proportional factor (learning rate).
	eT p_rate;

	/// Adaptive integral factor (learning rate).
	eT i_rate;

	/// Adaptive proportional factor (learning rate).
	eT d_rate;

	/// Decaying average of gradients up to time t - E[g].
	mic::types2::MatrixPtr<eT> Edx;

	/// Previous value of gradients.
	mic::types2::MatrixPtr<eT> dx_prev;

	/// Proportional update.
	mic::types2::MatrixPtr<eT> deltaP;

	/// Integral update.
	mic::types2::MatrixPtr<eT> deltaI;

	/// Derivative update.
	mic::types2::MatrixPtr<eT> deltaD;

	/// Calculated update.
	mic::types2::MatrixPtr<eT> delta;
};




/*!
 * \brief AdaGradPID - adaptive gradient descent with proportional, integral and derivative coefficients.
 * \author tkornuta
 */
template <typename eT=float>
class AdaGradPID : public OptimizationFunction<eT> {
public:

	/// Constructor.
	AdaGradPID(size_t dims_, eT learning_rate_=0.01, eT decay_ = 0.9, eT eps_ = 1e-8) : decay(decay_), eps(eps_) {
		Edx = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*Edx)[i] = 0.0;

		dx_prev = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*dx_prev)[i] = 0.0;

		deltaP = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaP)[i] = 0.0;

		deltaI = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaI)[i] = 0.0;

		deltaD = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*deltaD)[i] = 0.0;

		delta = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*delta)[i] = 0.0;

		// Initialize ratios and variables.
		p_rate = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*p_rate)[i] = learning_rate_;

		// Initialize ratios and variables.
		i_rate = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*i_rate)[i] = learning_rate_;

		// Initialize ratios and variables.
		d_rate = MAKE_MATRIX_PTR(eT, dims_, 1);
		for(size_t i=0; i< dims_; i++)
			(*d_rate)[i] = learning_rate_;

	}

	/// Performs update in the direction of gradient descent.
	void update(mic::types2::MatrixPtr<eT> x_, mic::types2::MatrixPtr<eT> dx_) {
		assert(x_->size() == dx_->size());
		assert(x_->size() == Edx->size());

		// Update decaying sum of gradients - up to time t.
/*		for (size_t i=0; i<Edx->size(); i++) {
			(*Edx)[i] = decay *(*Edx)[i] + (1.0 - decay) * (*dx_)[i];
			assert(std::isfinite((*Edx)[i]));
		}*/

		// DEBUG
/*		std::cout << "surprisal = " << surprisal << std::endl;
		std::cout << "p_rate = " << p_rate << std::endl;
		std::cout << "i_rate = " << i_rate << std::endl;
		std::cout << "d_rate = " << d_rate << std::endl;*/
/*		std::cout<< " x: ";
		for(size_t i=0; i< delta->size(); i++)
			std::cout<< (*x_)[i] << " | ";
		std::cout << std::endl;

		std::cout<< " dx: ";
		for(size_t i=0; i< delta->size(); i++)
			std::cout<<  (*dx_)[i] << " | ";
		std::cout << std::endl << std::endl;*/

		// Calculate gradients updates.
		// Proportional.
//		std::cout<< " deltaP: ";
		for(size_t i=0; i< delta->size(); i++){
			// Calculate surprisal.
//			double surp_p = logistic(calculateSigmoidSurprisalMod((*deltaP)[i], (*dx_)[i]), 1.0);
//			(*deltaP)[i] = (1-surp_p) * (*deltaP)[i] + (*p_rate)[i] * surp_p * (*dx_)[i];
//			std::cout<< (*deltaP)[i] << " | ";
		}
//		std::cout << std::endl ;

		// Integral.
//		std::cout<< " deltaI: ";
		for(size_t i=0; i< delta->size(); i++){
//			double surp_i = logistic(calculateSigmoidSurprisalMod((*deltaI)[i], (decay *(*deltaI)[i] + (1.0 - decay) * (*dx_)[i])), 1.0);
//			(*deltaI)[i] = (1-surp_i) * (*deltaI)[i] + (*i_rate)[i] * surp_i * (decay *(*deltaI)[i] + (1.0 - decay) * (*dx_)[i]);
//			std::cout<< (*deltaI)[i] << " | ";
		}
//		std::cout << std::endl ;

		// Derivative.
//		std::cout<< " deltaD: ";
		for(size_t i=0; i< delta->size(); i++){
//			double surp_d = logistic(calculateSigmoidSurprisalMod((*deltaD)[i], ((*dx_)[i] - (*dx_prev)[i])), 1.0);
//			(*deltaD)[i] = (1-surp_d) * (*deltaD)[i] + (*d_rate)[i] * surp_d * ((*dx_)[i] - (*dx_prev)[i]);
//			std::cout<< (*deltaD)[i] << " | ";
		}
//		std::cout << std::endl ;

		// Adaptive rate update.
/*		// Update P ratio.
		eT up;
		std::cout<< " p_rate: ";
		for (size_t i=0; i<delta->size(); i++){
//			up = std::abs((*deltaP)[i] / ((*dx_)[i] + eps)); // softsign
//			std::cout << "up = " << up << std::endl;
//			std::cout << "tanh(up) = " << tanh(up) << std::endl;
			//decay * (*p_rate)[i] + (1 - decay) * tanh(up);//(logistic<eT>(up ,5) / 5);
			std::cout << (*p_rate)[i] << " | ";
		}
		std::cout << std::endl;

		// Update I ratio.
		eT ui;
		std::cout<< " i_rate: ";
		for (size_t i=0; i<delta->size(); i++){
//			ui = std::abs((*deltaI)[i] / ((*dx_)[i] + eps)); // softsign
//			(*i_rate)[i] = decay * (*i_rate)[i] + (1 - decay) * calculateSigmoidSurprisal((*i_rate)[i], ui);
			//decay * (*i_rate)[i] + (1 - decay) * tanh(ui);//(logistic<eT>(ui ,5) / 5);
			std::cout << (*i_rate)[i] << " | ";
		}
		std::cout << std::endl ;

		// Update D ratio.
		eT ud;
/		std::cout<< " d_rate: ";
		for (size_t i=0; i<delta->size(); i++){
//			ud = std::abs((*deltaD)[i] / ((*dx_)[i] + eps)); // softsign
//			(*d_rate)[i] = decay * (*d_rate)[i] + (1 - decay) * calculateSigmoidSurprisal((*d_rate)[i], ud);
			//decay * (*d_rate)[i] + (1 - decay) * tanh(ud);//(logistic<eT>(ud, 5) / 5);
			std::cout << (*d_rate)[i] << " | ";
		}
		std::cout << std::endl ;*/

		// Calculate update.
//		std::cout<< " delta: ";
		for(size_t i=0; i< delta->size(); i++) {
			(*delta)[i] = (*deltaP)[i] + (*deltaI)[i] + (*deltaD)[i];
	//		std::cout<< (*delta)[i] << " | ";
			assert(std::isfinite((*delta)[i]));
		}
	//	std::cout << std::endl << std::endl;

		// Perform the update.
		for (size_t i=0; i<delta->size(); i++) {
			(*x_)[i] -= (*delta)[i];
		}

		// Store past gradiens.
		// Perform the update.
		for (size_t i=0; i<dx_->size(); i++) {
			(*dx_prev)[i] = (*dx_)[i];
		}

//		std::cout << "-------------------" <<  std::endl;
	}

protected:		// Initialize ratios and variables.

	/// Decay ratio, similar to momentum.
	eT decay;

	/// Smoothing term that avoids division by zero.
	eT eps;

	/// Adaptive proportional factor (learning rate).
	mic::types2::MatrixPtr<eT> p_rate;

	/// Adaptive integral factor (learning rate).
	mic::types2::MatrixPtr<eT> i_rate;

	/// Adaptive proportional factor (learning rate).
	mic::types2::MatrixPtr<eT> d_rate;

	/// Surprisal - for feed forward nets it is based on the difference between the prediction and target.
//	eT surprisal;

	/// Decaying average of gradients up to time t - E[g].
	mic::types2::MatrixPtr<eT> Edx;

	/// Previous value of gradients.
	mic::types2::MatrixPtr<eT> dx_prev;

	/// Proportional update.
	mic::types2::MatrixPtr<eT> deltaP;

	/// Integral update.
	mic::types2::MatrixPtr<eT> deltaI;

	/// Derivative update.
	mic::types2::MatrixPtr<eT> deltaD;

	/// Calculated update.
	mic::types2::MatrixPtr<eT> delta;
};




#endif /* GRADPID_HPP_ */
