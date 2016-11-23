/*!
 * @file: MatrixSpeedTest.cpp
 * @Author: Tomasz Kornuta <tkornut@us.ibm.com>
 * @Date:   Feb 17, 2016
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

#include <types2/Matrix.hpp>

#include <iostream>
#include <chrono>

#ifdef OpenBLAS_FOUND
#include <cblas.h>
#endif

#ifdef OpenBLAS_FOUND
// c = a * b
void BLAS_mmul( mic::types2::Matrix<float>& __restrict c, mic::types2::Matrix<float>& __restrict a,
		mic::types2::Matrix<float>& __restrict b, bool aT = false, bool bT = false) {
	// Both matrices are not transposed - as default.
	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.n_rows;
	size_t N = c.n_cols;
	size_t K = aT ? a.n_rows : a.n_cols;

	float alpha = 1.0f;
	float beta = 0.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	cblas_sgemm( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.memptr(), lda,
				 b.memptr(), ldb, beta, c.memptr(), ldc );


	//	std::cout <<"c=\n" << c <<  std::endl;
}

// returns a * b
mic::types2::Matrix<float> BLAS_mmultilpy(mic::types2::Matrix<float>& __restrict a, mic::types2::Matrix<float>& __restrict b) {
	// Both matrices are not transposed - as default.
	size_t M = a.n_rows;
	size_t K =  a.n_cols;
	size_t N = b.n_cols;

	mic::types2::Matrix<float> c(M,N);

	cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
				 a.memptr(), M,
				 b.memptr(), K, 0.0, c.memptr(), M );

//	std::cout <<"c=\n" << c <<  std::endl;

	return c;
}

#endif /* USE_BLAS */


/*!
 * \brief Program for testing multiplication of matrices with/without OpenBLAS.
 * \author tkornuta
 * @param[in] argc Number of parameters (not used).
 * @param[in] argv List of parameters (not used).
 * @return (not used).
 */
int main(int argc, char* argv[]) {

	// Default sizes of matrices.
	const size_t N = 200;
	const size_t M = 300;
	const size_t K = 400;
	const size_t number_of_multiplications = 100;

 	// Matrices.
	mic::types2::Matrix<float> nm(N, M);
	mic::types2::Matrix<float> mk(M, K);
	mic::types2::Matrix<float> nk(N, K);

	// Initialize matrices with random numbers.
	nm.randn();
	mk.randn();
	//std::cout <<"nm=\n" << nm <<  std::endl;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	// Start!
	start = std::chrono::system_clock::now();
	for (size_t i = 0; i < number_of_multiplications; i++) {
		//std::cout <<  std::setfill('=') << std::setw(80)<< "=\n" <<std::setfill(' ');

		// Multiplication.
//		nk = nm * mk;
#ifdef OpenBLAS_FOUND
		BLAS_mmul(nk, nm, mk);
#else
		nk = nm * mk;
#endif
	}//: for
    end = std::chrono::system_clock::now();
    // End!
    std::chrono::duration<double> elapsed_seconds = end-start;

#ifdef OpenBLAS_FOUND
    std::cout  <<  "OpenBLAS ";
#else
    std::cout  <<  "Armadillo ";
#endif

	std::cout  <<  "multiplication time = " << elapsed_seconds.count() << "s\n";

}//: main


