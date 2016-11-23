/*!
 * \file matrix_test.cpp
 * \brief Test file.
 * \author tkornut
 * \date Nov 17, 2016
 */

#include "types2/Matrix.hpp"

#include <iostream>
#include <chrono>

/*!
 * \brief Program for testing multiplication of matrices with/without OpenBLAS.
 * \author tkornuta
 * @param[in] argc Number of parameters (not used).
 * @param[in] argv List of parameters (not used).
 * @return (not used).
 */
int main(int argc, char* argv[]) {

	// Default sizes of matrices.
	const size_t N = 2;
	const size_t M = 3;
	const size_t K = 4;
	const size_t number_of_multiplications = 100;

    // size and N are specified by the user on the command line
    mic::types2::Matrix<double> A = arma::randu(N,M);
    mic::types2::Matrix<double> B = arma::randu(N,M);

    mic::types2::Matrix<double> Z = arma::zeros(N,M);

    Z = A+B;  //  or Z = A+B+C ... etc

    std::cout << "Z = " << Z << std::endl;

	// Matrices.
	mic::types2::Matrix<float> nm(N, M);
	mic::types2::Matrix<float> mk(M, K);
	mic::types2::Matrix<float> nk(N, K);
	mic::types2::Matrix<float> nm_zero = arma::zeros<mic::types2::Matrix<float> >(N,M);

	// Initialize matrices with random numbers.
	nm.randn();
	mk.randn();
	//std::cout <<"nm=\n" << nm <<  std::endl;

	// Initialize vector.
/*	mic::types2::VectorXf vect(N);
	vect.setRandom();

	// Examples of mapping from vectors to matrices.
	mic::types2::Matrix<float> vect_copy = vect;
	mic::types2::Matrix<float> vect_copy2;
	vect_copy2 = (mic::types2::Matrix<float>)vect;
*/
/*	arma::rowvec vect(N);
	// Adding test.
	vect_copy2 += vect;
	std::cout <<"vect_copy=\n" << vect_copy <<  std::endl;
	std::cout <<"vect_copy2=\n" << vect_copy2 <<  std::endl;

	std::cout <<"vect=\n" << vect <<  std::endl;
	nm.colwise() += vect;
	std::cout <<"nm+=vect\n" << nm <<  std::endl;*/



	// Elementwise function test.
	mic::types2::Matrix<float> nm2 (N, M);
	nm2.randn();
	std::cout <<"nm2=\n" << nm2 <<  std::endl;
	nm += nm2;
	std::cout <<"nm+=nm2\n" << nm <<  std::endl;

	//std::cout <<"mk=\n" << mk <<  std::endl;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
	for (size_t i = 0; i < number_of_multiplications; i++) {
		//std::cout <<  std::setfill('=') << std::setw(80)<< "=\n" <<std::setfill(' ');

		// Multiplication.
		nk = nm * mk;

/*#ifdef OpenBLAS_FOUND
		BLAS_mmul(nk, nm, mk);
		//nk = BLAS_mmultilpy(nm, mk);
		//std::cout <<"blas nk=\n" << nk <<  std::endl;
#else
		nk = nm * mk;
		//std::cout <<"eigen nk=\n" << nk <<  std::endl;
#endif*/
	}//: for
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

	//std::cout <<"nk=\n" << nk <<  std::endl;

	std::cout  <<  "Multiplication time = " << elapsed_seconds.count() << "s\n";


	const char* fileName = "saved.txt";

	// Save data
	{
		// Create an output archive
		std::ofstream ofs(fileName);
		boost::archive::text_oarchive ar(ofs);
		// Write data
		ar & nm;
		std::cout << "Saved matrix = " << nm << std::endl;

	}

	// Restore data
	mic::types2::Matrix<float> restored_mat;

	{
		// Create and input archive
		std::ifstream ifs(fileName);
		boost::archive::text_iarchive ar(ifs);
		// Load data
		ar & restored_mat;
		std::cout << "Restored matrix = " << restored_mat << std::endl;
	}

	// Matrix pointer.
	mic::types2::MatrixPtr<double> mat_ptr (new mic::types2::Matrix<double>(N, M));
	mat_ptr->randn();
	std::cout << "Pointer  = " << *mat_ptr << std::endl;


}//: main


