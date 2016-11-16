/*!
 * \file matrix_test.cpp
 * \brief Test file.
 * \author tkornut
 * \date Nov 17, 2016
 */

#include <armadillo>
#include <iostream>

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

	timer.tic();

    // size and N are specified by the user on the command line
    arma::mat A = arma::randu(N,M);
    arma::mat B = arma::randu(N,M);

    arma::mat Z = arma::zeros(N,M);

    Z = A+B;  //  or Z = A+B+C ... etc

    std::cout << "Z = " << Z << std::endl;


}//: main


