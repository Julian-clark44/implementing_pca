#include <iostream>
#include <complex>
#include <chrono>
#include <cmath>
#include <vector>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace Eigen;

int main() {
	// Record the runtime of the algorithm.
	const auto t1 = std::chrono::steady_clock::now();

	// X is an n x t matrix, where n is the dimensionality of each sample
	// and t is the number of samples.
	Matrix3f X;
	X <<
		1.0, 2.0, 5.0,
		2.0, -2.0, 3.0,
		3.0, -1.8, 4.6;

	// U is the eigenvectors of X * X^{T}
	Matrix3f XXT = X * X.transpose();
	EigenSolver<Matrix3f> eigen_solver(XXT);
	EigenSolver<Matrix3f>::EigenvalueType eigen_values =
		eigen_solver.eigenvalues();
	EigenSolver<Matrix3f>::EigenvectorsType eigen_vectors =
		eigen_solver.eigenvectors();

	// Reduce U to 2 dimensions.
	std::vector<std::pair<float, Vector3f>> eigen_vals_and_vecs;
	for (int i = 0; i < 3; i++) {
		eigen_vals_and_vecs.push_back(
				std::make_pair(eigen_values(i).real(), eigen_vectors.col(i).real()));
	}
	std::sort(eigen_vals_and_vecs.begin(), eigen_vals_and_vecs.end(),
		[&](std::pair<float, Vector3f>& a, std::pair<float, Vector3f>& b) -> bool {
			return a.first > b.first;
		});
	Matrix<float, 3, 2> U;
	U.col(0) = eigen_vals_and_vecs[0].second;
	U.col(1) = eigen_vals_and_vecs[1].second;

	// Compress X to Y.
	Matrix<float, 2, 3> Y;
	Y = U.transpose() * X;

	// Reconstruct X_hat.
	Matrix3f X_hat = U * Y;
	
	const auto t2 = std::chrono::steady_clock::now();
	const std::chrono::duration<double, std::milli> dt = t2 - t1;

	std::cout << "Original X:\n" << X << std::endl;
	std::cout << "X*X^{T} Eigen Vectors (U):\n" << eigen_vectors.real() << std::endl;
	std::cout << "X*X^{T} Eigen Values:\n" << eigen_values.real() << std::endl;
	std::cout << "U_d:\n" << U << std::endl;
	std::cout << "Y:\n" << Y << std::endl;
	std::cout << "Reconstructed X_hat:\n" << X_hat << std::endl;
	std::cout << "X - X_hat:\n" << X - X_hat << std::endl;
	std::cout << "Runtime: " << dt.count() << "ms" << std::endl;
}
