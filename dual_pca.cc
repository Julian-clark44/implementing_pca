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

	// V is the eigenvectors of X^{T} * X
	Matrix3f XTX = X.transpose() * X;
	EigenSolver<Matrix3f> eigen_solver(XTX);
	EigenSolver<Matrix3f>::EigenvalueType eigen_values =
		eigen_solver.eigenvalues();
	EigenSolver<Matrix3f>::EigenvectorsType eigen_vectors =
		eigen_solver.eigenvectors();

	// Reduce V to 2 dimensions.
	std::vector<std::pair<float, Vector3f>> eigen_vals_and_vecs;
	for (int i = 0; i < 3; i++) {
		eigen_vals_and_vecs.push_back(
				std::make_pair(eigen_values(i).real(), eigen_vectors.col(i).real()));
	}
	std::sort(eigen_vals_and_vecs.begin(), eigen_vals_and_vecs.end(),
		[&](std::pair<float, Vector3f>& a, std::pair<float, Vector3f>& b) -> bool {
			return a.first > b.first;
		});
	Matrix<float, 3, 2> V;
	V.col(0) = eigen_vals_and_vecs[0].second;
	V.col(1) = eigen_vals_and_vecs[1].second;

	// Calculate sigma.
	Matrix2f sigma = Matrix2f::Zero();
	for (int i = 0; i < 2; i++) {
		sigma(i, i) = std::sqrt(eigen_vals_and_vecs[i].first);
	}

	// Compress X to Y.
	Matrix<float, 2, 3> Y;
	Y = sigma * V.transpose();

	// Reconstruct X_hat.
	Matrix3f X_hat = X * V * V.transpose();
	
	const auto t2 = std::chrono::steady_clock::now();
	const std::chrono::duration<double, std::milli> dt = t2 - t1;

	std::cout << "Original X:\n" << X << std::endl;
	std::cout << "X^{T}*X Eigen Vectors (V):\n" << eigen_vectors.real() << std::endl;
	std::cout << "X^{T}*X Eigen Values:\n" << eigen_values.real() << std::endl;
	std::cout << "V_d:\n" << V << std::endl;
	std::cout << "Y:\n" << Y << std::endl;
	std::cout << "Reconstructed X_hat:\n" << X_hat << std::endl;
	std::cout << "X - X_hat:\n" << X - X_hat << std::endl;
	std::cout << "Runtime: " << dt.count() << "ms" << std::endl;
}
