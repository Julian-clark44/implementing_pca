#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace Eigen;

int main(int argc, char **argv) {
	// Ensure proper usage.
	if (argc != 3) {
		std::cerr << "Usage: ./imageCompression <image_path> <\% of columns to use in (0, 1]>"
			<< std::endl;
		return 1;
	}

	// Read the image file.
	Matrix<unsigned char, Dynamic, Dynamic> eigen_image;
	try {
		cv::cv2eigen(std::move(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)), eigen_image);
	} catch (const std::exception& e) {
		std::cerr << "Failed to read image file." << std::endl;
		return 1;
	}
	
	// Get the scaling factor.
	float modifier;
	try {
		modifier = std::stof(argv[2]);
	} catch (const std::exception& e) {
		std::cerr << "Failed to read read \% of columns to use." << std::endl;
		return 1;
	}

	// Calculate the SVD.
	MatrixXf eigen_image_float = std::move(eigen_image).cast<float>();
	BDCSVD<Matrix<float, Dynamic, Dynamic>> svd(eigen_image_float, ComputeThinV);
	MatrixXf V = svd.matrixV();

	// Resize V.
	int rows = V.rows();
	int cols = V.cols();
	int d = modifier * cols;
	V.conservativeResize(rows, d);

	// Reconstruct the image from the projection.
	Matrix<unsigned char, Dynamic, Dynamic> eigen_image_reconstructed
		= (std::move(eigen_image_float) * V * V.transpose()).cast<unsigned char>();
	cv::Mat cv_image;
	cv::eigen2cv(std::move(eigen_image_reconstructed), cv_image);
	cv::imshow("reconstructed image", cv_image);
	cv::waitKey(0);
}
